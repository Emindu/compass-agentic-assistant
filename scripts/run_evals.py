"""LangSmith eval harness over evals/ground_truth.json: the real graph, deterministic scoring.

Usage:
    uv run python -m scripts.run_evals [--dataset assignment-evals] [--include-rlm]
        [--dry-run] [--concurrency 2] [--only N]

Why code-based evaluators (evals/evaluators.py): they are deterministic, free and
reproducible, and each maps to a grading point in the assignment brief — citations, RBAC
filtering, prompt-injection handling, version conflicts and RLM coverage — instead of
delegating to a judge model whose own drift would make runs incomparable.

Why deterministic example ids (uuid5 over question+role): every re-run syncs the dataset
by upserting, not appending — new cases are created, changed ground truth is updated via
``client.update_example``, untouched rows keep their id so experiments stay comparable.

Why the graph in-process instead of the HTTP API: the harness owns graph state directly —
cited chunk ids, the validation verdict, retrieval doc ids, per-node errors — which the SSE
stream never exposes; no rate limiter applies; and every run still lands in LangSmith via
the same ``run_config`` the API uses, so eval traces sit next to production traffic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langsmith import Client, aevaluate
from langsmith.schemas import Example

from app.agents.graph import build_graph
from app.api.routes.chat import run_config
from app.core.config import get_settings
from app.security.rbac import Role, UserContext
from evals.evaluators import ALL_EVALUATORS, RLM_EVALUATORS, pass_threshold

REPO_ROOT = Path(__file__).parents[1]
GROUND_TRUTH_PATH = REPO_ROOT / "evals" / "ground_truth.json"

# A namespaced uuid5 domain, so example ids cannot collide with ids minted from the stock
# NAMESPACE_URL by unrelated tooling.
_EXAMPLE_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "compass-eval")

# Demo identities mirror app/api/auth.py. UserContext is built directly, without a password
# or a JWT: this is a trusted offline runner inside the repo, not a public endpoint, and RBAC
# is enforced against the role each case requests, exactly as a verified token would supply.
_DEMO_IDENTITIES: dict[str, tuple[str, str, str]] = {
    "viewer": ("vera", "Vera Viewer", "retail"),
    "analyst": ("anil", "Anil Analyst", "payments"),
    "admin": ("ada", "Ada Admin", "platform"),
}


# --- Examples ---


def example_id(question: str, role: str) -> str:
    """Stable id per question+role, so dataset sync is an upsert instead of duplication."""
    return str(uuid.uuid5(_EXAMPLE_NAMESPACE, f"{role}|{question}"))


def build_examples(include_rlm: bool) -> list[dict[str, Any]]:
    """One LangSmith example per ground-truth case; reference outputs carry the grader's data."""
    truth = json.loads(GROUND_TRUTH_PATH.read_text())
    examples = [
        {
            "id": example_id(case["question"], case["role"]),
            "inputs": {"question": case["question"], "role": case["role"]},
            "reference_outputs": {k: v for k, v in case.items() if k not in ("question", "role")},
        }
        for case in truth["qa"]
    ]
    if include_rlm:
        rlm = dict(truth["rlm_payment_outages"])
        question = rlm.pop("question")
        examples.append(
            {
                "id": example_id(question, "analyst"),
                "inputs": {"question": question, "role": "analyst"},
                "reference_outputs": rlm,
            }
        )
    return examples


def sync_dataset(
    client: Client, dataset_name: str, examples: list[dict[str, Any]]
) -> list[Example]:
    """Bring the LangSmith dataset in line with ground truth, row by row.

    Kept small and sync on purpose: it runs once per invocation, before the async eval.
    Existing rows whose reference outputs changed are updated; identical rows stay
    untouched, so LangSmith does not mint a new dataset version for a no-op run.
    """
    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(
            dataset_name=dataset_name,
            description="Compass eval cases from evals/ground_truth.json",
        )
    existing = {str(ex.id): ex for ex in client.list_examples(dataset_name=dataset_name)}
    synced: list[Example] = []
    for example in examples:
        current = existing.get(example["id"])
        if current is None:
            synced.append(
                client.create_example(
                    inputs=example["inputs"],
                    reference_outputs=example["reference_outputs"],
                    dataset_name=dataset_name,
                    example_id=example["id"],
                )
            )
        elif (current.outputs or {}) != example["reference_outputs"]:
            client.update_example(current.id, outputs=example["reference_outputs"])
            synced.append(current)
        else:
            synced.append(current)
    return synced


# --- Target: the real graph, in-process ---


@lru_cache(maxsize=1)
def _compiled_graph() -> CompiledStateGraph:
    """One graph per process, like the API: ainvoke is safe concurrently on the same object,
    and per-case rebuilds would re-open clients and buy nothing."""
    return build_graph(InMemorySaver())


def _demo_user(role: str) -> UserContext:
    user_id, name, department = _DEMO_IDENTITIES[role]
    return UserContext(user_id=user_id, name=name, role=Role(role), department=department)


def _final_answer(state: dict[str, Any]) -> str:
    """The finalize node commits the answer as the thread's last AI message, so that is the
    authoritative text; the draft is only a fallback for paths that never reached finalize."""
    for message in reversed(state.get("messages") or []):
        if getattr(message, "type", None) == "ai":
            return str(message.content)
    draft = state.get("draft_answer")
    return str(draft) if draft else ""


def _summarise(state: dict[str, Any]) -> dict[str, Any]:
    """Flatten the finished turn into exactly the fields evaluators look at."""
    validation = state.get("validation")
    route = state.get("route")
    research = state.get("research")
    # Doc id is the part of a chunk id before the '#' (e.g. RB-PAY-003-V2 in RB-PAY-003-V2#03)
    cited_doc_ids = sorted(
        {c.split("#", 1)[0] for c in (validation.cited_ids if validation else [])}
    )
    return {
        "answer": _final_answer(state),
        "route": route.intent if route is not None else None,
        "blocked": state.get("guard") is not None and not state["guard"].allowed,
        "cited_docs": cited_doc_ids,
        "retrieved_docs": sorted({c.doc_id for c in state.get("retrieved") or []}),
        "research_documents": sorted(set(research.documents)) if research is not None else [],
        "citations_valid": validation.valid if validation is not None else None,
        "errors": [f"{e.node}:{e.code}" for e in state.get("errors") or []],
        "interrupted": False,
    }


async def run_case(inputs: dict[str, Any]) -> dict[str, Any]:
    """Eval target: one fresh thread through the real graph as the case's demo user."""
    user = _demo_user(inputs["role"])
    thread_id = f"eval-{uuid.uuid4().hex[:12]}"  # fresh per case: eval turns never share memory
    graph = _compiled_graph()
    config = run_config(user, thread_id)
    config["tags"] = [*config["tags"], "eval"]
    state = await graph.ainvoke(
        {"messages": [HumanMessage(inputs["question"])], "user": user}, config
    )
    result = _summarise(state)
    if (await graph.aget_state(config)).next:
        # Paused on an admin-approval interrupt: never resume inside an eval — the harness
        # is not a human approver, so report the pause instead of shipping a half turn.
        result.update(answer="", interrupted=True, errors=[*result["errors"], "awaiting_approval"])
    return result


# --- Scoring and reporting (shared by --dry-run and the LangSmith run) ---


def evaluate_locally(
    inputs: dict[str, Any], outputs: dict[str, Any], reference: dict[str, Any]
) -> list[dict[str, Any]]:
    """Run the evaluators in-process; skipped evaluators contribute no row."""
    evaluators = [*ALL_EVALUATORS, *(RLM_EVALUATORS if reference.get("in_scope_incidents") else [])]
    scored: list[dict[str, Any]] = []
    for evaluator in evaluators:
        result = evaluator(inputs, outputs, reference)
        if "key" in result:
            scored.append(
                {
                    "key": result["key"],
                    "score": float(result["score"]),
                    "comment": result.get("comment") or "",
                }
            )
    return scored


def _clip(text: str, width: int = 36) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def render_table(rows: list[dict[str, Any]]) -> list[str]:
    """Plain text table (case, role, one column per evaluator); missing scores show as '-'."""
    keys: list[str] = []
    for row in rows:
        for score in row["scores"]:
            if score["key"] not in keys:
                keys.append(score["key"])
    header = ["case", "role", *keys]
    table = [header]
    for row in rows:
        scored = {score["key"]: f"{score['score']:g}" for score in row["scores"]}
        table.append([_clip(row["case"]), row["role"], *[scored.get(key, "-") for key in keys]])
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    return [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in table
    ]


def _failures(rows: list[dict[str, Any]]) -> list[str]:
    """One line per evaluator below its threshold, with the evaluator's comment."""
    return [
        f"[{row['role']}] {_clip(row['case'], 60)}: "
        f"{score['key']}={score['score']:g} ({score['comment']})"
        for row in rows
        for score in row["scores"]
        if score["score"] < pass_threshold(score["key"])
    ]


def _print_report(rows: list[dict[str, Any]]) -> list[str]:
    for line in render_table(rows):
        print(line)
    failures = _failures(rows)
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("\nAll applicable evaluator scores reached their threshold.")
    return failures


async def dry_run(examples: list[dict[str, Any]]) -> int:
    """No LangSmith at all: the target and the scorers run locally, results print and gate."""
    print(f"Dry run over {len(examples)} case(s) — no LangSmith calls are made.\n")
    rows: list[dict[str, Any]] = []
    for example in examples:
        outputs = await run_case(example["inputs"])
        rows.append(
            {
                "case": example["inputs"]["question"],
                "role": example["inputs"]["role"],
                "scores": evaluate_locally(
                    example["inputs"], outputs, example["reference_outputs"]
                ),
            }
        )
    return 1 if _print_report(rows) else 0


async def langsmith_run(args: argparse.Namespace) -> int:
    """Sync the dataset, then aevaluate; print the experiment URL and the summary table."""
    client = Client()
    all_examples = build_examples(include_rlm=args.include_rlm)
    focused = all_examples[: args.only] if args.only else all_examples
    synced = sync_dataset(client, args.dataset, all_examples)
    by_id = {str(example.id): example for example in synced}
    results = await aevaluate(
        run_case,
        data=[by_id[example["id"]] for example in focused],
        evaluators=[*ALL_EVALUATORS, *(RLM_EVALUATORS if args.include_rlm else [])],
        experiment_prefix="assignment",
        max_concurrency=args.concurrency,
        metadata={"git_sha": git_sha(), "rerank_enabled": get_settings().rerank_enabled},
        client=client,
    )
    rows: list[dict[str, Any]] = []
    async for row in results:
        rows.append(
            {
                "case": row["example"].inputs["question"],
                "role": row["example"].inputs["role"],
                "scores": [
                    {"key": fr.key, "score": float(fr.score), "comment": fr.comment or ""}
                    for fr in row["evaluation_results"]["results"]
                    if fr.score is not None
                ],
            }
        )
    print(f"Experiment: {results.url}\n")
    return 1 if _print_report(rows) else 0


def git_sha() -> str:
    """Read .git plumbing directly: no subprocess, no dependency, works in a read-only repo."""
    head = (REPO_ROOT / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = REPO_ROOT / ".git" / head.removeprefix("ref: ").strip()
        if ref.is_file():
            return ref.read_text().strip()
    return head  # detached HEAD already stores the sha itself


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Compass eval harness on LangSmith.")
    parser.add_argument("--dataset", default="assignment-evals", help="LangSmith dataset name")
    parser.add_argument(
        "--include-rlm",
        action="store_true",
        help="add the RLM payment-outage research case",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run every case locally, print scores, never contact LangSmith",
    )
    parser.add_argument(
        "--concurrency", type=int, default=2, help="max parallel cases in aevaluate"
    )
    parser.add_argument(
        "--only",
        type=int,
        default=None,
        help="run only the first N cases (dataset sync still covers every case)",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()  # LangSmith and provider SDKs read os.environ, not app Settings
    if args.dry_run:
        # Belt and braces for "no LangSmith calls at all": the harness itself never talks
        # to LangSmith in this mode, and this stops the graph's auto-instrumentation from
        # uploading traces either. The env var is read per run, not at import time.
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGSMITH_TRACING_ENABLED"] = "false"
        examples = build_examples(include_rlm=args.include_rlm)
        if args.only:
            examples = examples[: args.only]
        return await dry_run(examples)
    return await langsmith_run(args)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
