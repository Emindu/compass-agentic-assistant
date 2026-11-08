"""Deterministic citation guardrail between the response agent and finalization.

Why validate citations at all: every id in the answer is checked against what was actually
retrieved, so a model cannot invent a source — the check is a set difference, no LLM involved,
which makes it free, fast and reproducible; an LLM evidence-quality judge can be layered on
later without changing this contract. Why a retry budget: invalidating an answer is only
useful if the response agent gets one chance to fix it with the validation issues as feedback.
After `MAX_RESPONSE_ATTEMPTS` we stop retrying and fall back to stripping the unverifiable
references, because an answer with honest caveats beats an endless loop or no answer at all.
"""

import re

from app.agents.events import emit
from app.agents.state import AgentState, ValidationResult
from app.core.logging import get_logger
from app.retrieval.models import RetrievedChunk

# One id like "INC-2026-0214#03"; a bracket holds a single id, or several ids separated by
# commas ("[A#01, B#02]"). Adjacent ids also arrive as separate brackets ("[A#01][B#02]");
# matching whole brackets (instead of bare ids) lets brackets that only contain prose
# ("[note]", "[1]") be ignored, and lets an unknown id be excised from a mixed bracket.
_CHUNK_ID = r"[A-Z]{2,5}-[A-Z0-9-]+#\d{2}"
CITATION_RE = re.compile(rf"\[\s*({_CHUNK_ID}(?:\s*,\s*{_CHUNK_ID})*)\s*\]")

# Phrases an honest "no evidence" answer uses; matched case-insensitively so paraphrasing the
# fallback in any register is accepted without an LLM.
_NO_INFO_PHRASES = (
    "couldn't find",
    "could not find",
    "no information",
    "not able to find",
    "don't have",
    "do not have",
)

MAX_RESPONSE_ATTEMPTS = 2

logger = get_logger(__name__)


def extract_citations(answer: str) -> list[str]:
    """Unique cited chunk ids in order of first appearance (order keeps the UI list stable)."""
    seen: list[str] = []
    for match in CITATION_RE.finditer(answer):
        for cid in (part.strip() for part in match.group(1).split(",")):
            if cid not in seen:
                seen.append(cid)
    return seen


def _admits_no_information(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _NO_INFO_PHRASES)


def validate_answer(
    answer: str, retrieved: list[RetrievedChunk], requires_evidence: bool
) -> ValidationResult:
    """Check the answer against the retrieval result; pure, no LLM, no side effects."""
    cited = extract_citations(answer)
    retrieved_ids = {c.id for c in retrieved}
    unknown = [cid for cid in cited if cid not in retrieved_ids]
    known = [cid for cid in cited if cid in retrieved_ids]

    issues: list[str] = []
    if not answer.strip():
        # An empty answer has nothing else worth reporting; one clear issue beats noise.
        issues.append("Answer is empty.")
    else:
        if unknown:
            issues.append(f"Cited references that were never retrieved: {', '.join(unknown)}.")
        if requires_evidence and not cited:
            if retrieved:
                issues.append(
                    "Evidence is required but the answer cites none of the retrieved chunks."
                )
            elif not _admits_no_information(answer):
                # Nothing was retrieved, so the only acceptable answer is a candid
                # "couldn't find" — anything else implies unsupported claims.
                issues.append(
                    "Evidence is required, nothing was retrieved, and the answer does not "
                    "say it found no information."
                )

    return ValidationResult(
        valid=not issues, cited_ids=known, unknown_citations=unknown, issues=issues
    )


def strip_unknown_citations(answer: str, unknown: list[str]) -> str:
    """Remove unknown ids from the answer, leaving valid citations untouched.

    Why per-bracket surgery instead of deleting substrings: a mixed bracket like "[A#01,
    FAKE#02]" must keep its real id, so each bracket is rebuilt from its surviving ids and
    dropped entirely when none survive — no empty "[]" or dangling ", " is left behind.
    """
    if not unknown:
        return answer
    unknown_set = set(unknown)

    out: list[str] = []
    last = 0
    for match in CITATION_RE.finditer(answer):
        ids = [part.strip() for part in match.group(1).split(",")]
        kept = [cid for cid in ids if cid not in unknown_set]
        if len(kept) == len(ids):
            continue  # nothing to strip in this bracket
        segment = answer[last : match.start()]
        last = match.end()
        if kept:
            out.append(segment)
            out.append("[" + ", ".join(kept) + "]")
        else:
            # Swallow the space before a fully removed bracket so "fact [FAKE#01]."
            # becomes "fact." rather than "fact .", and sentences read cleanly.
            if segment.endswith(" ") and answer[match.end() : match.end() + 1] in " .,;:!?":
                segment = segment[:-1]
            out.append(segment)
    out.append(answer[last:])
    return "".join(out)


async def validator_node(state: AgentState) -> dict:
    """Validate the draft answer and either admit it or prepare the stripped fallback."""
    answer = state.get("draft_answer") or ""
    retrieved = state.get("retrieved") or []
    route = state.get("route")
    requires_evidence = route is not None and bool(route.agents)
    attempts = state.get("response_attempts", 0)

    result = validate_answer(answer, retrieved, requires_evidence)
    emit(
        "validation",
        node="validator",
        valid=result.valid,
        cited=result.cited_ids,
        unknown=result.unknown_citations,
        issues=result.issues,
    )
    logger.info(
        "validation_result",
        valid=result.valid,
        cited=len(result.cited_ids),
        unknown=len(result.unknown_citations),
    )

    update: dict = {"validation": result}
    if not result.valid and attempts >= MAX_RESPONSE_ATTEMPTS and result.unknown_citations:
        # Retry budget exhausted: ship the answer minus the invented references, with an
        # explicit caveat, so the user keeps the substance instead of a dead end.
        update["draft_answer"] = (
            strip_unknown_citations(answer, result.unknown_citations)
            + "\n\n_Note: some references could not be verified and were removed._"
        )
    return update


def route_after_validation(state: AgentState) -> str:
    """Send back to the response agent while attempts remain, then finalize regardless."""
    validation = state.get("validation")
    invalid = validation is not None and not validation.valid
    if invalid and state.get("response_attempts", 0) < MAX_RESPONSE_ATTEMPTS:
        # The invalid ValidationResult is already in state, so the retry sees why it failed.
        return "response"
    return "finalize"
