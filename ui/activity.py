"""Pure event → view-model reducers for the agent activity panel.

Keeping this free of Streamlit means the whole panel (titles, tables, timings, retries)
is deterministic and unit-testable against canned event sequences.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ui.client import StreamEvent

_NODE_TITLES = {
    "load_memory": "📥 Loading memory",
    "supervisor": "🧭 Supervisor routing",
    "retrieval": "🔎 Retrieval",
    "research": "🔬 Research",
    "response": "✍️ Writing answer",
    "validator": "🛡️ Validating",
    "finalize": "🏁 Finalizing",
    "update_memory": "💾 Updating memory",
    "rlm_plan": "🧠 Planning RLM search",
    "rlm_execute": "⚡ Executing sandboxed plan",
    "rlm_reduce": "🔗 Reducing findings",
}


@dataclass
class Step:
    """One human-readable line in the activity panel."""

    node: str
    icon: str
    title: str
    detail: str | None = None
    status: str = "running"
    code: str | None = None
    table: list[dict[str, Any]] | None = None


@dataclass
class TurnState:
    """Everything the UI needs to render one chat turn, accumulated from SSE events."""

    answer_text: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    thread_id: str | None = None
    done: bool = False
    research_plan: str | None = None
    hits: list[dict[str, Any]] = field(default_factory=list)
    node_timings: dict[str, float] = field(default_factory=dict)
    # Approval payload (tool, args, action, thread_id…) from the `approval` event; the UI
    # shows Approve/Reject buttons while this is set on the last turn.
    pending_approval: dict[str, Any] | None = None


def apply(state: TurnState, event: StreamEvent) -> TurnState:
    """Fold one SSE event into the turn state; never raises on missing fields.

    Mutates and returns `state` so callers can do `state = apply(state, ev)`.
    """
    data = event.data if isinstance(event.data, dict) else {}
    node = event.node or event.type
    handler = _HANDLERS.get(event.type)
    if handler is None:
        _handle_unknown(state, event.type, node, data)
    else:
        handler(state, node, data)
    return state


def render_citations(citations: list[dict[str, Any]]) -> str:
    """Render citations as a markdown list for the Sources expander."""
    lines: list[str] = []
    for citation in citations:
        if not isinstance(citation, dict):
            lines.append(f"- **[unknown]** {citation!r}")
            continue
        title = citation.get("title") or citation.get("doc_id") or ""
        line = f"- **[{citation.get('id', '?')}]** {title}"
        section = citation.get("section")
        if section:
            line += f" — {section}"
        lines.append(line)
    return "\n".join(lines)


def format_status_badge(role: str) -> str:
    """Short role label for the sidebar; roles come straight from the JWT."""
    badges = {"viewer": "👤 Viewer", "analyst": "📊 Analyst", "admin": "👑 Admin"}
    return badges.get(role, f"👤 {role.replace('_', ' ').capitalize()}")


def _handle_node_start(state: TurnState, node: str, data: dict[str, Any]) -> None:
    # A retried response node restarts the stream, so earlier partial text is obsolete.
    if node == "response":
        state.answer_text = ""
    if node.startswith("rlm_batch_"):
        title = f"🧩 Batch {node.removeprefix('rlm_batch_')}"
    else:
        title = _NODE_TITLES.get(node, f"▶️ {node.replace('_', ' ').title()}")
    state.steps.append(Step(node=node, icon=title.split()[0], title=title, status="running"))


def _handle_node_end(state: TurnState, node: str, data: dict[str, Any]) -> None:
    status = str(data.get("status") or "ok")
    duration = data.get("duration_ms")
    if isinstance(duration, int | float):
        state.node_timings[node] = float(duration)
    # Prefer the step opened by node_start (the latest still-running one), so duration
    # is not stamped onto a step that a specific event already finalised.
    for step in reversed(state.steps):
        if step.node == node and step.status == "running":
            step.status = status
            if isinstance(duration, int | float):
                step.title += f" · {duration:.0f} ms"
            return
    for step in reversed(state.steps):
        if step.node == node:
            step.status = status
            return


def _handle_memory(state: TurnState, node: str, data: dict[str, Any]) -> None:
    action = str(data.get("action") or "memory")
    if action == "loaded":
        title = "🧠 Memory: loaded context"
        detail = f"{data.get('previous_turns', 0)} previous turns"
        summary = "summary present" if data.get("has_summary") else "no summary"
        detail += f", {summary}"
    elif action == "saved":
        title = "💾 Memory: saved turn"
        detail = f"{data.get('messages', 0)} messages stored"
    elif action == "summarised":
        title = "📝 Memory: summarised"
        detail = json.dumps(data)[:300]
    else:
        title = f"🧠 Memory: {action}"
        detail = json.dumps(data)[:300]
    state.steps.append(Step(node=node, icon="🧠", title=title, detail=detail, status="ok"))


def _handle_route(state: TurnState, node: str, data: dict[str, Any]) -> None:
    agents = ", ".join(str(a) for a in data.get("agents") or ["(no agent)"])
    title = f"🧭 Routed: {data.get('intent', 'chat')} → {agents}"
    if data.get("degraded"):
        title += " (degraded)"
    details = [str(data.get("reasoning") or "")]
    query = data.get("search_query")
    if query:
        details.append(f"query → {query}")
    state.steps.append(
        Step(node=node, icon="🧭", title=title, detail=" ".join(details), status="ok")
    )


def _handle_retrieval(state: TurnState, node: str, data: dict[str, Any]) -> None:
    status = str(data.get("status") or "")
    hits = [h for h in data.get("hits") or [] if isinstance(h, dict)]
    if status == "searching":
        departments = ", ".join(str(d) for d in data.get("departments") or [])
        clearance = str(data.get("clearance") or "-")
        scope = departments or "all departments"
        title = f"🔎 Searching {scope} (clearance: {clearance})"
        state.steps.append(Step(node=node, icon="🔎", title=title, status="running"))
        return
    if status == "degraded":
        notes = ", ".join(str(n) for n in data.get("notes") or [])
        title = f"⚠️ Degraded retrieval: {notes or 'partial results'}"
    else:
        title = f"📚 {len(hits)} chunks (hybrid)"
    state.hits = hits
    state.steps.append(
        Step(
            node=node,
            icon=title.split()[0],
            title=title,
            detail=", ".join(str(n) for n in data.get("notes") or []) or None,
            status="degraded" if status == "degraded" else "ok",
            table=[_hit_row(hit) for hit in hits] or None,
        )
    )


def _hit_row(hit: dict[str, Any]) -> dict[str, Any]:
    """Flatten a retrieval hit into a dataframe row with scores rounded for display."""
    return {
        "citation": hit.get("citation", ""),
        "title": hit.get("title", ""),
        "access": hit.get("access_level", ""),
        "dense": _round3(hit.get("dense")),
        "sparse": _round3(hit.get("sparse")),
        "fused": _round3(hit.get("fused")),
        "source": hit.get("source", ""),
    }


def _handle_tool_call(state: TurnState, node: str, data: dict[str, Any]) -> None:
    tool = str(data.get("tool") or "tool")
    status = str(data.get("status") or "running")
    titles = {
        "running": f"⏳ {tool} running",
        "ok": f"✅ {tool}",
        "failed": f"💥 {tool} failed",
        "denied": f"⛔ {tool} denied",
        "pending_approval": f"⏸️ {tool} awaiting approval",
        "rejected": f"🚫 {tool} rejected",
    }
    title = titles.get(status, f"⚙️ {tool} ({status})")
    detail = str(data.get("reason") or data.get("error") or "") or None
    if status == "pending_approval":
        # The action sentence (not raw args) belongs in the panel; full args show via the
        # `approval` event / its step.
        detail = str(data.get("action") or data.get("why") or "") or None
    table = None
    if status == "ok":
        calls = [c for c in data.get("primitive_calls") or [] if isinstance(c, dict)]
        table = [
            {
                "primitive": str(c.get("name", "")),
                "args": _snip(json.dumps(c.get("kwargs", {}), default=str), 80),
                "ms": _round3(c.get("duration_ms")),
                "ok": bool(c.get("ok")),
            }
            for c in calls
        ] or None
    elif status == "failed":
        detail = detail or str(data.get("error") or "unknown error")
    state.steps.append(
        Step(
            node=node,
            icon=title.split()[0],
            title=title,
            detail=detail,
            status="failed" if status == "failed" else status,
            table=table,
        )
    )


def _handle_research(state: TurnState, node: str, data: dict[str, Any]) -> None:
    stage = str(data.get("stage") or "")
    if stage == "plan":
        code = str(data.get("code") or "")
        state.research_plan = code
        state.steps.append(
            Step(
                node=node,
                icon="🧠",
                title=f"🧠 RLM plan (depth {data.get('depth', '?')})",
                detail=None,
                status="ok",
                code=code or None,
            )
        )
    elif stage == "batches":
        documents = data.get("documents") or []
        sizes = data.get("batch_sizes") or []
        title = f"🧩 {len(sizes)} batches over {len(documents)} documents"
        state.steps.append(
            Step(
                node=node,
                icon="🧩",
                title=title,
                status="ok",
                detail=", ".join(str(d) for d in documents) or None,
            )
        )
    elif stage == "batch_done":
        findings = [f for f in data.get("findings") or [] if isinstance(f, dict)]
        title = f"✅ Batch {data.get('index', '?')}: {len(findings)} findings"
        detail = "; ".join(f"{f.get('doc_id') or '?'} — {f.get('category', '?')}" for f in findings)
        state.steps.append(
            Step(node=node, icon="✅", title=title, detail=detail or None, status="ok")
        )
    elif stage == "reduce":
        themes = [str(t) for t in data.get("themes") or []]
        title = f"🔗 Reduce: {', '.join(themes) or 'no themes'}"
        follow_up = data.get("follow_up")
        detail = f"follow-up: {follow_up}" if follow_up else None
        state.steps.append(Step(node=node, icon="🔗", title=title, detail=detail, status="ok"))
    elif stage == "done":
        documents = len(data.get("documents") or [])
        title = f"🔬 Research complete ({documents} documents, depth {data.get('depth', '?')})"
        state.steps.append(Step(node=node, icon="🔬", title=title, status="ok"))
    else:
        _handle_unknown(state, node, data)


def _handle_validation(state: TurnState, node: str, data: dict[str, Any]) -> None:
    valid = bool(data.get("valid"))
    if valid:
        title = f"🛡️ Validation passed ({len(data.get('cited') or [])} citations)"
        status = "ok"
    else:
        title = "🚧 Validation failed"
        status = "error"
    details: list[str] = []
    unknown = data.get("unknown") or []
    if unknown:
        details.append(f"unknown citations: {', '.join(str(u) for u in unknown)}")
    for issue in data.get("issues") or []:
        details.append(str(issue))
    state.steps.append(
        Step(
            node=node,
            icon="🛡️" if valid else "🚧",
            title=title,
            detail="; ".join(details) or None,
            status=status,
        )
    )


def _handle_guardrail(state: TurnState, node: str, data: dict[str, Any]) -> None:
    """Typed guardrail events; the `stage`-free shape stays as the generic fallback."""
    stage = str(data.get("stage") or "")
    if stage == "input":
        if bool(data.get("allowed")):
            title = "🛡️ Input guard: allowed"
            status = "ok"
        else:
            category = data.get("category", "?")
            title = f"🛡️ Input guard: blocked ({category}, {data.get('layer', '?')})"
            status = "error"
        state.steps.append(Step(node=node, icon="🛡️", title=title, status=status))
        return
    if stage == "retrieved_text":
        chunks = [c for c in data.get("chunks") or []]
        title = f"🧼 Sanitised {len(chunks)} retrieved chunk(s)"
        findings = ", ".join(str(f) for f in data.get("findings") or [])
        state.steps.append(
            Step(node=node, icon="🧼", title=title, detail=findings or None, status="ok")
        )
        return
    if stage == "output":
        issues = [i for i in data.get("issues") or [] if isinstance(i, dict)]
        codes = ", ".join(str(i.get("code") or "?") for i in issues) or "no issues"
        state.steps.append(Step(node=node, icon="🛡️", title=f"🛡️ Output guard: {codes}", status="ok"))
        return
    action = str(data.get("action") or "intervened")
    state.steps.append(
        Step(
            node=node,
            icon="🛡️",
            title=f"🛡️ Guardrail: {action}",
            detail=_snip(json.dumps(data, default=str), 300),
            status="ok",
        )
    )


def _handle_approval(state: TurnState, node: str, data: dict[str, Any]) -> None:
    """The backend could not finish the turn without a human decision; hold the request."""
    action = str(data.get("action") or "requested tool action")
    tool = str(data.get("tool") or "tool")
    try:
        args_json = json.dumps(data.get("args") or {}, default=str)
    except (TypeError, ValueError):
        args_json = str(data.get("args") or {})
    state.steps.append(
        Step(
            node=node,
            icon="⏸️",
            title=f"⏸️ Approval required: {action}",
            detail=f"{tool} {_snip(args_json, 200)}",
            status="ok",
        )
    )
    state.pending_approval = dict(data)


def _handle_token(state: TurnState, node: str, data: dict[str, Any]) -> None:
    state.answer_text += str(data.get("text") or "")


def _handle_answer(state: TurnState, node: str, data: dict[str, Any]) -> None:
    # The answer event is authoritative: it replaces any streamed (possibly retried) text.
    state.answer_text = str(data.get("text") or "")
    citations = data.get("citations")
    state.citations = [c for c in citations or [] if isinstance(c, dict)]
    notes = data.get("notes")
    state.notes = [str(n) for n in notes or []]
    for error in data.get("errors") or []:
        if isinstance(error, dict):
            state.errors.append(dict(error))


def _handle_error(state: TurnState, node: str, data: dict[str, Any]) -> None:
    message = str(data.get("message") or "unknown error")
    state.steps.append(
        Step(
            node=node,
            icon="❌",
            title=f"❌ {message}",
            detail=str(data.get("code") or "") or None,
            status="error",
        )
    )
    state.errors.append({"code": data.get("code"), "message": message})


def _handle_done(state: TurnState, node: str, data: dict[str, Any]) -> None:
    thread_id = data.get("thread_id")
    state.thread_id = thread_id if isinstance(thread_id, str) else None
    state.done = True


def _handle_unknown(state: TurnState, event_type: str, node: str, data: dict[str, Any]) -> None:
    """Forward compatibility: future event types degrade into a generic, truncated step."""
    state.steps.append(
        Step(
            node=node,
            icon="❔",
            title=f"❔ {event_type}",
            detail=_snip(json.dumps(data, default=str, sort_keys=True), 300),
            status="ok",
        )
    )


def _round3(value: Any) -> float | None:
    if isinstance(value, int | float):
        return round(float(value), 3)
    return None


def _snip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


_HANDLERS = {
    "node_start": _handle_node_start,
    "node_end": _handle_node_end,
    "memory": _handle_memory,
    "route": _handle_route,
    "retrieval": _handle_retrieval,
    "tool_call": _handle_tool_call,
    "research": _handle_research,
    "validation": _handle_validation,
    "guardrail": _handle_guardrail,
    "approval": _handle_approval,
    "token": _handle_token,
    "answer": _handle_answer,
    "error": _handle_error,
    "done": _handle_done,
}
