"""Input guard node: first line of defence, before any agent sees the message.

Why a node (rather than a middleware call): the guard runs inside the graph, so its verdict
is part of turn state (visible to every downstream agent), its decision is a typed branch
(`route_after_guard`), and its result shows in the activity panel like any other step.
"""

from app.agents.events import emit
from app.agents.nodes.common import instrumented, last_user_text
from app.agents.state import AgentState
from app.security.input_guard import GuardVerdict, check_input, check_patterns, refusal_message


def _apply(state: AgentState, verdict: GuardVerdict) -> dict:
    """Shared verdict handling: always emit the guardrail; on block, prepare the refusal."""
    emit(
        "guardrail",
        node="input_guard",
        stage="input",
        allowed=verdict.allowed,
        category=verdict.category,
        layer=verdict.layer,
        matched=verdict.matched,
    )
    if verdict.allowed:
        return {"guard": verdict}
    # Blocked turns never reach the supervisor: the refusal is already the answer, and
    # `route=None` keeps the finalize path from expecting evidence or citations.
    return {"guard": verdict, "draft_answer": refusal_message(verdict), "route": None}


def _fallback(state: AgentState, _exc: Exception) -> dict:
    """Fail closed only for pattern hits: if the guard itself errors, re-run the cheap
    deterministic layer (no model) and continue when it passes, so a broken classifier
    never takes the whole assistant out of service."""
    verdict = check_patterns(last_user_text(state))
    return _apply(state, verdict)


@instrumented("input_guard", fallback=_fallback)
async def input_guard_node(state: AgentState) -> dict:
    """Check the latest user message; blocked messages are turned into refusals here."""
    return _apply(state, await check_input(last_user_text(state)))


def route_after_guard(state: AgentState) -> str:
    """Blocked turns go straight to finalize (which emits the refusal); the rest continue."""
    guard = state.get("guard")
    return "finalize" if guard is not None and not guard.allowed else "supervisor"
