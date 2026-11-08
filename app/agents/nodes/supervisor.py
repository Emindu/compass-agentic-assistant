"""Supervisor agent: intent understanding, query rewriting and routing to specialists.

Uses structured output (a `RouteDecision`), so routing is a typed value the graph branches on,
not free text parsed with regexes. If the model is unavailable, the fallback routes to document
search with the raw question, so a supervisor failure degrades routing quality, not the answer.
"""

from langchain_core.messages import SystemMessage

from app.agents import prompts
from app.agents.events import emit
from app.agents.nodes.common import instrumented, last_user_text
from app.agents.state import AgentState, RouteDecision
from app.core.llm import ainvoke_structured

HISTORY_MESSAGES = 6  # recent turns given verbatim; older context comes from the summary


def _fallback(state: AgentState, _exc: Exception) -> dict:
    route = RouteDecision(
        intent="knowledge_qa",
        agents=["retrieval"],
        search_query=last_user_text(state),
        reasoning="Supervisor unavailable; defaulting to document search.",
    )
    emit("route", node="supervisor", **route.model_dump(), degraded=True)
    return {"route": route}


@instrumented("supervisor", fallback=_fallback)
async def supervisor_node(state: AgentState) -> dict:
    user = state["user"]
    system = prompts.SUPERVISOR.format(
        name=user.name,
        role=user.role.value,
        department=user.department,
        summary=state.get("summary") or "(none)",
    )
    history = state.get("messages", [])[-HISTORY_MESSAGES:]
    route = await ainvoke_structured("main", RouteDecision, [SystemMessage(system), *history])
    if route.intent in ("smalltalk", "out_of_scope"):
        route.agents = []
    elif not route.agents:
        route.agents = ["retrieval"]
    emit("route", node="supervisor", **route.model_dump())
    return {"route": route}


def route_after_supervisor(state: AgentState) -> str:
    """Branch on the typed decision. Research/analysis specialists join in the next block;
    until then they are served by retrieval so the graph never routes to a missing node."""
    route = state.get("route")
    if route is None or not route.agents:
        return "respond_direct"
    return "retrieval"
