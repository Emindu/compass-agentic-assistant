"""Supervisor agent: intent understanding, query rewriting and routing to specialists.

Uses structured output (a `RouteDecision`), so routing is a typed value the graph branches on,
not free text parsed with regexes. If the model is unavailable, the fallback routes to document
search with the raw question, so a supervisor failure degrades routing quality, not the answer.
"""

import re

from langchain_core.messages import SystemMessage

from app.agents import prompts
from app.agents.events import emit
from app.agents.nodes.common import instrumented, last_user_text
from app.agents.state import AgentState, RouteDecision
from app.core.llm import ainvoke_structured
from app.security.rbac import can_use

HISTORY_MESSAGES = 6  # recent turns given verbatim; older context comes from the summary


def _named_departments(departments: list | None, message: str) -> list | None:
    """Keep a department filter only if the employee actually named that department.

    The model tends to scope searches to the asker's own department (a retail viewer asking
    about the card switch got retail-only results), and a prompt rule alone did not stop it.
    Access control is enforced separately, so an unscoped search is always the safe default.
    """
    if not departments:
        return None
    text = message.lower()
    named = [d for d in departments if re.search(rf"\b{re.escape(d)}\b", text)]
    return named or None


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
    route.departments = _named_departments(route.departments, last_user_text(state))
    if route.intent in ("smalltalk", "out_of_scope"):
        route.agents = []
    elif not route.agents:
        route.agents = ["retrieval"]
    emit("route", node="supervisor", **route.model_dump())
    return {"route": route}


def route_after_supervisor(state: AgentState) -> str:
    """Branch on the typed decision, filtered by what the user's role may use.

    Deep research and analytics run the Python analysis tool (Analyst/Admin only). A Viewer
    asking for them is told so in the activity panel and gets a normal document search.
    The tools agent follows the same rule: directory lookups need "mcp" (else fall back to
    document search), admin actions need "admin" (else a direct refusal — no fallback exists).
    """
    route = state.get("route")
    if route is None or not route.agents:
        return "respond_direct"
    role = state["user"].role
    if {"research", "analysis"} & set(route.agents):
        if can_use(role, "python_analysis"):
            return "research"
        emit(
            "tool_call",
            node="supervisor",
            tool="python_analysis",
            status="denied",
            reason=f"Role '{role}' may not run analytics tools; using document search instead.",
        )
    if "tools" in route.agents:
        permission = "admin" if route.intent == "admin_action" else "mcp"
        if can_use(role, permission):
            return "tools"
        emit(
            "tool_call",
            node="supervisor",
            tool="tools",
            status="denied",
            reason=(
                f"Role '{role}' may not run admin actions; declining."
                if permission == "admin"
                else f"Role '{role}' may not use directory tools; using document search instead."
            ),
        )
        if route.intent == "admin_action":
            return "respond_direct"
    return "retrieval"
