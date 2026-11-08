"""Research agent: runs the RLM subgraph for deep investigation and analytics questions.

The subgraph's evidence (every chunk the plan fetched) becomes `retrieved`, so the response agent
cites real chunks and the validator can check them. The permission check is repeated here at
execution time even though routing already checked it: a routing bug or a manipulated decision
still cannot run the Python analysis tool for a role that may not use it.
"""

from langgraph.config import get_config

from app.agents.events import emit
from app.agents.nodes.common import instrumented, last_user_text
from app.agents.state import AgentState, ResearchOutcome
from app.core.errors import AuthorizationError
from app.rlm.graph import research_graph
from app.security.rbac import can_use

MAX_EVIDENCE = 40  # chunks passed on to the response agent


def _fallback(_state: AgentState, _exc: Exception) -> dict:
    return {"retrieved": [], "retrieval_notes": ["Deep analysis failed for this question."]}


@instrumented("research", fallback=_fallback)
async def research_node(state: AgentState) -> dict:
    user = state["user"]
    if not can_use(user.role, "python_analysis"):
        raise AuthorizationError(f"role {user.role} may not use python_analysis")

    route = state.get("route")
    question = last_user_text(state)
    if route and route.search_query and route.search_query != question:
        question = f"{question}\n(Context from the conversation: {route.search_query})"

    # Parent config passed explicitly so the subgraph's runs nest under this node in LangSmith
    # and its custom events reach the parent stream.
    out = await research_graph.ainvoke(
        {"question": question, "user": user, "depth": 0}, config=get_config()
    )
    evidence = out.get("evidence") or {}
    synthesis = out.get("synthesis")
    findings = out.get("findings") or []

    # Cited evidence first, so the response agent sees the chunks the findings rely on.
    cited = [cid for f in findings for cid in f.evidence]
    ordered = list(dict.fromkeys([*cited, *evidence]))[:MAX_EVIDENCE]
    outcome = ResearchOutcome(
        synthesis=synthesis.summary if synthesis else "",
        themes=synthesis.recurring_themes if synthesis else [],
        findings=[f.model_dump() for f in findings],
        data=out.get("data") or {},
        documents=out.get("documents") or [],
        plan_code=out.get("code") or "",
        depth=out.get("depth", 0),
    )
    emit(
        "research", node="research", stage="done", documents=outcome.documents, depth=outcome.depth
    )
    return {"retrieved": [evidence[cid] for cid in ordered], "research": outcome}
