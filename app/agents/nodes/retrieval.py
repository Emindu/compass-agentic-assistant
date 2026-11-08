"""Retrieval agent: hybrid search scoped by the supervisor and bounded by the user's clearance.

The supervisor chooses *what* to search (query, departments); the user's role decides *what may
be seen* (clearance from the JWT). The two never mix, so a prompt cannot widen access.
"""

from app.agents.events import emit
from app.agents.nodes.common import instrumented, last_user_text
from app.agents.state import AgentState
from app.retrieval.hybrid import hybrid_search
from app.retrieval.models import ALL_DEPARTMENTS, RetrievalQuery

TOP_K = 8


def _fallback(_state: AgentState, _exc: Exception) -> dict:
    return {"retrieved": [], "retrieval_notes": ["Document search failed for this question."]}


@instrumented("retrieval", fallback=_fallback)
async def retrieval_node(state: AgentState) -> dict:
    user = state["user"]
    route = state.get("route")
    query = RetrievalQuery(
        text=(route.search_query if route else "") or last_user_text(state),
        clearance=user.clearance,
        departments=(route.departments if route and route.departments else list(ALL_DEPARTMENTS)),
        top_k=TOP_K,
    )
    emit(
        "retrieval",
        node="retrieval",
        status="searching",
        query=query.text,
        departments=query.departments,
        clearance=query.clearance.name,
    )
    result = await hybrid_search(query)
    emit(
        "retrieval",
        node="retrieval",
        status="degraded" if result.degraded else "done",
        notes=result.notes,
        timings_ms=result.timings_ms,
        hits=[
            {
                "id": c.id,
                "citation": c.citation,
                "title": c.title,
                "access_level": c.access_level,
                "dense": c.dense_score,
                "sparse": c.sparse_score,
                "fused": c.fused_score,
                "source": c.source,
            }
            for c in result.chunks
        ],
    )
    return {"retrieved": result.chunks, "retrieval_notes": result.notes}
