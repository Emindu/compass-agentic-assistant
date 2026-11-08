"""The Compass agent graph.

    START → load_memory → supervisor ─┬─(needs documents)→ retrieval → response
                                      └─(smalltalk/out of scope)──────→ response
    response → validator ─┬─(invalid, retries left)→ response
                          └─(valid / out of retries)→ finalize → update_memory → END

Each node is a small specialist that reads and writes the typed `AgentState`. Conditional edges
branch on typed values (`RouteDecision`, `ValidationResult`), never on free text. The
checkpointer persists state per `thread_id`, which is the conversation memory.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes.common import instrumented
from app.agents.nodes.memory import load_memory_node, update_memory_node
from app.agents.nodes.response import finalize_node, response_node
from app.agents.nodes.retrieval import retrieval_node
from app.agents.nodes.supervisor import route_after_supervisor, supervisor_node
from app.agents.nodes.validator import route_after_validation, validator_node
from app.agents.state import AgentState


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    g = StateGraph(AgentState)
    g.add_node("load_memory", load_memory_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("response", response_node)
    # Wrapped here so the validator emits node_start/node_end and contains its own failures.
    g.add_node("validator", instrumented("validator")(validator_node))
    g.add_node("finalize", finalize_node)
    g.add_node("update_memory", update_memory_node)

    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"retrieval": "retrieval", "respond_direct": "response"},
    )
    g.add_edge("retrieval", "response")
    g.add_edge("response", "validator")
    g.add_conditional_edges(
        "validator", route_after_validation, {"response": "response", "finalize": "finalize"}
    )
    g.add_edge("finalize", "update_memory")
    g.add_edge("update_memory", END)
    return g.compile(checkpointer=checkpointer, name="compass")
