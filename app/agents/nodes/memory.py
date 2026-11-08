"""Conversation memory nodes.

Design: the checkpointer (SQLite, keyed by thread_id) persists `messages` and `summary` between
turns, which covers "previous questions" and "survives multiple turns". User context comes from
the JWT into state each turn, never from chat text. To cap tokens, once a thread grows past
`MAX_MESSAGES`, older messages are folded into a rolling summary by the fast model and removed;
the last `KEEP_MESSAGES` stay verbatim. Every read and write emits a `memory` event.
"""

from langchain_core.messages import HumanMessage, RemoveMessage

from app.agents import prompts
from app.agents.events import emit
from app.agents.nodes.common import instrumented
from app.agents.state import AgentState
from app.core.llm import ainvoke_llm

MAX_MESSAGES = 12  # about 6 turns before summarising
KEEP_MESSAGES = 6


@instrumented("load_memory")
async def load_memory_node(state: AgentState) -> dict:
    """Start of turn: report what memory holds and reset per-turn fields."""
    messages = state.get("messages", [])
    emit(
        "memory",
        node="load_memory",
        action="loaded",
        previous_turns=sum(1 for m in messages if m.type == "human") - 1,
        has_summary=bool(state.get("summary")),
        user=state["user"].model_dump(),
    )
    return {
        "route": None,
        "guard": None,
        "retrieved": [],
        "retrieval_notes": [],
        "research": None,
        "tool_results": [],
        "draft_answer": None,
        "validation": None,
        "response_attempts": 0,
        "errors": None,
    }


@instrumented("update_memory")
async def update_memory_node(state: AgentState) -> dict:
    """End of turn: summarise and trim long threads; the checkpointer saves the rest."""
    messages = state.get("messages", [])
    if len(messages) <= MAX_MESSAGES:
        emit("memory", node="update_memory", action="saved", messages=len(messages))
        return {}

    old = messages[:-KEEP_MESSAGES]
    transcript = "\n".join(f"{m.type}: {m.content}" for m in old)
    prompt = prompts.SUMMARISE.format(
        summary=state.get("summary") or "(none)", transcript=transcript
    )
    reply = await ainvoke_llm("fast", [HumanMessage(prompt)])
    emit(
        "memory",
        node="update_memory",
        action="summarised",
        folded_messages=len(old),
        kept_messages=KEEP_MESSAGES,
    )
    return {"summary": reply.text, "messages": [RemoveMessage(id=m.id) for m in old]}
