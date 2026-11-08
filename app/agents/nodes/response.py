"""Response agent: streams the final answer, grounded in retrieved chunks, with citations.

Tokens are emitted as they arrive so the UI streams the answer. Evidence is wrapped in
<document> tags with its chunk id; the prompt treats it as untrusted data. If the model fails,
the fallback still returns the most relevant sources, so the employee gets something useful.
"""

from html import escape

from langchain_core.messages import AIMessage, SystemMessage

from app.agents import prompts
from app.agents.events import emit
from app.agents.nodes.common import instrumented
from app.agents.state import AgentState
from app.core.llm import astream_llm
from app.retrieval.models import RetrievedChunk

HISTORY_MESSAGES = 6


def format_documents(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no documents found that you are allowed to access)"
    return "\n\n".join(
        f'<document id="{c.id}" title="{escape(c.title)}" status="{c.status}" '
        f'date="{c.created_date}">\n{c.text}\n</document>'
        for c in chunks
    )


def _fallback(state: AgentState, _exc: Exception) -> dict:
    chunks = state.get("retrieved") or []
    if chunks:
        sources = "\n".join(f"- {c.title} [{c.id}]" for c in chunks[:5])
        text = (
            "I can't generate a full answer right now because the language model is "
            f"unavailable. These documents look most relevant:\n{sources}"
        )
    else:
        text = (
            "I can't answer right now because the language model is unavailable. "
            "Please try again shortly."
        )
    return {"draft_answer": text, "response_attempts": state.get("response_attempts", 0) + 1}


def _research_block(state: AgentState) -> str:
    research = state.get("research")
    if research is None:
        return ""
    return (
        "\nResearch findings from the parallel analysis of the documents below (use them, "
        "and cite the underlying chunk ids):\n"
        f"{research.synthesis}\nThemes: {research.themes}\nComputed data: {research.data}\n"
        f"Documents analysed: {research.documents}\n"
    )


def _feedback(state: AgentState) -> str:
    validation = state.get("validation")
    if validation is None or validation.valid:
        return ""
    problems = "; ".join(validation.issues)
    return f"\nYour previous draft was rejected by the citation checker: {problems}. Fix this.\n"


@instrumented("response", fallback=_fallback)
async def response_node(state: AgentState) -> dict:
    user = state["user"]
    route = state.get("route")
    attempt = state.get("response_attempts", 0) + 1
    common = {"name": user.name, "role": user.role.value}
    if route is not None and route.agents:
        system = prompts.RESPONSE.format(
            **common,
            department=user.department,
            summary=state.get("summary") or "(none)",
            feedback=_research_block(state) + _feedback(state),
            documents=format_documents(state.get("retrieved") or []),
        )
    else:
        system = prompts.DIRECT.format(**common)

    history = state.get("messages", [])[-HISTORY_MESSAGES:]
    parts: list[str] = []
    async for chunk in astream_llm("main", [SystemMessage(system), *history]):
        text = chunk.text
        if text:
            parts.append(text)
            emit("token", node="response", text=text, attempt=attempt)
    return {"draft_answer": "".join(parts).strip(), "response_attempts": attempt}


@instrumented("finalize")
async def finalize_node(state: AgentState) -> dict:
    """Commit the validated answer to conversation memory and tell the UI it is final."""
    answer = state.get("draft_answer") or ""
    retrieved = {c.id: c for c in state.get("retrieved") or []}
    validation = state.get("validation")
    cited = [retrieved[i] for i in (validation.cited_ids if validation else []) if i in retrieved]
    emit(
        "answer",
        node="finalize",
        text=answer,
        citations=[
            {"id": c.id, "doc_id": c.doc_id, "title": c.title, "section": c.section} for c in cited
        ],
        notes=state.get("retrieval_notes") or [],
        errors=[e.model_dump() for e in state.get("errors") or []],
    )
    return {"messages": [AIMessage(answer)]}
