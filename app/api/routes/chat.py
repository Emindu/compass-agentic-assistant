"""Chat endpoints: streamed answers over SSE, and thread history.

Why SSE: the server pushes a one-way stream of typed events (activity + tokens) over plain HTTP,
which is simpler than WebSockets and works through proxies. Thread ids are namespaced by the
authenticated user, so one user can never load or continue another user's conversation.
"""

import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.agents.events import AgentEvent
from app.core.errors import AppError
from app.core.logging import get_logger, request_id_var
from app.core.rate_limit import enforce_rate_limit
from app.security.rbac import UserContext

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.graph


def scoped_thread(user: UserContext, thread_id: str) -> str:
    return f"{user.user_id}:{thread_id}"


def run_config(user: UserContext, thread_id: str) -> dict:
    """Checkpointer key + LangSmith metadata, so every trace is filterable by user/role/thread."""
    return {
        "configurable": {"thread_id": scoped_thread(user, thread_id)},
        "run_name": "compass_chat",
        "tags": [f"role:{user.role.value}", "chat"],
        "metadata": {
            "user_id": user.user_id,
            "role": user.role.value,
            "thread_id": thread_id,
            "request_id": request_id_var.get(),
        },
    }


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    user: UserContext = Depends(enforce_rate_limit),
    graph: CompiledStateGraph = Depends(get_graph),
) -> StreamingResponse:
    thread_id = body.thread_id or uuid.uuid4().hex[:12]
    inputs = {"messages": [HumanMessage(body.message)], "user": user}
    config = run_config(user, thread_id)
    log.info("chat_request", thread_id=thread_id, chars=len(body.message))

    async def events() -> AsyncIterator[str]:
        try:
            async for event in graph.astream(inputs, config, stream_mode="custom"):
                yield AgentEvent.model_validate(event).sse()
        except Exception as exc:  # stream already started: report in-band, never crash
            log.exception("chat_stream_failed")
            message = exc.message if isinstance(exc, AppError) else "Something went wrong."
            yield AgentEvent(type="error", data={"message": message}).sse()
        yield AgentEvent(type="done", data={"thread_id": thread_id}).sse()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history/{thread_id}")
async def history(
    thread_id: str,
    user: UserContext = Depends(enforce_rate_limit),
    graph: CompiledStateGraph = Depends(get_graph),
) -> dict:
    snapshot = await graph.aget_state(run_config(user, thread_id))
    values = snapshot.values or {}
    return {
        "thread_id": thread_id,
        "summary": values.get("summary"),
        "messages": [{"role": m.type, "content": m.content} for m in values.get("messages", [])],
    }
