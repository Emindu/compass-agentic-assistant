"""Chat endpoints: streamed answers over SSE, thread history, and approval round-trips.

Why SSE: the server pushes a one-way stream of typed events (activity + tokens) over plain HTTP,
which is simpler than WebSockets and works through proxies. Thread ids are namespaced by the
authenticated user, so one user can never load or continue another user's conversation.

Admin actions pause the graph via LangGraph interrupts; after a stream ends the endpoint checks
the thread's state and emits one `approval` event so the UI (or any API client) can collect the
decision and continue the thread through POST /chat/resume.
"""

import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agents.events import AgentEvent
from app.core.errors import AppError, AuthorizationError, InvalidRequestError
from app.core.logging import get_logger, request_id_var
from app.core.rate_limit import enforce_rate_limit
from app.security.rbac import UserContext, can_use

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class ResumeRequest(BaseModel):
    """The human's decision on a pending approval."""

    thread_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    approved: bool


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


def _pending_interrupt(snapshot: Any) -> dict | None:
    """First pending interrupt payload on a thread state snapshot, if any.

    Tolerant across langgraph versions: the interrupts live either on the snapshot itself
    (`snapshot.interrupts`) or on its pending tasks (`snapshot.tasks[*].interrupts`), and older
    revisions hand plain dicts instead of `Interrupt` objects.
    """
    interrupts = list(getattr(snapshot, "interrupts", None) or ())
    for task in getattr(snapshot, "tasks", None) or ():
        interrupts.extend(getattr(task, "interrupts", None) or ())
    for intr in interrupts:
        if isinstance(intr, dict):
            value = intr.get("value", intr)
        else:
            value = getattr(intr, "value", None)
        if isinstance(value, dict):
            return value
        if value is not None:
            return {"value": value}
    return None


async def _event_stream(
    graph: CompiledStateGraph, inputs: Any, config: dict, thread_id: str
) -> AsyncIterator[str]:
    """Shared SSE loop: relay custom events, then surface a pending approval, then `done`."""
    try:
        # subgraphs=True: the research agent runs a nested graph whose plan/batch events
        # must reach the activity panel too; the namespace tuple is not needed.
        async for _ns, event in graph.astream(inputs, config, stream_mode="custom", subgraphs=True):
            yield AgentEvent.model_validate(event).sse()
    except Exception as exc:  # stream already started: report in-band, never crash
        log.exception("chat_stream_failed")
        message = exc.message if isinstance(exc, AppError) else "Something went wrong."
        yield AgentEvent(type="error", data={"message": message}).sse()
    # The stream ends on an interrupt without a "done"; tell the client what is pending.
    try:
        interrupt_value = _pending_interrupt(await graph.aget_state(config))
    except Exception:
        log.warning("interrupt_check_failed", thread_id=thread_id)
        interrupt_value = None
    if interrupt_value is not None:
        yield AgentEvent(
            type="approval", node="tools", data={**interrupt_value, "thread_id": thread_id}
        ).sse()
    yield AgentEvent(type="done", data={"thread_id": thread_id}).sse()


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
    return StreamingResponse(
        _event_stream(graph, inputs, config, thread_id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/resume")
async def chat_resume(
    body: ResumeRequest,
    user: UserContext = Depends(enforce_rate_limit),
    graph: CompiledStateGraph = Depends(get_graph),
) -> StreamingResponse:
    """Continue a thread paused on a human-approval interrupt; admins only, by design."""
    if not can_use(user.role, "admin"):
        raise AuthorizationError("Only an admin may resolve pending approval requests.")
    config = run_config(user, body.thread_id)
    pending = _pending_interrupt(await graph.aget_state(config))
    if pending is None:
        raise InvalidRequestError("This thread has no pending approval to resolve.")
    log.info("chat_resume", thread_id=body.thread_id, approved=body.approved)
    # The resume value is filled from the checkpointed interrupt, never from the client: the
    # tools node (including the LLM plan) re-runs on resume, so the decision the human made
    # carries the exact action they saw.
    inputs = Command(
        resume={
            "approved": body.approved,
            "tool": pending.get("tool"),
            "args": pending.get("args"),
        }
    )
    return StreamingResponse(
        _event_stream(graph, inputs, config, body.thread_id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
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
