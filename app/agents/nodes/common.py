"""Instrumentation shared by every graph node: activity events, timing and failure containment.

Why a wrapper instead of try/except in each node: containment is a graph-wide policy. Any node
that raises is turned into an `AgentError` in state plus an `error` event, and returns its
`fallback` update, so downstream nodes still run on whatever state is available (a degraded
answer instead of a crashed request). That is the "butterfly effect" defence: one failing
agent cannot take down the others.
"""

import functools
import time
from collections.abc import Awaitable, Callable
from typing import Any

from app.agents.events import emit
from app.agents.state import AgentError, AgentState
from app.core.errors import AppError
from app.core.logging import get_logger

log = get_logger(__name__)

Node = Callable[[AgentState], Awaitable[dict[str, Any]]]
Fallback = Callable[[AgentState, Exception], dict[str, Any]]


def instrumented(name: str, fallback: Fallback | None = None) -> Callable[[Node], Node]:
    def decorator(fn: Node) -> Node:
        @functools.wraps(fn)
        async def wrapper(state: AgentState) -> dict[str, Any]:
            emit("node_start", node=name)
            start = time.perf_counter()
            try:
                update = await fn(state)
                status = "ok"
            except Exception as exc:
                status = "error"
                code = exc.code if isinstance(exc, AppError) else "node_failed"
                retryable = exc.retryable if isinstance(exc, AppError) else False
                user_message = exc.message if isinstance(exc, AppError) else "Internal error."
                log.exception("node_failed", node=name, code=code)
                emit("error", node=name, code=code, message=user_message)
                update = fallback(state, exc) if fallback else {}
                update["errors"] = [
                    AgentError(node=name, code=code, message=repr(exc), retryable=retryable)
                ]
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            emit("node_end", node=name, status=status, duration_ms=duration_ms)
            return update

        return wrapper

    return decorator


def last_user_text(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if msg.type == "human":
            return str(msg.content)
    return ""
