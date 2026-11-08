"""Typed activity events streamed to the UI's agent activity panel over SSE.

Nodes emit events through LangGraph's custom stream writer, so the panel shows what the agent
is doing as it happens: active node, routing, retrieval, memory, validation, tokens.
"""

import time
from typing import Any, Literal

from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

EventType = Literal[
    "node_start",
    "node_end",
    "route",
    "retrieval",
    "tool_call",
    "memory",
    "validation",
    "token",
    "answer",
    "error",
    "done",
]


class AgentEvent(BaseModel):
    type: EventType
    node: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)

    def sse(self) -> str:
        """Server-sent events wire format: `event:` line names the type, `data:` holds JSON."""
        return f"event: {self.type}\ndata: {self.model_dump_json()}\n\n"


def emit(type_: EventType, node: str | None = None, **data: Any) -> None:
    """Send an event to the client if the graph is being streamed; no-op otherwise (tests)."""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(AgentEvent(type=type_, node=node, data=data).model_dump())
