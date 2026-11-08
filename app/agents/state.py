"""Typed shared state for the agent graph.

Why one shared state: every agent reads and writes the same typed object, so hand-offs are
explicit and visible in traces. Why errors live in state: a node that fails records an
`AgentError` and returns instead of raising, so one failure is contained and the graph routes
around it (degraded answer) rather than the whole request crashing — no butterfly effect.

Per-turn fields are reset at the start of each turn; `messages` and `summary` persist across
turns through the checkpointer (conversation memory).
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.retrieval.models import RetrievedChunk
from app.security.rbac import UserContext
from ingestion.models import Department

Intent = Literal["knowledge_qa", "deep_research", "analytics", "smalltalk", "out_of_scope"]
AgentName = Literal["retrieval", "research", "analysis"]


class RouteDecision(BaseModel):
    """Supervisor output: what the user wants and which specialists should handle it."""

    intent: Intent
    agents: list[AgentName] = Field(
        default_factory=list, description="Specialists to run; empty for smalltalk/out_of_scope"
    )
    search_query: str = Field(
        description="Standalone search query rewritten from the conversation (resolve pronouns)"
    )
    departments: list[Department] | None = Field(
        default=None, description="Departments to search; null means all"
    )
    reasoning: str = Field(description="One sentence explaining the routing, shown in the UI")


class AgentError(BaseModel):
    node: str
    code: str
    message: str
    retryable: bool = False


class ValidationResult(BaseModel):
    """Outcome of the citation check on a drafted answer."""

    valid: bool
    cited_ids: list[str] = Field(default_factory=list, description="Chunk ids cited and found")
    unknown_citations: list[str] = Field(
        default_factory=list, description="Cited ids that were never retrieved (hallucinated)"
    )
    issues: list[str] = Field(default_factory=list, description="Human-readable problems")


def errors_reducer(left: list[AgentError] | None, right: list[AgentError] | None):
    """Append errors from parallel nodes; a node returns `None` to reset at a new turn."""
    if right is None:
        return []
    return [*(left or []), *right]


class AgentState(TypedDict, total=False):
    # Persistent across turns (checkpointed memory)
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    user: UserContext

    # Per turn
    route: RouteDecision | None
    retrieved: list[RetrievedChunk]
    retrieval_notes: list[str]
    draft_answer: str | None
    validation: ValidationResult | None
    response_attempts: int
    errors: Annotated[list[AgentError], errors_reducer]
