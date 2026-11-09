"""Tools agent: MCP directory lookups and approval-gated admin actions.

The model never *executes* anything here — it only proposes typed `ToolPlan` calls, and every
proposal is re-bound to the caller's role twice: once against `registry.tools_for(role)`
(a tool the role cannot use is skipped with a `denied` event) and once inside the tool
modules at execution time. Admin actions pause the whole graph with `interrupt()` until a
human resolves the approval — the model cannot approve its own request.

Containment: a directory or admin failure is recorded as a failed tool result plus an
`AgentError` in state, so the response agent can say the directory is unavailable instead of
the request crashing.
"""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError

from app.agents import prompts
from app.agents.events import emit
from app.agents.nodes.common import instrumented, last_user_text
from app.agents.state import AgentError, AgentState
from app.core.llm import ainvoke_structured
from app.security.rbac import UserContext
from app.tools import admin, registry
from app.tools.mcp_client import ToolUnavailableError, call_mcp_tool

MAX_TOOL_CALLS = 3

_APPROVAL_MISMATCH = (
    "The proposed action changed after approval, so nothing was run. Please ask again."
)
_DIRECTORY_UNAVAILABLE = "The directory service is temporarily unavailable."


class ToolCallRequest(BaseModel):
    """One proposed call. `why` is short and lands in the activity panel traces."""

    tool: str
    args: dict = Field(default_factory=dict)
    why: str = Field(default="")


class ToolPlan(BaseModel):
    calls: list[ToolCallRequest] = Field(default_factory=list, max_length=MAX_TOOL_CALLS)


def _preview(result: Any) -> str:
    """Short snippet of a tool result for the activity panel (full data goes to the model)."""
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    return text[:200]


def _entry(tool: str, args: dict, *, ok: bool, payload: Any) -> dict:
    """One `tool_results` item: `result` on success, `error` on failure (never both)."""
    key = "result" if ok else "error"
    return {"tool": tool, "args": args, "ok": ok, key: payload}


async def _mcp_call(
    tool: str, args: dict, *, why: str, user: UserContext
) -> tuple[dict, AgentError | None]:
    emit("tool_call", node="tools", tool=tool, status="running", args=args, why=why)
    try:
        # RBAC is re-checked inside call_mcp_tool; validation happens before any I/O.
        result = await call_mcp_tool(tool, args, user=user)
    except ToolUnavailableError as exc:
        emit("tool_call", node="tools", tool=tool, status="failed", args=args, error=str(exc))
        # Degraded, not fatal: the answer must tell the user the directory is down.
        return (
            _entry(tool, args, ok=False, payload=_DIRECTORY_UNAVAILABLE),
            AgentError(node="tools", code="mcp_unavailable", message=str(exc), retryable=True),
        )
    except (ValidationError, ValueError, PermissionError) as exc:
        emit("tool_call", node="tools", tool=tool, status="failed", args=args, error=str(exc))
        return (
            _entry(tool, args, ok=False, payload=str(exc)),
            AgentError(node="tools", code="invalid_tool_call", message=str(exc)),
        )
    emit("tool_call", node="tools", tool=tool, status="ok", args=args, preview=_preview(result))
    return _entry(tool, args, ok=True, payload=result), None


async def _admin_call(
    tool: str, args: dict, *, why: str, user: UserContext
) -> tuple[dict, AgentError | None]:
    # Everything before interrupt() re-runs on resume — and that includes the LLM plan — so
    # this part stays free of side effects besides events (validate + describe only) and the
    # approval is bound below to the exact tool and args the human approved.
    try:
        validated = admin.ADMIN_TOOLS[tool].model_validate(args).model_dump()
    except ValidationError as exc:
        emit(
            "tool_call",
            node="tools",
            tool=tool,
            status="failed",
            args=args,
            error="; ".join(e["msg"] for e in exc.errors()),
        )
        return (
            _entry(tool, args, ok=False, payload="invalid arguments"),
            AgentError(node="tools", code="invalid_tool_call", message=repr(exc)),
        )
    action = admin.describe_action(tool, validated)
    emit(
        "tool_call",
        node="tools",
        tool=tool,
        status="pending_approval",
        args=validated,
        action=action,
        why=why,
    )
    decision = interrupt(
        {
            "type": "approval",
            "tool": tool,
            "args": validated,
            "action": action,
            "requested_by": user.user_id,
        }
    )
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    if not approved:
        emit("tool_call", node="tools", tool=tool, status="rejected", args=validated, why=why)
        return _entry(tool, validated, ok=False, payload="Rejected by the approver."), None
    if not (
        isinstance(decision, dict)
        and decision.get("tool") == tool
        and decision.get("args") == validated
    ):
        # Fail closed: the plan re-ran on resume, so a bare approval (or one for a different
        # tool/args) is not authorization for what this run is about to execute.
        emit(
            "tool_call",
            node="tools",
            tool=tool,
            status="failed",
            args=validated,
            error=_APPROVAL_MISMATCH,
        )
        return (
            _entry(tool, validated, ok=False, payload=_APPROVAL_MISMATCH),
            AgentError(node="tools", code="approval_mismatch", message=_APPROVAL_MISMATCH),
        )
    try:
        result = await admin.execute_admin_tool(tool, validated, user=user, approved=True)
    except Exception as exc:
        emit("tool_call", node="tools", tool=tool, status="failed", args=validated, error=str(exc))
        return (
            _entry(tool, validated, ok=False, payload=str(exc)),
            AgentError(node="tools", code="admin_failed", message=str(exc)),
        )
    emit(
        "tool_call", node="tools", tool=tool, status="ok", args=validated, preview=_preview(result)
    )
    return _entry(tool, validated, ok=True, payload=result), None


@instrumented("tools")
async def tools_node(state: AgentState) -> dict:
    """Plan and run up to three role-bound tool calls; failures degrade, never raise."""
    user = state["user"]
    route = state.get("route")
    # Role-filtered binding #1: the catalogue (and therefore the LLM) only ever contains
    # tools `tools_for(role)` allows, and the chosen names are re-checked against the same
    # set before execution.
    allowed = {spec.name for spec in registry.tools_for(user.role)}
    system = prompts.TOOLS.format(
        catalogue=registry.catalogue_text(user.role),
        name=user.name,
        role=user.role.value,
        department=user.department,
    )
    question = last_user_text(state)
    query = f"Question: {question}\nSearch query: {route.search_query if route else ''}"
    plan = await ainvoke_structured("main", ToolPlan, [SystemMessage(system), HumanMessage(query)])

    results: list[dict] = []
    errors: list[AgentError] = []
    for call in plan.calls[:MAX_TOOL_CALLS]:
        if call.tool not in allowed:
            emit(
                "tool_call",
                node="tools",
                tool=call.tool,
                status="denied",
                reason=f"Tool '{call.tool}' is not available to role '{user.role}'.",
            )
            continue
        spec = registry.TOOLS[call.tool]
        if spec.permission == "admin":
            entry, error = await _admin_call(call.tool, call.args, why=call.why, user=user)
        else:
            entry, error = await _mcp_call(call.tool, call.args, why=call.why, user=user)
        results.append(entry)
        if error is not None:
            errors.append(error)

    update: dict = {"tool_results": results}
    if errors:
        # Appended (not replaced) by the reducer into the turn's error list in finalize.
        update["errors"] = errors
    return update


def route_after_tools(state: AgentState) -> str:
    """Directory questions may also need documents, so retrieval runs after tools when asked."""
    route = state.get("route")
    if route is not None and "retrieval" in (route.agents or []):
        return "retrieval"
    return "response"
