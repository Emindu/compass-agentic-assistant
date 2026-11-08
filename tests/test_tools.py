"""Tests for the tool layer: MCP client (RBAC, validation, faults), admin approval flow,
and the role-filtered registry. Network-free: MCP_MODE=inprocess and env-driven flags."""

import time

import pytest
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.errors import AuthorizationError
from app.security.rbac import Role, UserContext
from app.tools.admin import describe_action, execute_admin_tool
from app.tools.mcp_client import ToolUnavailableError, call_mcp_tool
from app.tools.registry import TOOLS, catalogue_text, tools_for

VIEWER = UserContext(user_id="u-viewer", name="Vera", role=Role.viewer, department="viewer-demo")
ANALYST = UserContext(
    user_id="u-analyst", name="Anil", role=Role.analyst, department="analyst-demo"
)
ADMIN = UserContext(user_id="u-admin", name="Ada", role=Role.admin, department="admin-demo")


@pytest.fixture(autouse=True)
def _fresh_settings_cache():
    """get_settings is lru_cached; clear before and after each test so env overrides
    (FAULT_MCP, FAULT_TOOL_TIMEOUT) apply at call time and never leak across tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def inprocess(monkeypatch):
    monkeypatch.setenv("MCP_MODE", "inprocess")


async def test_viewer_cannot_call_mcp_tools(inprocess):
    # RBAC is re-checked at execution time, not just at binding time.
    with pytest.raises(AuthorizationError):
        await call_mcp_tool("lookup_employee", {"query": "nimal"}, user=VIEWER)


async def test_analyst_can_call_mcp_tool_inprocess(inprocess):
    results = await call_mcp_tool("lookup_employee", {"query": "Nimal"}, user=ANALYST)
    assert [r["employee_id"] for r in results] == ["EMP-001"]
    assert "private" not in results[0]


async def test_inprocess_status_and_incidents(inprocess):
    status = await call_mcp_tool("get_service_status", {"service": "biller-gateway"}, user=ANALYST)
    assert status["status"] == "outage"
    incidents = await call_mcp_tool("list_open_incidents", {}, user=ANALYST)
    assert len(incidents) == 2
    payments_only = await call_mcp_tool(
        "list_open_incidents", {"department": "payments"}, user=ANALYST
    )
    assert len(payments_only) == 2


async def test_unknown_tool_name_is_rejected(inprocess):
    with pytest.raises(ValueError, match="unknown MCP tool"):
        await call_mcp_tool("purge_database", {}, user=ANALYST)


async def test_bad_args_fail_validation_before_any_call(inprocess):
    # extra="forbid": invented arguments are rejected before the (in-process) call runs.
    with pytest.raises(ValidationError):
        await call_mcp_tool("lookup_employee", {"query": "nimal", "extra": 1}, user=ANALYST)
    with pytest.raises(ValidationError):
        await call_mcp_tool("get_service_status", {"service": ""}, user=ANALYST)


async def test_fault_mcp_raises_tool_unavailable(inprocess, monkeypatch):
    monkeypatch.setenv("FAULT_MCP", "true")
    with pytest.raises(ToolUnavailableError, match="FAULT_MCP"):
        await call_mcp_tool("lookup_employee", {"query": "nimal"}, user=ANALYST)


async def test_fault_tool_timeout_raises_quickly(inprocess, monkeypatch):
    monkeypatch.setenv("FAULT_TOOL_TIMEOUT", "true")
    started = time.perf_counter()
    with pytest.raises(ToolUnavailableError, match="timed out"):
        await call_mcp_tool("lookup_employee", {"query": "nimal"}, user=ANALYST, timeout_s=0.05)
    # The simulated hang outlives timeout_s, so the error arrives at ~timeout_s, not 1s+.
    assert time.perf_counter() - started < 1.0


async def test_analyst_cannot_run_admin_tools():
    with pytest.raises(AuthorizationError):
        await execute_admin_tool(
            "reindex_department", {"department": "payments"}, user=ANALYST, approved=True
        )


async def test_admin_tool_requires_human_approval():
    with pytest.raises(PermissionError, match="human approval"):
        await execute_admin_tool(
            "reindex_department", {"department": "payments"}, user=ADMIN, approved=False
        )


async def test_admin_tool_executes_when_approved():
    result = await execute_admin_tool(
        "reindex_department", {"department": "payments"}, user=ADMIN, approved=True
    )
    assert result == {
        "status": "done",
        "action": "Re-index all payments documents",
        "requested_by": "u-admin",
    }


async def test_admin_tool_rejects_invalid_doc_id():
    with pytest.raises(ValidationError):
        await execute_admin_tool(
            "revoke_document",
            {"doc_id": "rb-pay-003", "reason": "superseded by v2 runbook"},
            user=ADMIN,
            approved=True,
        )


async def test_admin_tool_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown admin tool"):
        await execute_admin_tool("delete_user", {"user": "nimal"}, user=ADMIN, approved=True)


def test_describe_action_is_human_readable():
    assert (
        describe_action("reindex_department", {"department": "retail"})
        == "Re-index all retail documents"
    )
    assert (
        describe_action("revoke_document", {"doc_id": "RB-PAY-003-V2", "reason": "withdrawn"})
        == "Revoke document RB-PAY-003-V2 (reason: withdrawn)"
    )


def test_viewer_catalogue_has_only_knowledge_search():
    assert [spec.name for spec in tools_for(Role.viewer)] == ["knowledge_search"]
    assert "revoke_document" not in catalogue_text(Role.viewer)
    assert "lookup_employee" not in catalogue_text(Role.viewer)


def test_analyst_catalogue_has_no_admin_tools():
    names = [spec.name for spec in tools_for(Role.analyst)]
    assert "reindex_department" not in names
    assert "revoke_document" not in names
    assert "lookup_employee" in names


def test_admin_catalogue_lists_everything():
    assert {spec.name for spec in tools_for(Role.admin)} == set(TOOLS)
    text = catalogue_text(Role.admin)
    assert "reindex_department" in text
    assert "requires human approval" in text


def test_catalogue_text_includes_arg_schema():
    text = catalogue_text(Role.analyst)
    # The model gets a machine-readable JSON schema inline in the prompt.
    assert "lookup_employee" in text
    assert '"query"' in text
    assert "anyOf" in text  # list_open_incidents' optional department serialises as anyOf
