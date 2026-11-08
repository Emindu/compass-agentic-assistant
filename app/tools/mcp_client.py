"""Client layer for the compass-directory MCP server.

Everything an agent needs to call MCP tools goes through `call_mcp_tool`, which owns the
cross-cutting concerns the POC is demonstrating:

* RBAC is re-checked here, at execution time — an LLM that hallucinates a tool call still
  cannot escape the calling user's role.
* Arguments are validated against per-tool Pydantic models *before* any connection is
  opened, so bad calls fail fast and cheaply.
* Fault flags (`fault_mcp`, `fault_tool_timeout`) turn into one typed error
  (`ToolUnavailableError`) so the supervisor can degrade gracefully.
* `MCP_MODE=inprocess` calls the server's pure functions directly: no subprocess, no
  network — used by tests and Docker-less demos.
"""

import asyncio
import json
import os
import sys
import time
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import get_settings
from app.core.errors import AuthorizationError
from app.core.logging import get_logger
from app.security.rbac import UserContext, can_use

logger = get_logger(__name__)


class ToolUnavailableError(Exception):
    """The MCP tool could not deliver a result (down, failed, or timed out).

    Deliberately an infrastructure error, not an AppError: callers treat it as
    "try another path / tell the user" rather than a permissions or request bug.
    """


MCP_TOOL_NAMES = ("lookup_employee", "get_service_status", "list_open_incidents")

# extra="forbid": an LLM that invents extra arguments gets a validation error instead of
# silently sending junk to the server.
_FORBID = ConfigDict(extra="forbid")


class LookupEmployeeArgs(BaseModel):
    model_config = _FORBID

    query: str = Field(min_length=1, max_length=100)


class GetServiceStatusArgs(BaseModel):
    model_config = _FORBID

    service: str = Field(min_length=1, max_length=100)


class ListOpenIncidentsArgs(BaseModel):
    model_config = _FORBID

    department: str | None = Field(default=None, min_length=1, max_length=100)


_ARGS_MODELS: dict[str, type[BaseModel]] = {
    "lookup_employee": LookupEmployeeArgs,
    "get_service_status": GetServiceStatusArgs,
    "list_open_incidents": ListOpenIncidentsArgs,
}


async def _call_inprocess(name: str, args: dict) -> Any:
    """Fallback path (MCP_MODE=inprocess): call the pure functions directly.

    Imports live inside the function so `app` code can be imported without the
    `mcp_server` package on the path (e.g. in slim deployments).
    """
    from mcp_server import server

    functions = {
        "lookup_employee": server.lookup_employee,
        "get_service_status": server.get_service_status,
        "list_open_incidents": server.list_open_incidents,
    }
    fn = functions[name]
    return fn(**args)


# Lazily-initialised holder for the connected client and its tools. Connecting spawns a
# stdio subprocess, so it is done once per process, guarded by a lock against concurrent
# first calls; a failed call clears the cache so the next attempt reconnects.
_lock: asyncio.Lock = asyncio.Lock()
_client: Any = None  # MultiServerMCPClient; typed Any to keep the import lazy-free
_tools: dict[str, BaseTool] = {}


def _reset_client_cache() -> None:
    global _client, _tools
    _client = None
    _tools = {}


async def _get_tool(name: str) -> BaseTool:
    """Return the cached LangChain tool for `name`, connecting on first use."""
    global _client, _tools
    from langchain_mcp_adapters.client import MultiServerMCPClient

    async with _lock:
        if name in _tools:
            return _tools[name]
        connection = {
            "compass": {
                "command": sys.executable,
                "args": ["-m", "mcp_server.server"],
                "transport": "stdio",
            }
        }
        # Fresh process of the server: same interpreter, stdio transport.
        _client = MultiServerMCPClient(connection)
        _tools = {tool.name: tool for tool in await _client.get_tools()}
        if name not in _tools:
            raise ValueError(f"server does not expose tool {name!r}")
        return _tools[name]


_LIST_TOOLS = frozenset({"lookup_employee", "list_open_incidents"})


def _parse_result(result: Any) -> Any:
    """Unwrap FastMCP's text content and decode JSON payloads into Python objects.

    A list return value is serialised as one text block per element (not a JSON array),
    so a multi-block body is decoded element-by-element after a joint parse fails.
    """
    items: list[str] = []
    if isinstance(result, str):
        items = [result]
    elif (
        isinstance(result, list)
        and result
        and all(isinstance(item, dict) and item.get("type") == "text" for item in result)
    ):
        items = [item["text"] for item in result]
    if not items:
        return result
    try:
        return json.loads("\n".join(items))
    except json.JSONDecodeError:
        try:
            return [json.loads(item) for item in items]
        except json.JSONDecodeError:
            return "\n".join(items)


async def call_mcp_tool(
    name: str, args: dict, *, user: UserContext, timeout_s: float = 10.0
) -> Any:
    """Call one compass-directory tool on behalf of `user`.

    Raises AuthorizationError (role), ValueError (unknown tool), pydantic ValidationError
    (bad args) or ToolUnavailableError (server down / timed out).
    """
    if not can_use(user.role, "mcp"):
        raise AuthorizationError(f"role {user.role} may not use MCP tools")
    if name not in MCP_TOOL_NAMES:
        raise ValueError(f"unknown MCP tool {name!r}; expected one of {MCP_TOOL_NAMES}")
    # Validate before any I/O: fail fast on malformed model output.
    validated = _ARGS_MODELS[name].model_validate(args).model_dump()

    started = time.perf_counter()
    ok = False
    try:
        settings = get_settings()  # call time, not import time: tests flip flags mid-run
        if settings.fault_mcp:
            raise ToolUnavailableError("MCP server unavailable (FAULT_MCP)")
        if settings.fault_tool_timeout:
            # Simulate a hung server: the sleep outlives the timeout below.
            try:
                async with asyncio.timeout(timeout_s):
                    await asyncio.sleep(timeout_s + 1)
            except TimeoutError as exc:
                raise ToolUnavailableError("MCP tool timed out (FAULT_TOOL_TIMEOUT)") from exc

        result = await _dispatch(name, validated, timeout_s=timeout_s)
        parsed = _parse_result(result)
        # A one-element list arrives as a single text block and decodes to a dict; restore the
        # declared list shape so callers never branch on result cardinality.
        if name in _LIST_TOOLS and not isinstance(parsed, list):
            parsed = [] if parsed in (None, "") else [parsed]
        ok = True
        return parsed
    finally:
        logger.info(
            "mcp_call",
            tool=name,
            ok=ok,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            user_id=user.user_id,
            role=str(user.role),
        )


async def _dispatch(name: str, args: dict, *, timeout_s: float) -> Any:
    """Run the call (in-process or stdio) under a timeout, reconnecting on failure."""
    if os.environ.get("MCP_MODE", "").lower() == "inprocess":
        return await _call_inprocess(name, args)

    tool = await _get_tool(name)
    try:
        async with asyncio.timeout(timeout_s):
            return await tool.ainvoke(args)
    except Exception:
        # Session died: drop the cached tools so the next call reconnects instead of
        # reusing a dead subprocess.
        _reset_client_cache()
        raise
