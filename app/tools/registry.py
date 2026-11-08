"""Role-filtered tool catalogue.

Binding tools by role happens twice: only allowed tools are *offered to the LLM* (via
`catalogue_text`, so the model never sees — and never argues for — tools the caller
cannot use), and `can_use` is re-checked at execution time inside each tool module.
"""

import json
from dataclasses import dataclass

from pydantic import BaseModel

from app.security.rbac import Role, can_use
from app.tools.admin import ADMIN_TOOLS, ReindexDepartmentArgs, RevokeDocumentArgs
from app.tools.mcp_client import (
    MCP_TOOL_NAMES,
    GetServiceStatusArgs,
    ListOpenIncidentsArgs,
    LookupEmployeeArgs,
)


@dataclass(frozen=True)
class ToolSpec:
    """One tool in the catalogue.

    `permission` is the `can_use` key, not the tool name, so several tools can share one
    permission (all three MCP tools check "mcp").
    """

    name: str
    permission: str
    description: str
    requires_approval: bool = False
    args_model: type[BaseModel] | None = None


_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="knowledge_search",
        permission="knowledge_search",
        description="Hybrid keyword + semantic search over the department knowledge base.",
    ),
    ToolSpec(
        name="python_analysis",
        permission="python_analysis",
        description="Run a small, sandboxed Python snippet to analyse retrieved data.",
    ),
    ToolSpec(
        name="lookup_employee",
        permission="mcp",
        description="Find employees by name, employee id or team; returns contact details.",
        args_model=LookupEmployeeArgs,
    ),
    ToolSpec(
        name="get_service_status",
        permission="mcp",
        description="Get the current status of a service (operational, degraded, outage).",
        args_model=GetServiceStatusArgs,
    ),
    ToolSpec(
        name="list_open_incidents",
        permission="mcp",
        description="List unresolved incidents, optionally filtered by department.",
        args_model=ListOpenIncidentsArgs,
    ),
    ToolSpec(
        name="reindex_department",
        permission="admin",
        description="Re-index all documents of a department (destructive, needs approval).",
        requires_approval=True,
        args_model=ReindexDepartmentArgs,
    ),
    ToolSpec(
        name="revoke_document",
        permission="admin",
        description="Revoke a document from the knowledge base (needs approval).",
        requires_approval=True,
        args_model=RevokeDocumentArgs,
    ),
]

# Keyed by name so lookups in UI / tests never scan the list.
TOOLS: dict[str, ToolSpec] = {spec.name: spec for spec in _TOOLS}

# Cross-check at import time: the MCP names and admin arg models must match what the
# tool modules actually expose, or the catalogue would lie to the model.
assert set(MCP_TOOL_NAMES) <= TOOLS.keys()
assert set(ADMIN_TOOLS) <= TOOLS.keys()


def tools_for(role: Role) -> list[ToolSpec]:
    """Only the tools `role` may use — the LLM is never shown the rest."""
    return [spec for spec in _TOOLS if can_use(role, spec.permission)]


def catalogue_text(role: Role) -> str:
    """Compact catalogue of the role's tools (name, description, JSON args schema).

    Kept as one text block so it can be dropped verbatim into a system prompt.
    """
    lines: list[str] = []
    for spec in tools_for(role):
        lines.append(f"- {spec.name}: {spec.description}")
        if spec.args_model is not None:
            schema = spec.args_model.model_json_schema()
            # Inline JSON keeps each tool on one block, easy for the model to parse.
            lines.append(f"  args: {json.dumps(schema)}")
        else:
            lines.append("  args: none")
        if spec.requires_approval:
            lines.append("  note: requires human approval before execution")
    return "\n".join(lines)
