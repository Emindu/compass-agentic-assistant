"""Roles and what each may see.

Only the clearance mapping lives here for now; tool permissions are added with the agent graph.
Clearance is a ceiling on document access level: a role sees every level up to and including it.
"""

from enum import StrEnum

from pydantic import BaseModel

from ingestion.models import AccessLevel


class Role(StrEnum):
    viewer = "viewer"
    analyst = "analyst"
    admin = "admin"


ROLE_CLEARANCE: dict[Role, AccessLevel] = {
    Role.viewer: AccessLevel.internal,
    Role.analyst: AccessLevel.confidential,
    Role.admin: AccessLevel.restricted,
}


def clearance_for(role: Role) -> AccessLevel:
    return ROLE_CLEARANCE[role]


class UserContext(BaseModel):
    """Who is asking. Built only from a verified JWT, never from chat text, and placed in graph
    state so every node (and every trace) knows the caller's role and clearance."""

    user_id: str
    name: str
    role: Role
    department: str

    @property
    def clearance(self) -> AccessLevel:
        return clearance_for(self.role)


# Tool permissions, from the brief: Viewer = chat + search; Analyst = search + analytics + MCP;
# Admin = all tools. Checked twice: before a route/tool is offered to the LLM, and again at
# execution, so a manipulated model still cannot run a tool its user's role does not allow.
TOOL_PERMISSIONS: dict[str, frozenset[Role]] = {
    "knowledge_search": frozenset({Role.viewer, Role.analyst, Role.admin}),
    "python_analysis": frozenset({Role.analyst, Role.admin}),  # RLM research + analytics
    "mcp": frozenset({Role.analyst, Role.admin}),
    "admin": frozenset({Role.admin}),
}


def can_use(role: Role, tool: str) -> bool:
    """Unknown tools are denied by default."""
    return role in TOOL_PERMISSIONS.get(tool, frozenset())
