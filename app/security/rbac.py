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
