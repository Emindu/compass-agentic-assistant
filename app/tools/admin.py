"""Admin tools that require a human in the loop.

The POC's demo admin actions (re-index a department, revoke a document) have no real side
effects — they log and return a confirmation — but the pattern they establish is the point:

* The agent can *propose* the action and render `describe_action` in a UI; nothing runs
  until `execute_admin_tool` is called with `approved=True`.
* Approval is enforced server-side here, not in the prompt — a model cannot approve its
  own request.
* Every execution lands in the `admin_action` audit log regardless of outcome.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.errors import AuthorizationError
from app.core.logging import get_logger
from app.security.rbac import UserContext, can_use

logger = get_logger(__name__)

_FORBID = ConfigDict(extra="forbid")


class ReindexDepartmentArgs(BaseModel):
    model_config = _FORBID

    department: Literal["payments", "platform", "security", "retail", "corporate"]


class RevokeDocumentArgs(BaseModel):
    model_config = _FORBID

    doc_id: str = Field(pattern=r"^[A-Z]{2,5}-[A-Z0-9-]{2,30}$")
    reason: str = Field(min_length=5, max_length=200)


# Registry of admin action names to their argument models — the registry module and the
# approval UI iterate this so the two can never drift apart.
ADMIN_TOOLS: dict[str, type[BaseModel]] = {
    "reindex_department": ReindexDepartmentArgs,
    "revoke_document": RevokeDocumentArgs,
}

# Human phrasing per tool, used to build the approval prompt and the audit entry.
_TOOLS_SUBJ: dict[str, str] = {
    "reindex_department": "Re-index all {department} documents",
    "revoke_document": "Revoke document {doc_id} (reason: {reason})",
}


def describe_action(name: str, args: dict) -> str:
    """Human-readable one-liner for the approval prompt ("Re-index all payments documents").

    Args are validated before this runs, but built defensively with defaults so rendering
    an approval UI can never crash on model-proposed arguments.
    """
    return _TOOLS_SUBJ[name].format(**args)


async def execute_admin_tool(name: str, args: dict, *, user: UserContext, approved: bool) -> dict:
    """Run an admin action after human approval.

    Raises AuthorizationError (non-admin role), pydantic ValidationError (bad args or
    unknown action) or PermissionError (no human approval recorded).
    """
    model = ADMIN_TOOLS.get(name)
    if model is None:
        raise ValueError(f"unknown admin tool {name!r}; expected one of {tuple(ADMIN_TOOLS)}")

    # Re-check at execution: binding the tool away from the model is not enough, the
    # role check must live at the effectful boundary.
    if not can_use(user.role, "admin"):
        raise AuthorizationError(f"role {user.role} may not perform admin actions")

    validated = model.model_validate(args).model_dump()
    if not approved:
        raise PermissionError("admin action requires human approval")
    action = describe_action(name, validated)
    result = {"status": "done", "action": action, "requested_by": user.user_id}
    # Audit trail: the action, its arguments, and who requested it.
    logger.info(
        "admin_action",
        tool=name,
        args=validated,
        requested_by=user.user_id,
        role=str(user.role),
    )
    return result
