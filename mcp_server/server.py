"""Compass directory MCP server ("compass-directory").

Why a separate read-only MCP server: structured enterprise data (people, services,
incidents) does not belong in the vector corpus, and an MCP server lets the agent
query it with typed tools while keeping the source data in one auditable place.

Three pure functions do all the work; FastMCP wraps them as tools. The functions stay
importable so tests (and the in-process fallback in `app.tools.mcp_client`) can call
them without spawning a stdio subprocess.

Data minimisation: the employee records on disk contain a `private` sub-object
(salary, national_id) to demonstrate that the tool layer must strip fields the
consumer is not entitled to see — the response is built field-by-field instead of
dumping the record.
"""

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("compass-directory")

# Loaded once at startup: the dataset is small and static for this POC, and reading
# it per call would add I/O for no benefit.
_DATA_DIR = Path(__file__).parent / "data"


def _load(filename: str) -> list[dict]:
    with (_DATA_DIR / filename).open(encoding="utf-8") as fh:
        return json.load(fh)


_EMPLOYEES: list[dict] = _load("employees.json")
_SERVICES: list[dict] = _load("services.json")
_INCIDENTS: list[dict] = _load("incidents.json")

MAX_EMPLOYEE_RESULTS = 5


def _validate_query(value: str) -> str:
    """Normalise and validate a free-text query.

    Strip, enforce 1..100 chars, reject control characters — so a model cannot smuggle
    newlines/invisible codepoints into what will be logged and substring-matched.
    """
    query = value.strip()
    if not 1 <= len(query) <= 100:
        raise ValueError("query must be 1 to 100 characters after trimming whitespace")
    if not query.isprintable():
        raise ValueError("query must not contain control characters")
    return query


def lookup_employee(query: str) -> list[dict]:
    """Case-insensitive match on name, employee_id or team; at most 5 results.

    Only work-contact fields are returned — salary/national_id never leave this module.
    """
    needle = _validate_query(query).casefold()
    matches = []
    for emp in _EMPLOYEES:
        haystacks = (emp["name"], emp["employee_id"], emp["team"])
        if any(needle in h.casefold() for h in haystacks):
            # Field-by-field copy: the `private` sub-object is deliberately dropped.
            matches.append(
                {
                    "employee_id": emp["employee_id"],
                    "name": emp["name"],
                    "title": emp["title"],
                    "team": emp["team"],
                    "department": emp["department"],
                    "email": emp["email"],
                    "on_call": bool(emp["on_call"]),
                }
            )
            if len(matches) >= MAX_EMPLOYEE_RESULTS:
                break
    return matches


def get_service_status(service: str) -> dict:
    """Exact (case-insensitive) service name lookup.

    Returns an error dict rather than raising so the model can react to the miss.
    """
    name = _validate_query(service)
    for svc in _SERVICES:
        if svc["service"].casefold() == name.casefold():
            return {
                "service": svc["service"],
                "owner_team": svc["owner_team"],
                "status": svc["status"],
                "last_incident": svc["last_incident"],
                "runbook": svc["runbook"],
            }
    return {"error": "unknown service"}


def list_open_incidents(department: str | None = None) -> list[dict]:
    """Incidents not yet resolved, optionally filtered by department."""
    dept = _validate_query(department) if department is not None else None
    return [
        {
            "incident_id": inc["incident_id"],
            "title": inc["title"],
            "service": inc["service"],
            "department": inc["department"],
            "severity": inc["severity"],
            "status": inc["status"],
            "opened": inc["opened"],
        }
        for inc in _INCIDENTS
        if inc["status"] != "resolved" and (dept is None or inc["department"] == dept)
    ]


# Registered together so the tool surface on the wire stays exactly these three.
mcp.tool()(lookup_employee)
mcp.tool()(get_service_status)
mcp.tool()(list_open_incidents)


if __name__ == "__main__":
    # stdio is the default for local agent use; streamable-http is handy when the
    # server runs as its own container in the demo deployment.
    if os.environ.get("MCP_TRANSPORT") == "streamable-http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
