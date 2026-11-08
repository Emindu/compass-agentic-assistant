"""Tests for the compass-directory MCP server's pure functions. Network-free: no server
process is started, the FastMCP registration is exercised implicitly by import."""

import pytest

from mcp_server import server


def test_lookup_employee_by_name_is_case_insensitive():
    results = server.lookup_employee("NIMAL PERERA")
    assert [r["employee_id"] for r in results] == ["EMP-001"]
    assert results[0]["name"] == "Nimal Perera"
    assert results[0]["title"] == "Senior SRE Engineer"
    assert results[0]["on_call"] is True


def test_lookup_employee_by_employee_id_and_team():
    assert [r["employee_id"] for r in server.lookup_employee("EMP-003")] == ["EMP-003"]
    by_team = {r["employee_id"] for r in server.lookup_employee("payments-sre")}
    assert by_team == {"EMP-001", "EMP-002"}


def test_lookup_employee_matches_partial_and_strips_whitespace():
    assert [r["employee_id"] for r in server.lookup_employee("  nimal  ")] == ["EMP-001"]
    assert [r["employee_id"] for r in server.lookup_employee("kavya")] == ["EMP-002"]


def test_lookup_employee_never_returns_private_fields():
    for record in server.lookup_employee("payments"):
        # Field-by-field response: salary / national_id must not leak even though the
        # JSON on disk contains a `private` sub-object.
        assert set(record) == {
            "employee_id",
            "name",
            "title",
            "team",
            "department",
            "email",
            "on_call",
        }


def test_lookup_employee_caps_results_at_five():
    # Six employees sit in payments-* teams; only the first five matches are returned.
    assert len(server.lookup_employee("payments")) == 5


@pytest.mark.parametrize("bad", ["", "   ", "a" * 101, "a\nb", "\x00nimal", "perera\x1b"])
def test_lookup_employee_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        server.lookup_employee(bad)


def test_get_service_status_known_service():
    status = server.get_service_status("card-switch-connector")
    assert status == {
        "service": "card-switch-connector",
        "owner_team": "payments-sre",
        "status": "operational",
        "last_incident": "INC-2026-0819",
        "runbook": "RB-PAY-003-V2",
    }


def test_get_service_status_is_case_insensitive():
    assert server.get_service_status("CARD-SWITCH-CONNECTOR") == server.get_service_status(
        "card-switch-connector"
    )


def test_get_service_status_unknown_service_returns_error_dict():
    # An error dict instead of an exception: the model can react to the miss.
    assert server.get_service_status("swift-gateway") == {"error": "unknown service"}


def test_list_open_incidents_returns_only_unresolved():
    open_ids = {inc["incident_id"] for inc in server.list_open_incidents()}
    assert open_ids == {"INC-2026-1002", "INC-2026-0925"}
    for inc in server.list_open_incidents():
        assert inc["status"] != "resolved"


def test_list_open_incidents_filters_by_department():
    assert {inc["incident_id"] for inc in server.list_open_incidents("payments")} == {
        "INC-2026-1002",
        "INC-2026-0925",
    }
    # Nothing unresolved outside payments in the fictional dataset.
    assert server.list_open_incidents("platform") == []
    assert server.list_open_incidents("retail") == []
