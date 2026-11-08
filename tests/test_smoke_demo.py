"""Unit tests for the smoke-demo assertion helpers and CLI filtering.

Network-free: every check is exercised against canned `StreamEvent` lists that mimic
what the API streams over SSE (`ui.client._parse_event` shape), so no API needs to run.
"""

import pytest

from scripts.smoke_demo import (
    CASES,
    answer_text,
    check_approval,
    check_cert,
    check_injection,
    check_mcp,
    check_memory,
    check_rbac,
    check_rlm,
    cited_ids,
    find_first,
    load_ground_truth,
    parse_args,
    selected_cases,
)
from ui.client import StreamEvent

IN_SCOPE = [
    "INC-2025-1207",
    "INC-2026-0125",
    "INC-2026-0214",
    "INC-2026-0304",
    "INC-2026-0525",
    "INC-2026-0712",
    "INC-2026-0819",
    "INC-2026-0925",
]
MUST_EXCLUDE = {
    "INC-2024-0815": "older than one year",
    "INC-2026-0712-DRAFT": "draft superseded",
    "INC-2026-0611": "core banking, not payments",
}
RLM_GROUND_TRUTH = {
    "rlm_payment_outages": {
        "in_scope_incidents": IN_SCOPE,
        "must_exclude": MUST_EXCLUDE,
    }
}


def ev(type_: str, data: dict | None = None, node: str | None = None) -> StreamEvent:
    """One canned SSE event in the shape `ui.client` yields."""
    return StreamEvent(type=type_, node=node, data=data or {}, ts=None)


def answer(citations: list[str], text: str = "Answer.") -> StreamEvent:
    return ev("answer", {"text": text, "citations": [{"id": c} for c in citations]})


def batches(documents: list[str]) -> StreamEvent:
    return ev("research", {"stage": "batches", "documents": documents})


# --- cert / memory: citation prefixes ---


def test_cert_pass() -> None:
    ok, _ = check_cert([batches(["X"]), answer(["RB-PAY-003-V2#2"])])
    assert ok


def test_cert_fail_cites_superseded_v1() -> None:
    ok, reason = check_cert([answer(["RB-PAY-003-V1#2"])])
    assert not ok
    assert "RB-PAY-003-V2" in reason


def test_cert_fail_no_answer_event() -> None:
    ok, _ = check_cert([ev("route", {"agents": ["retrieval"]})])
    assert not ok


def test_memory_pass() -> None:
    ok, _ = check_memory([answer(["RB-PAY-003-V1#1"])])
    assert ok


def test_memory_fail() -> None:
    ok, _ = check_memory([answer(["RB-PAY-003-V2#1"])])
    assert not ok


def test_cited_ids_tolerates_missing_and_malformed_keys() -> None:
    events = [
        answer(["RB-PAY-003-V2#1"]),
        ev("answer", {"text": "no citations key"}),
        ev("answer", {"citations": [{"doc_id": "no-id"}, "junk"]}),
    ]
    assert cited_ids(events) == ["RB-PAY-003-V2#1"]


# --- rlm: document set and exclusions ---


def test_rlm_pass() -> None:
    stream = [batches(IN_SCOPE), answer(IN_SCOPE[:3])]
    ok, _ = check_rlm(stream, RLM_GROUND_TRUTH)
    assert ok


def test_rlm_fail_documents_differ() -> None:
    stream = [batches(IN_SCOPE[:-1] + ["INC-2026-1211"]), answer(IN_SCOPE[:3])]
    ok, reason = check_rlm(stream, RLM_GROUND_TRUTH)
    assert not ok
    assert "INC-2026-0925" in reason  # missing in-scope incident
    assert "INC-2026-1211" in reason  # unexpected incident


def test_rlm_fail_excluded_incident_cited() -> None:
    stream = [batches(IN_SCOPE), answer(["INC-2026-0712-DRAFT"])]
    ok, reason = check_rlm(stream, RLM_GROUND_TRUTH)
    assert not ok
    assert "INC-2026-0712-DRAFT" in reason


def test_rlm_fail_no_batches_event() -> None:
    ok, _ = check_rlm([answer(IN_SCOPE)], RLM_GROUND_TRUTH)
    assert not ok


def test_rlm_fail_missing_ground_truth() -> None:
    ok, _ = check_rlm([batches(IN_SCOPE)], {})
    assert not ok


# --- rbac: denied tool call, no research ---


def test_rbac_pass() -> None:
    denied = ev("tool_call", {"tool": "python_analysis", "status": "denied"})
    ok, _ = check_rbac([denied, answer([], "Document search only.")])
    assert ok


def test_rbac_fail_no_denied_call() -> None:
    ok, _ = check_rbac([answer([], "Answer from documents.")])
    assert not ok


def test_rbac_fail_research_ran_anyway() -> None:
    denied = ev("tool_call", {"tool": "python_analysis", "status": "denied"})
    ok, _ = check_rbac([denied, batches(IN_SCOPE), answer(IN_SCOPE)])
    assert not ok


# --- mcp: successful MCP tool call and on-call name ---


def test_mcp_pass() -> None:
    call = ev("tool_call", {"tool": "get_service_status", "status": "ok"})
    ok, _ = check_mcp([call, answer([], "Nimal is on call for the connector.")])
    assert ok


def test_mcp_fail_non_mcp_tool_ok() -> None:
    call = ev("tool_call", {"tool": "python_analysis", "status": "ok"})
    ok, _ = check_mcp([call, answer([], "Nimal is on call.")])
    assert not ok


def test_mcp_fail_name_missing() -> None:
    call = ev("tool_call", {"tool": "lookup_employee", "status": "ok"})
    ok, _ = check_mcp([call, answer([], "Someone else is on call.")])
    assert not ok


# --- approval: event then rejected resume ---


def test_approval_pass() -> None:
    first = [
        ev("tool_call", {"tool": "reindex_payments", "status": "pending_approval"}),
        ev("approval", {"tool": "reindex_payments", "thread_id": "t1"}),
        ev("done", {"thread_id": "t1"}),
    ]
    resume = [
        ev("tool_call", {"tool": "reindex_payments", "status": "rejected"}),
        ev("done", {"thread_id": "t1"}),
    ]
    ok, _ = check_approval(first, resume)
    assert ok


def test_approval_fail_no_event() -> None:
    ok, _ = check_approval([ev("done", {"thread_id": "t1"})], [])
    assert not ok


def test_approval_fail_resume_not_rejected() -> None:
    first = [ev("approval", {"thread_id": "t1"})]
    resume = [ev("tool_call", {"tool": "reindex_payments", "status": "ok"})]
    ok, _ = check_approval(first, resume)
    assert not ok


# --- injection: input guard blocks, prompt never routed ---


def test_injection_pass() -> None:
    events = [
        ev("guardrail", {"stage": "input", "allowed": False, "category": "prompt_injection"}),
        answer([], "I can't help with that."),
        ev("done", {"thread_id": "t1"}),
    ]
    ok, _ = check_injection(events)
    assert ok


def test_injection_fail_guard_allowed() -> None:
    events = [
        ev("guardrail", {"stage": "input", "allowed": True}),
        ev("route", {"agents": ["retrieval"]}),
    ]
    ok, _ = check_injection(events)
    assert not ok


def test_injection_fail_route_ran() -> None:
    events = [
        ev("guardrail", {"stage": "input", "allowed": False}),
        ev("route", {"agents": ["retrieval"]}),
    ]
    ok, _ = check_injection(events)
    assert not ok


def test_injection_fail_no_guardrail() -> None:
    ok, _ = check_injection([ev("route", {"agents": []}), ev("done", {})])
    assert not ok


# --- helpers ---


def test_answer_text_prefers_answer_event_over_tokens() -> None:
    events = [ev("token", {"text": "part "}), ev("token", {"text": "tokens"}), answer([], "Final.")]
    assert answer_text(events) == "Final."


def test_answer_text_falls_back_to_token_stream() -> None:
    events = [ev("token", {"text": "Nimal "}), ev("token", {"text": "Perera"})]
    assert answer_text(events) == "Nimal Perera"


def test_find_first_matches_and_tolerates_absent_fields() -> None:
    events = [ev("tool_call", {"status": "failed"}), ev("tool_call", {"tool": "x", "status": "ok"})]
    assert find_first(events, "tool_call", status="ok") is events[1]
    assert find_first(events, "tool_call", tool="y") is None


def test_load_ground_truth_reads_repo_file() -> None:
    gt = load_ground_truth()
    assert "question" in gt.get("rlm_payment_outages", {})


# --- CLI: --only filtering ---


def test_selected_cases_defaults_to_all_in_order() -> None:
    selected = selected_cases(None)
    assert [name for name, _case in selected] == list(CASES)


def test_case_registry_covers_the_demo_scenarios() -> None:
    assert set(CASES) == {"cert", "memory", "rlm", "rbac", "mcp", "approval", "injection"}
    assert CASES["cert"].user == "vera"
    assert CASES["approval"].user == "ada"


def test_selected_cases_only_picks_one() -> None:
    selected = selected_cases("rbac")
    assert len(selected) == 1
    assert selected[0][1] is CASES["rbac"]


def test_selected_cases_unknown_name_exits() -> None:
    with pytest.raises(SystemExit):
        selected_cases("nope")


def test_parse_args_defaults_and_overrides() -> None:
    default = parse_args([])
    assert default.base == "http://localhost:8000"
    assert default.only is None
    args = parse_args(["--base", "http://api:8000", "--only", "cert"])
    assert args.base == "http://api:8000"
    assert args.only == "cert"
