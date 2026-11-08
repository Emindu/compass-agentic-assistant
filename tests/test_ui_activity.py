"""Unit tests for the activity-panel reducer against realistic event sequences."""

from __future__ import annotations

from typing import Any

from ui.activity import TurnState, apply, format_status_badge, render_citations
from ui.client import StreamEvent


def ev(event_type: str, node: str | None = None, data: dict[str, Any] | None = None) -> StreamEvent:
    return StreamEvent(type=event_type, node=node, data=data or {}, ts=0.0)


def titles(state: TurnState) -> list[str]:
    return [step.title for step in state.steps]


def test_realistic_turn_sequence() -> None:
    events = [
        ev("node_start", "load_memory"),
        ev("memory", "load_memory", {"action": "loaded", "previous_turns": 2, "has_summary": True}),
        ev("node_start", "supervisor"),
        ev(
            "route",
            "supervisor",
            {
                "intent": "deep_research",
                "agents": ["research", "analysis"],
                "search_query": "payments ledger mismatch",
                "departments": None,
                "reasoning": "needs document evidence",
            },
        ),
        ev("node_end", "supervisor", {"status": "ok", "duration_ms": 12.5}),
        ev(
            "retrieval",
            "retrieval",
            {
                "status": "searching",
                "query": "payments ledger",
                "departments": ["payments", "platform"],
                "clearance": "internal",
            },
        ),
        ev(
            "retrieval",
            "retrieval",
            {
                "status": "done",
                "notes": [],
                "timings_ms": {"dense": 30.2},
                "hits": [
                    {
                        "id": "c1",
                        "citation": "[1]",
                        "title": "Payments Runbook",
                        "access_level": "internal",
                        "dense": 0.8123456,
                        "sparse": 0.1234567,
                        "fused": 0.7778901,
                        "source": "pinecone",
                    },
                    {
                        "id": "c2",
                        "citation": "[2]",
                        "title": "Ledger Spec",
                        "access_level": "internal",
                        "dense": None,
                        "sparse": 0.5,
                        "fused": 0.42,
                        "source": "bm25",
                    },
                ],
            },
        ),
        ev(
            "validation",
            "validator",
            {"valid": True, "cited": ["c1", "c2"], "unknown": [], "issues": []},
        ),
        ev("token", "response", {"text": "Hel"}),
        ev("token", "response", {"text": "lo"}),
        ev("token", "response", {"text": " world"}),
        ev(
            "answer",
            "response",
            {
                "text": "Hello world (validated)",
                "citations": [
                    {"id": "c1", "doc_id": "d1", "title": "Payments Runbook", "section": "2.1"}
                ],
                "notes": ["ssss fallback used"],
                "errors": [],
            },
        ),
        ev("done", None, {"thread_id": "t-42"}),
    ]

    state = TurnState()
    for event in events:
        state = apply(state, event)

    assert state.answer_text == "Hello world (validated)"
    assert state.citations[0]["id"] == "c1"
    assert state.notes == ["ssss fallback used"]
    assert state.done is True
    assert state.thread_id == "t-42"

    # node_end updated the running supervisor step
    supervisor_steps = [
        s for s in state.steps if s.node == "supervisor" and s.title.startswith("🧭 Supervisor")
    ]
    assert supervisor_steps and supervisor_steps[0].status == "ok"
    assert "12 ms" in supervisor_steps[0].title
    assert state.node_timings["supervisor"] == 12.5

    # retrieval hits table with rounded scores
    retrieval_done = [s for s in state.steps if s.table][0]
    assert retrieval_done.table is not None
    assert len(retrieval_done.table) == 2
    row = retrieval_done.table[0]
    assert row["dense"] == 0.812
    assert row["sparse"] == 0.123
    assert row["fused"] == 0.778
    assert row["citation"] == "[1]"
    assert row["access"] == "internal"
    assert len(state.hits) == 2

    assert "🧭 Routed: deep_research → research, analysis" in titles(state)
    assert "🔎 Searching payments, platform (clearance: internal)" in titles(state)
    assert "📚 2 chunks (hybrid)" in titles(state)


def test_tokens_after_response_retry_replace_streamed_text() -> None:
    state = TurnState()
    for event in [
        ev("node_start", "response"),
        ev("token", "response", {"text": "first "}),
        ev("token", "response", {"text": "attempt"}),
        ev("node_start", "response"),
        ev("token", "response", {"text": "second "}),
        ev("token", "response", {"text": "attempt"}),
    ]:
        state = apply(state, event)

    assert state.answer_text == "second attempt"


def test_research_events() -> None:
    state = TurnState()
    plan_code = "hits = search('payments')\n"
    for event in [
        ev("research", "research", {"stage": "plan", "depth": 2, "code": plan_code}),
        ev(
            "research",
            "research",
            {
                "stage": "batches",
                "documents": ["d1", "d2", "d3", "d4"],
                "batch_sizes": [2, 2],
                "data": {},
            },
        ),
        ev(
            "research",
            "research",
            {
                "stage": "batch_done",
                "index": 0,
                "findings": [
                    {
                        "doc_id": "d1",
                        "summary": "s",
                        "root_cause": "rc",
                        "category": "config",
                        "evidence": ["c1"],
                    },
                ],
            },
        ),
        ev(
            "research",
            "research",
            {
                "stage": "reduce",
                "depth": 2,
                "themes": ["reconciliation", "timing"],
                "follow_up": "zoom on batch 1",
            },
        ),
    ]:
        state = apply(state, event)

    assert state.research_plan == plan_code
    plan_step = next(s for s in state.steps if "RLM plan" in s.title)
    assert plan_step.code == plan_code
    assert plan_step.title == "🧠 RLM plan (depth 2)"

    assert "🧩 2 batches over 4 documents" in titles(state)

    batch_step = next(s for s in state.steps if "Batch 0" in s.title)
    assert batch_step.title == "✅ Batch 0: 1 findings"
    assert batch_step.detail is not None and "d1 — config" in batch_step.detail

    reduce_step = next(s for s in state.steps if s.title.startswith("🔗 Reduce"))
    assert "reconciliation, timing" in reduce_step.title
    assert reduce_step.detail is not None and "zoom on batch 1" in reduce_step.detail


def test_tool_call_denied() -> None:
    state = apply(
        TurnState(),
        ev(
            "tool_call",
            "analysis",
            {
                "tool": "python_analysis",
                "status": "denied",
                "reason": "viewers may not execute code",
            },
        ),
    )
    denied_step = state.steps[-1]
    assert "denied" in denied_step.title
    assert "python_analysis" in denied_step.title
    assert denied_step.detail == "viewers may not execute code"
    assert denied_step.status == "denied"


def test_tool_call_ok_with_primitive_calls_table() -> None:
    state = apply(
        TurnState(),
        ev(
            "tool_call",
            "analysis",
            {
                "tool": "python_analysis",
                "status": "ok",
                "duration_ms": 220.4,
                "primitive_calls": [
                    {
                        "name": "search",
                        "kwargs": {"query": "payments"},
                        "duration_ms": 18.6,
                        "ok": True,
                    },
                ],
            },
        ),
    )
    step = state.steps[-1]
    assert step.table is not None
    row = step.table[0]
    assert row == {"primitive": "search", "args": '{"query": "payments"}', "ms": 18.6, "ok": True}


def test_unknown_event_and_empty_data_never_raise() -> None:
    state = TurnState()
    apply(state, ev("brand_new", None, None))
    apply(state, ev("node_end", "ghost_node", {"status": "degraded", "duration_ms": 1.0}))
    apply(state, ev("validation", "validator", {}))
    apply(state, ev("retrieval", "retrieval", {"status": "degraded", "notes": ["pinecone down"]}))
    apply(state, ev("done", None, {}))

    assert state.done is True
    unknown_step = next(s for s in state.steps if s.icon == "❔")
    assert unknown_step.title == "❔ brand_new"
    assert not any(s.status == "running" and s.node == "ghost_node" for s in state.steps)


def test_invalid_validation_lists_unknown_and_issues() -> None:
    state = apply(
        TurnState(),
        ev(
            "validation",
            "validator",
            {
                "valid": False,
                "cited": ["c1"],
                "unknown": ["cX"],
                "issues": ["hallucinated citation"],
            },
        ),
    )
    step = state.steps[-1]
    assert step.status == "error"
    assert step.detail is not None
    assert "cX" in step.detail
    assert "hallucinated citation" in step.detail


def test_render_citations_format() -> None:
    markdown = render_citations(
        [
            {"id": "r1", "doc_id": "d1", "title": "Payments Runbook", "section": "2.1"},
            {"id": "r2", "doc_id": "d2", "title": "Ledger Spec", "section": None},
        ]
    )
    assert markdown.splitlines()[0] == "- **[r1]** Payments Runbook — 2.1"
    assert markdown.splitlines()[1] == "- **[r2]** Ledger Spec"
    assert format_status_badge("viewer") == "👤 Viewer"
    assert format_status_badge("admin") == "👑 Admin"


# --- approval round-trip -------------------------------------------------------------------


def test_approval_event_sets_pending_approval() -> None:
    state = apply(
        TurnState(),
        ev(
            "approval",
            "tools",
            {
                "type": "approval",
                "tool": "reindex_department",
                "args": {"department": "payments"},
                "action": "Re-index all payments documents",
                "requested_by": "ada",
                "thread_id": "t-42",
            },
        ),
    )

    assert state.pending_approval is not None
    assert state.pending_approval["tool"] == "reindex_department"
    assert state.pending_approval["thread_id"] == "t-42"
    step = state.steps[-1]
    assert step.title == "⏸️ Approval required: Re-index all payments documents"
    assert step.detail is not None
    assert "reindex_department" in step.detail
    assert '"department": "payments"' in step.detail


def test_approval_event_with_missing_keys_never_raises() -> None:
    state = apply(TurnState(), ev("approval", "tools", {}))
    assert state.pending_approval == {}
    assert state.steps[-1].title == "⏸️ Approval required: requested tool action"


def test_tool_call_pending_approval_and_rejected_status() -> None:
    state = TurnState()
    apply(
        state,
        ev(
            "tool_call",
            "tools",
            {
                "tool": "reindex_department",
                "status": "pending_approval",
                "args": {"department": "payments"},
                "action": "Re-index all payments documents",
                "why": "requested by user",
            },
        ),
    )
    pending_step = state.steps[-1]
    assert pending_step.title == "⏸️ reindex_department awaiting approval"
    assert pending_step.detail == "Re-index all payments documents"

    apply(
        state,
        ev(
            "tool_call",
            "tools",
            {
                "tool": "reindex_department",
                "status": "rejected",
                "args": {"department": "payments"},
            },
        ),
    )
    rejected_step = state.steps[-1]
    assert rejected_step.title == "🚫 reindex_department rejected"
    assert rejected_step.status == "rejected"


# --- guardrail stages ------------------------------------------------------------------------


def test_guardrail_input_stage_allowed_and_blocked() -> None:
    allowed = apply(
        TurnState(),
        ev("guardrail", "input_guard", {"stage": "input", "allowed": True, "layer": "pattern"}),
    )
    assert allowed.steps[-1].title == "🛡️ Input guard: allowed"
    assert allowed.steps[-1].status == "ok"

    blocked = apply(
        TurnState(),
        ev(
            "guardrail",
            "input_guard",
            {
                "stage": "input",
                "allowed": False,
                "category": "prompt_injection",
                "layer": "pattern",
                "matched": ["instruction_override"],
            },
        ),
    )
    assert blocked.steps[-1].title == "🛡️ Input guard: blocked (prompt_injection, pattern)"
    assert blocked.steps[-1].status == "error"


def test_guardrail_retrieved_text_stage_sanitised() -> None:
    state = apply(
        TurnState(),
        ev(
            "guardrail",
            "response",
            {
                "stage": "retrieved_text",
                "action": "sanitised",
                "chunks": ["INC-1#01", "RUN-2#02"],
                "findings": ["document_tag_neutralised", "prompt_like_line_removed"],
            },
        ),
    )
    step = state.steps[-1]
    assert step.title == "🧼 Sanitised 2 retrieved chunk(s)"
    assert step.detail is not None
    assert "document_tag_neutralised" in step.detail


def test_guardrail_output_stage_codes() -> None:
    state = apply(
        TurnState(),
        ev(
            "guardrail",
            "validator",
            {
                "stage": "output",
                "issues": [
                    {"code": "financial_advice", "message": "…", "severity": "block"},
                    {"code": "brand_voice", "message": "…", "severity": "warn"},
                ],
            },
        ),
    )
    assert state.steps[-1].title == "🛡️ Output guard: financial_advice, brand_voice"


def test_guardrail_without_stage_keeps_generic_handling() -> None:
    state = apply(
        TurnState(), ev("guardrail", "validator", {"action": "replaced_unsupported_answer"})
    )
    step = state.steps[-1]
    assert step.title == "🛡️ Guardrail: replaced_unsupported_answer"
    assert step.status == "ok"


def test_guardrail_stage_events_with_empty_data_never_raise() -> None:
    state = TurnState()
    apply(state, ev("guardrail", "input_guard", {"stage": "input"}))
    apply(state, ev("guardrail", "response", {"stage": "retrieved_text"}))
    apply(state, ev("guardrail", "validator", {"stage": "output"}))
    apply(state, ev("guardrail", "response", {}))
    apply(state, ev("tool_call", "tools", {"tool": "", "status": ""}))
    assert len(state.steps) == 5
