"""Citation extraction, validation and fallback stripping — pure functions, no network."""

from app.agents.nodes.validator import (
    extract_citations,
    route_after_validation,
    strip_unknown_citations,
    validate_answer,
    validator_node,
)
from app.agents.state import AgentState, RouteDecision, ValidationResult
from app.retrieval.models import RetrievedChunk


def chunk(cid: str) -> RetrievedChunk:
    doc_id, _, section = cid.partition("#")
    return RetrievedChunk(
        id=cid,
        doc_id=doc_id,
        title="Runbook",
        section=section or "01",
        department="payments",
        document_type="incident",
        access_level="public",
        created_date="2026-01-01",
        status="final",
        source_path="docs/runbook.md",
        text="Retries had no backoff.",
    )


def routed() -> RouteDecision:
    return RouteDecision(
        intent="knowledge_qa",
        agents=["retrieval"],
        search_query="retries",
        reasoning="knowledge question",
    )


def test_extract_citations_separate_brackets():
    assert extract_citations("Retries [INC-1#01] and [INC-2#02]") == ["INC-1#01", "INC-2#02"]


def test_extract_citations_comma_list():
    assert extract_citations("Retries [INC-1#01, INC-2#02]") == ["INC-1#01", "INC-2#02"]


def test_extract_citations_unique_in_order():
    answer = "One [INC-1#01] two [INC-1#01] three [INC-2#02] and [INC-1#01, INC-2#02]"
    assert extract_citations(answer) == ["INC-1#01", "INC-2#02"]


def test_extract_citations_ignores_non_ids():
    answer = "See the runbook [note] and step [1], and [IN#1] and [INC-2026-0214#03]."
    # prose brackets are ignored; malformed ids are still treated as citations (to be rejected)
    assert extract_citations(answer) == ["IN#1", "INC-2026-0214#03"]


def test_valid_answer_citing_retrieved_ids():
    retrieved = [chunk("INC-1#01"), chunk("INC-2#02")]
    result = validate_answer("Retries had no backoff [INC-1#01][INC-2#02].", retrieved, True)
    assert result.valid is True
    assert result.issues == []
    assert result.cited_ids == ["INC-1#01", "INC-2#02"]
    assert result.unknown_citations == []


def test_unknown_citation_is_hallucination():
    result = validate_answer("Retries [FAKE-1#01].", [chunk("INC-1#01")], True)
    assert result.valid is False
    assert result.unknown_citations == ["FAKE-1#01"]
    assert result.cited_ids == []
    assert any("FAKE-1#01" in issue for issue in result.issues)


def test_missing_citations_when_evidence_required():
    result = validate_answer("Retries had no backoff.", [chunk("INC-1#01")], True)
    assert result.valid is False
    assert result.issues


def test_evidence_required_nothing_retrieved_candid_answer_ok():
    answer = "I couldn't find any information about that in the knowledge base."
    result = validate_answer(answer, [], True)
    assert result.valid is True
    assert result.issues == []


def test_evidence_required_nothing_retrieved_confident_answer_fails():
    result = validate_answer("Retries retry five times.", [], True)
    assert result.valid is False
    assert result.issues


def test_smalltalk_without_citations_is_valid():
    result = validate_answer("Hello! How can I help?", [], False)
    assert result.valid is True


def test_empty_answer_is_invalid():
    result = validate_answer("  ", [chunk("INC-1#01")], True)
    assert result.valid is False
    assert result.issues


def test_strip_unknown_citations_removes_fake_keeps_real():
    answer = "Backoff was missing [FAKE-1#01] and retries [INC-2026-0214#01, FAKE-1#01]."
    stripped = strip_unknown_citations(answer, ["FAKE-1#01"])
    assert "FAKE-1#01" not in stripped
    assert "[INC-2026-0214#01]" in stripped
    assert "[]" not in stripped
    assert stripped == "Backoff was missing and retries [INC-2026-0214#01]."


def test_strip_unknown_citations_only_fake_bracket_removed():
    answer = "Known [INC-1#01] and fake [FAKE-1#01]."
    stripped = strip_unknown_citations(answer, ["FAKE-1#01"])
    assert stripped == "Known [INC-1#01] and fake."
    assert strip_unknown_citations(answer, []) == answer


def _state(
    answer: str,
    retrieved: list[RetrievedChunk],
    attempts: int,
    validation: ValidationResult | None = None,
) -> AgentState:
    return AgentState(
        draft_answer=answer,
        retrieved=retrieved,
        route=routed(),
        response_attempts=attempts,
        validation=validation,
    )


def test_route_after_validation_retries_until_budget_exhausted():
    invalid = ValidationResult(valid=False, issues=["cited references that were never retrieved"])
    assert route_after_validation(_state("x", [], 1, invalid)) == "response"
    assert route_after_validation(_state("x", [], 2, invalid)) == "finalize"
    valid = ValidationResult(valid=True)
    assert route_after_validation(_state("x", [], 0, valid)) == "finalize"


async def test_validator_node_strips_fake_id_at_max_attempts():
    state = _state(
        "Retries had no backoff [INC-2026-0214#01][FAKE-1#01].",
        [chunk("INC-2026-0214#01")],
        attempts=2,
    )
    update = await validator_node(state)
    assert update["validation"].valid is False
    draft = update["draft_answer"]
    assert "FAKE-1#01" not in draft
    assert "Retries had no backoff [INC-2026-0214#01]" in draft
    assert "_Note: some references could not be verified" in draft


async def test_validator_node_without_stripping_when_valid():
    state = _state("Retries had no backoff [INC-1#01].", [chunk("INC-1#01")], attempts=0)
    update = await validator_node(state)
    assert update["validation"].valid is True
    assert "draft_answer" not in update


async def test_unsupported_answer_is_replaced_after_retries(monkeypatch):
    from app.agents.nodes import validator as v
    from app.agents.state import RouteDecision

    monkeypatch.setattr(v, "emit", lambda *a, **k: None)
    route = RouteDecision(
        intent="deep_research", agents=["research"], search_query="q", reasoning="r"
    )
    state = {
        "draft_answer": "Third-party outages caused most failures [3#01][7#02].",
        "retrieved": [],
        "route": route,
        "response_attempts": v.MAX_RESPONSE_ATTEMPTS,
    }
    out = await v.validator_node(state)
    assert out["validation"].unknown_citations == ["3#01", "7#02"]  # malformed ids caught
    assert out["draft_answer"] == v.NO_EVIDENCE_ANSWER
