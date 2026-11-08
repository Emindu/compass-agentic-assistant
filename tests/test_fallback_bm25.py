from datetime import date

from app.retrieval.fallback_bm25 import get_bm25_fallback, tokenize
from app.retrieval.models import RetrievalQuery
from ingestion.models import AccessLevel


def _search(text: str, **kwargs) -> list:
    return get_bm25_fallback().search(RetrievalQuery(text=text, **kwargs))


def test_tokenize_keeps_compound_ids_whole_and_yields_their_parts():
    tokens = tokenize("INC-2026-0214")
    assert "inc-2026-0214" in tokens
    assert {"inc", "2026", "0214"} <= set(tokens)


def test_connection_pool_query_finds_ledger_incident_first():
    results = _search("HikariPool Connection is not available", clearance=AccessLevel.internal)
    assert results[0].doc_id == "INC-2026-0125"


def test_rbac_hides_confidential_from_internal_but_not_from_confidential():
    text = "Q3 operational risk review compensation"
    internal = _search(text, clearance=AccessLevel.internal)
    assert internal
    assert all(r.access_level not in {"confidential", "restricted"} for r in internal)

    confidential = _search(text, clearance=AccessLevel.confidential)
    assert any(r.doc_id == "MTG-PAY-2026-10" for r in confidential)


def test_restricted_policy_only_visible_at_restricted_clearance():
    text = "break-glass accounts"
    assert all(r.doc_id != "POL-SEC-004" for r in _search(text, clearance=AccessLevel.confidential))
    assert any(r.doc_id == "POL-SEC-004" for r in _search(text, clearance=AccessLevel.restricted))


def test_metadata_filters_are_applied_in_python():
    platform = _search(
        "core banking end-of-day batch overran business hours",
        clearance=AccessLevel.internal,
        departments=["platform"],
    )
    assert platform
    assert all(r.department == "platform" for r in platform)

    recent = _search(
        "gateway connection failure expired certificate",
        clearance=AccessLevel.internal,
        date_from=date(2025, 11, 8),
    )
    assert recent
    assert all(r.doc_id != "INC-2024-0815" for r in recent)

    no_drafts = _search(
        "bill payment failures 12 July preliminary",
        clearance=AccessLevel.internal,
        exclude_statuses=["draft"],
    )
    assert no_drafts
    assert all(r.doc_id != "INC-2026-0712-DRAFT" for r in no_drafts)


def test_results_are_normalised_sorted_and_attributed_to_fallback():
    for clearance in (AccessLevel.internal, AccessLevel.restricted):
        results = _search("payment gateway certificate rotation", clearance=clearance)
        assert results
        for chunk in results:
            assert chunk.source == "bm25_fallback"
            assert chunk.dense_score is None
            assert chunk.sparse_score is not None and chunk.sparse_score > 0
            assert 0 < chunk.fused_score <= 1
        scores = [r.fused_score for r in results]
        assert scores == sorted(scores, reverse=True)
