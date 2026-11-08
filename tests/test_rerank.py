"""Rerank stage of hybrid search: ordering, per-doc cap, fallbacks and switch-offs.

No network: dense/sparse searchers and `hybrid.rerank` are stubbed with monkeypatch, like
tests/test_hybrid.py; an autouse fixture pins the fault/rerank env before each test.
"""

import pytest

from app.core.config import get_settings
from app.core.errors import VectorStoreError
from app.retrieval import hybrid
from app.retrieval.models import RetrievalQuery
from ingestion.models import AccessLevel


def hit(cid: str, score: float) -> tuple[str, float, dict]:
    fields = {
        "doc_id": cid.split("#")[0],
        "title": "t",
        "section": "s",
        "department": "payments",
        "document_type": "incident",
        "access_level": "internal",
        "created_date": "2026-01-01",
        "status": "final",
        "source_path": "p",
        "chunk_text": "text",
    }
    return cid, score, fields


def _dense(hits):
    async def search(_q):
        return hits

    return search


async def _no_sparse(_q):
    return []


async def _boom(_q):
    raise RuntimeError("pinecone down")


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    """Fresh, env-driven settings: RERANK_ENABLED=true and FAULT_PINECONE=false unless a test
    overrides them after this fixture (get_settings is lru_cached, so clear before and after)."""
    monkeypatch.setenv("FAULT_PINECONE", "false")
    monkeypatch.setenv("RERANK_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _rerank_recorder(calls: list):
    """Stub that records every call and returns the chunks unchanged.

    A raising stub cannot detect unwanted calls: hybrid_search catches every Exception
    around rerank and falls back to the fused order, so the assertion is swallowed too.
    """

    async def record(_query, chunks, top_n):
        calls.append((chunks, top_n))
        return chunks

    return record


REJECTED_NOTE = "Reranker unavailable; results use fused ranking."


async def test_reranked_order_replaces_fused_order(monkeypatch):
    dense = _dense([hit("A#00", 0.9), hit("B#00", 0.5)])
    monkeypatch.setattr(hybrid, "dense_search", dense)
    monkeypatch.setattr(hybrid, "sparse_search", _no_sparse)

    async def reverse_rerank(_query, chunks, top_n):
        assert top_n == len(chunks)
        rev = list(reversed(chunks))
        return [c.model_copy(update={"rerank_score": 0.9 - i * 0.1}) for i, c in enumerate(rev)]

    monkeypatch.setattr(hybrid, "rerank", reverse_rerank)
    q = RetrievalQuery(text="x", clearance=AccessLevel.internal)
    result = await hybrid.hybrid_search(q)
    assert [c.id for c in result.chunks] == ["B#00", "A#00"]  # not the fused [A#00, B#00]
    assert [c.rerank_score for c in result.chunks] == [0.9, 0.8]
    assert result.degraded is False and "rerank" in result.timings_ms


async def test_per_doc_cap_and_top_k_apply_to_reranked_order(monkeypatch):
    dense = _dense([hit("A#00", 0.9), hit("A#01", 0.8), hit("B#00", 0.5), hit("B#01", 0.4)])
    monkeypatch.setattr(hybrid, "dense_search", dense)
    monkeypatch.setattr(hybrid, "sparse_search", _no_sparse)

    async def rerank_b_first(_query, chunks, top_n):  # hybrid passes `top_n` by keyword
        order = [c for c in reversed(chunks)]  # B#01, B#00, A#01, A#00
        return [c.model_copy(update={"rerank_score": 0.9 - i * 0.1}) for i, c in enumerate(order)]

    monkeypatch.setattr(hybrid, "rerank", rerank_b_first)
    q = RetrievalQuery(text="x", clearance=AccessLevel.internal, top_k=3, max_per_doc=2)
    result = await hybrid.hybrid_search(q)
    # rerank was really used, not just a fused-order fallback
    assert all(c.rerank_score is not None for c in result.chunks) and "rerank" in result.timings_ms
    assert result.notes == []
    # The cut happens on the reranked order (fused order was [A#00, A#01, B#00]).
    assert [c.id for c in result.chunks] == ["B#01", "B#00", "A#01"]
    kept_per_doc: dict[str, int] = {}
    for c in result.chunks:
        kept_per_doc[c.doc_id] = kept_per_doc.get(c.doc_id, 0) + 1
    assert kept_per_doc == {"B": 2, "A": 1}


async def test_rerank_failure_keeps_fused_order_without_degrading(monkeypatch):
    dense = _dense([hit("A#00", 0.9), hit("B#00", 0.5)])
    monkeypatch.setattr(hybrid, "dense_search", dense)
    monkeypatch.setattr(hybrid, "sparse_search", _no_sparse)

    async def failing_rerank(_query, _chunks, top_n):  # hybrid passes `top_n` by keyword
        raise VectorStoreError("rerank timed out after 5.0s")

    monkeypatch.setattr(hybrid, "rerank", failing_rerank)
    q = RetrievalQuery(text="x", clearance=AccessLevel.internal)
    result = await hybrid.hybrid_search(q)
    assert [c.id for c in result.chunks] == ["A#00", "B#00"]  # fused order survives
    assert all(c.rerank_score is None for c in result.chunks)
    assert result.notes == [REJECTED_NOTE]
    assert result.degraded is False  # quality-only loss, unlike an index failure


async def test_query_can_switch_rerank_off(monkeypatch):
    rerank_calls: list = []
    monkeypatch.setattr(hybrid, "rerank", _rerank_recorder(rerank_calls))
    monkeypatch.setattr(hybrid, "dense_search", _dense([hit("A#00", 0.9)]))
    monkeypatch.setattr(hybrid, "sparse_search", _no_sparse)
    q = RetrievalQuery(text="x", clearance=AccessLevel.internal, rerank=False)
    result = await hybrid.hybrid_search(q)
    assert [c.id for c in result.chunks] == ["A#00"] and result.notes == []
    assert rerank_calls == [] and REJECTED_NOTE not in result.notes


async def test_rerank_disabled_in_settings_is_not_called(monkeypatch):
    rerank_calls: list = []
    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setattr(hybrid, "dense_search", _dense([hit("A#00", 0.9)]))
    monkeypatch.setattr(hybrid, "sparse_search", _no_sparse)
    monkeypatch.setattr(hybrid, "rerank", _rerank_recorder(rerank_calls))
    q = RetrievalQuery(text="x", clearance=AccessLevel.internal)
    result = await hybrid.hybrid_search(q)
    assert [c.id for c in result.chunks] == ["A#00"] and result.notes == []
    assert rerank_calls == [] and REJECTED_NOTE not in result.notes


async def test_fault_pinecone_never_reranks(monkeypatch):
    rerank_calls: list = []
    monkeypatch.setenv("FAULT_PINECONE", "true")
    monkeypatch.setattr(hybrid, "dense_search", _boom)  # must not be called either
    monkeypatch.setattr(hybrid, "sparse_search", _boom)
    monkeypatch.setattr(hybrid, "rerank", _rerank_recorder(rerank_calls))
    result = await hybrid.hybrid_search(RetrievalQuery(text="certificate", clearance=1))
    assert result.degraded and "FAULT_PINECONE" in result.notes[0]
    assert rerank_calls == [] and REJECTED_NOTE not in result.notes


async def test_bm25_fallback_never_reranks(monkeypatch):
    rerank_calls: list = []
    monkeypatch.setattr(hybrid, "dense_search", _boom)
    monkeypatch.setattr(hybrid, "sparse_search", _boom)
    monkeypatch.setattr(hybrid, "rerank", _rerank_recorder(rerank_calls))
    result = await hybrid.hybrid_search(
        RetrievalQuery(text="HikariPool connection", clearance=AccessLevel.internal)
    )
    assert result.degraded and all(c.source == "bm25_fallback" for c in result.chunks)
    assert rerank_calls == [] and REJECTED_NOTE not in result.notes
