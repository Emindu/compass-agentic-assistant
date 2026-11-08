"""Fusion maths and degradation paths, with Pinecone calls stubbed out (no network)."""

import pytest

from app.core.config import get_settings
from app.retrieval import hybrid
from app.retrieval.models import RetrievalQuery, build_pinecone_filter
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


def test_fuse_weights_normalised_scores():
    dense = [hit("A#00", 0.6), hit("B#00", 0.3)]
    sparse = [hit("B#00", 10.0), hit("C#00", 5.0)]
    chunks = hybrid.fuse(dense, sparse, alpha=0.5, top_k=10)
    by_id = {c.id: c for c in chunks}
    assert by_id["A#00"].fused_score == 0.5  # 0.5*1.0 + 0.5*0
    assert by_id["B#00"].fused_score == 0.75  # 0.5*0.5 + 0.5*1.0
    assert by_id["C#00"].fused_score == 0.25  # 0.5*0 + 0.5*0.5
    assert [c.id for c in chunks] == ["B#00", "A#00", "C#00"]
    assert by_id["B#00"].dense_score == 0.3 and by_id["B#00"].sparse_score == 10.0


def test_filter_always_starts_with_access_clause():
    q = RetrievalQuery(text="x", clearance=AccessLevel.public, document_types=["incident"])
    flt = build_pinecone_filter(q)
    assert flt["$and"][0] == {"access_rank": {"$lte": 0}}


@pytest.fixture
def no_fault(monkeypatch):
    monkeypatch.setenv("FAULT_PINECONE", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _ok(_q):
    return [hit("A#00", 0.5)]


async def _boom(_q):
    raise RuntimeError("pinecone down")


async def test_one_index_down_degrades_to_the_other(monkeypatch, no_fault):
    monkeypatch.setattr(hybrid, "dense_search", _boom)
    monkeypatch.setattr(hybrid, "sparse_search", _ok)
    result = await hybrid.hybrid_search(RetrievalQuery(text="x", clearance=AccessLevel.internal))
    assert result.degraded and [c.id for c in result.chunks] == ["A#00"]
    assert result.chunks[0].fused_score == 1.0  # alpha forced to 0: sparse only


async def test_both_down_falls_back_to_bm25(monkeypatch, no_fault):
    monkeypatch.setattr(hybrid, "dense_search", _boom)
    monkeypatch.setattr(hybrid, "sparse_search", _boom)
    q = RetrievalQuery(text="HikariPool connection", clearance=AccessLevel.internal)
    result = await hybrid.hybrid_search(q)
    assert result.degraded
    assert result.chunks and all(c.source == "bm25_fallback" for c in result.chunks)


async def test_fault_flag_skips_pinecone(monkeypatch):
    monkeypatch.setenv("FAULT_PINECONE", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(hybrid, "dense_search", _boom)  # must not be called
    result = await hybrid.hybrid_search(RetrievalQuery(text="certificate", clearance=1))
    get_settings.cache_clear()
    assert result.degraded and "FAULT_PINECONE" in result.notes[0]


def test_fuse_caps_chunks_per_document():
    dense = [hit("A#00", 0.9), hit("A#01", 0.8), hit("A#02", 0.7), hit("B#00", 0.5)]
    chunks = hybrid.fuse(dense, [], alpha=1.0, top_k=3, max_per_doc=2)
    assert [c.id for c in chunks] == ["A#00", "A#01", "B#00"]
