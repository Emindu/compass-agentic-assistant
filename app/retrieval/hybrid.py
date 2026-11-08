"""Hybrid retrieval: dense + sparse in parallel, weighted score fusion, graceful degradation.

Fusion: each list's scores are divided by that list's top score (max-normalisation), then
    fused = alpha * dense_norm + (1 - alpha) * sparse_norm
with a missing score counted as 0. Why max- rather than min-max normalisation: min-max forces
the weakest hit of each list to 0 even when it is relevant, and cosine (≈0.3–0.6) and sparse
dot-product (unbounded, ≈1–10) only need to be put on a common 0..1 scale. Why weighted fusion
rather than reciprocal-rank fusion: the weight `alpha` is one explainable knob, and the UI can
show both raw scores next to the fused one.

Degradation: if one index fails, results come from the other (flagged `degraded`); if both
fail, or FAULT_PINECONE is on, the local BM25 fallback answers with the same access filters.
"""

import asyncio
import time

from langsmith import traceable

from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval.dense import dense_search
from app.retrieval.fallback_bm25 import get_bm25_fallback
from app.retrieval.models import RetrievalQuery, RetrievalResult, RetrievedChunk
from app.retrieval.sparse import sparse_search

log = get_logger(__name__)

Hits = list[tuple[str, float, dict]]


def _normalise(hits: Hits) -> dict[str, float]:
    """id -> score / max score. Keeps the best score per id when a chunk appears twice."""
    best: dict[str, float] = {}
    for chunk_id, score, _ in hits:
        best[chunk_id] = max(score, best.get(chunk_id, float("-inf")))
    top = max(best.values(), default=0.0)
    return {k: (v / top if top > 0 else 0.0) for k, v in best.items()}


def fuse(
    dense: Hits, sparse: Hits, alpha: float, top_k: int, max_per_doc: int = 50
) -> list[RetrievedChunk]:
    """Merge both result lists by chunk id, rank by weighted normalised score, and keep at most
    `max_per_doc` chunks per document so several sources can be cited."""
    fields = {cid: f for cid, _, f in [*sparse, *dense]}
    raw_dense = {cid: s for cid, s, _ in dense}
    raw_sparse = {cid: s for cid, s, _ in sparse}
    nd, ns = _normalise(dense), _normalise(sparse)

    chunks = []
    for cid, f in fields.items():
        chunks.append(
            RetrievedChunk(
                id=cid,
                doc_id=f["doc_id"],
                title=f["title"],
                section=f["section"],
                department=f["department"],
                document_type=f["document_type"],
                access_level=f["access_level"],
                created_date=f["created_date"],
                status=f["status"],
                source_path=f["source_path"],
                text=f["chunk_text"],
                dense_score=raw_dense.get(cid),
                sparse_score=raw_sparse.get(cid),
                fused_score=round(alpha * nd.get(cid, 0.0) + (1 - alpha) * ns.get(cid, 0.0), 4),
            )
        )
    chunks.sort(key=lambda c: c.fused_score, reverse=True)
    return cap_per_doc(chunks, top_k, max_per_doc)


def cap_per_doc(chunks: list[RetrievedChunk], top_k: int, max_per_doc: int) -> list[RetrievedChunk]:
    """Take ranked chunks in order, skipping a document once it has `max_per_doc` entries."""
    per_doc: dict[str, int] = {}
    kept: list[RetrievedChunk] = []
    for c in chunks:
        if per_doc.get(c.doc_id, 0) < max_per_doc:
            kept.append(c)
            per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
        if len(kept) == top_k:
            break
    return kept


def _bm25(q: RetrievalQuery, notes: list[str], timings: dict[str, float]) -> RetrievalResult:
    start = time.perf_counter()
    chunks = get_bm25_fallback().search(q)
    timings["bm25_fallback"] = round((time.perf_counter() - start) * 1000, 1)
    return RetrievalResult(chunks=chunks, degraded=True, notes=notes, timings_ms=timings)


def _trace_outputs(result: RetrievalResult) -> dict:
    """Shape the trace as LangSmith retriever documents so chunks render as documents."""
    return {
        "documents": [
            {
                "page_content": c.text,
                "type": "Document",
                "metadata": c.model_dump(exclude={"text"}),
            }
            for c in result.chunks
        ],
        "degraded": result.degraded,
        "notes": result.notes,
    }


@traceable(run_type="retriever", name="hybrid_search", process_outputs=_trace_outputs)
async def hybrid_search(q: RetrievalQuery) -> RetrievalResult:
    s = get_settings()
    alpha = s.hybrid_alpha if q.alpha is None else q.alpha
    notes: list[str] = []
    timings: dict[str, float] = {}

    if s.fault_pinecone:
        notes.append("Pinecone disabled by FAULT_PINECONE; using local BM25 keyword search.")
        return _bm25(q, notes, timings)

    async def timed(name: str, coro):
        start = time.perf_counter()
        try:
            return await coro
        finally:
            timings[name] = round((time.perf_counter() - start) * 1000, 1)

    dense, sparse = await asyncio.gather(
        timed("dense", dense_search(q)),
        timed("sparse", sparse_search(q)),
        return_exceptions=True,
    )
    dense_failed = isinstance(dense, BaseException)
    sparse_failed = isinstance(sparse, BaseException)

    if dense_failed and sparse_failed:
        log.error("retrieval_both_failed", dense=repr(dense), sparse=repr(sparse))
        notes.append("Vector search unavailable; using local BM25 keyword search.")
        return _bm25(q, notes, timings)
    if dense_failed:
        log.warning("retrieval_dense_failed", error=repr(dense))
        notes.append("Semantic search unavailable; results use keyword search only.")
        dense, alpha = [], 0.0
    if sparse_failed:
        log.warning("retrieval_sparse_failed", error=repr(sparse))
        notes.append("Keyword search unavailable; results use semantic search only.")
        sparse, alpha = [], 1.0

    chunks = fuse(dense, sparse, alpha, q.top_k, q.max_per_doc)
    log.info(
        "hybrid_search",
        hits=len(chunks),
        dense_candidates=len(dense),
        sparse_candidates=len(sparse),
        alpha=alpha,
        clearance=q.clearance.name,
        **{f"{k}_ms": v for k, v in timings.items()},
    )
    return RetrievalResult(chunks=chunks, degraded=bool(notes), notes=notes, timings_ms=timings)
