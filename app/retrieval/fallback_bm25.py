"""Degraded-mode retrieval: local BM25 keyword search used when Pinecone is unavailable.

Why a fallback at all: the assistant must degrade instead of failing, and incidents where
vector search is down are exactly when someone needs the runbooks. BM25Okapi gives true
keyword scoring over the same chunks the primary path indexes.

Why the same access filters: degradation must never widen access. The filter set mirrors
`build_pinecone_filter`, so a degraded response is no more permissive than a normal one.

Why in-process: the corpus is small, so one BM25 index fits in memory and loads lazily.
Production scale would move this to OpenSearch/Elasticsearch behind the same interface.
"""

import functools
import re

from rank_bm25 import BM25Okapi

from app.core.logging import get_logger
from app.retrieval.models import RetrievalQuery, RetrievedChunk
from ingestion.chunk import chunk_corpus
from ingestion.models import AccessLevel, Chunk

log = get_logger(__name__)

# Keeps ids like rb-pay-003, x-request-id and card-switch-connector whole instead of
# fragmenting them into parts that also match unrelated text.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_PARTS = re.compile(r"[-_.]")


def tokenize(text: str) -> list[str]:
    """Whole compound tokens plus their parts, so both sides of a hyphen can match.

    "rotate certificate RB-PAY-003" must find `rb-pay-003` exactly, while "certificate" alone
    must still match it; emitting both forms serves keyword search in both directions.
    """
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.lower()):
        token = match.group(0)
        tokens.append(token)
        parts = _PARTS.split(token)
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


class BM25Fallback:
    """In-memory BM25 index with Pinecone-equivalent metadata and RBAC filtering."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._index = BM25Okapi([tokenize(c.text) for c in chunks])

    def _is_allowed(self, chunk: Chunk, q: RetrievalQuery) -> bool:
        """The security control: identical semantics to `build_pinecone_filter`."""
        if AccessLevel[chunk.meta.access_level] > q.clearance:
            return False
        if chunk.meta.department not in q.departments:
            return False
        if q.document_types and chunk.meta.document_type not in q.document_types:
            return False
        if q.date_from and chunk.meta.created_date < q.date_from:
            return False
        if q.date_to and chunk.meta.created_date > q.date_to:
            return False
        if chunk.meta.status in q.exclude_statuses:
            return False
        return True

    def search(self, q: RetrievalQuery) -> list[RetrievedChunk]:
        scores = self._index.get_scores(tokenize(q.text))
        allowed = [
            (score, chunk)
            for score, chunk in zip(scores, self._chunks, strict=True)
            if score > 0 and self._is_allowed(chunk, q)
        ]
        allowed.sort(key=lambda pair: pair[0], reverse=True)
        top_score = allowed[0][0] if allowed else 0.0
        results = [
            RetrievedChunk(
                id=chunk.id,
                doc_id=chunk.meta.doc_id,
                title=chunk.meta.title,
                section=chunk.section,
                department=chunk.meta.department,
                document_type=chunk.meta.document_type,
                access_level=chunk.meta.access_level,
                created_date=chunk.meta.created_date.isoformat(),
                status=chunk.meta.status,
                source_path=chunk.source_path,
                text=chunk.text,
                dense_score=None,
                sparse_score=score,
                fused_score=score / top_score,
                source="bm25_fallback",
            )
            for score, chunk in allowed[: q.top_k]
        ]
        log.info("bm25_fallback_search", hits=len(results), candidates_allowed=len(allowed))
        return results


@functools.lru_cache
def get_bm25_fallback() -> BM25Fallback:
    """Lazy singleton: the corpus is only tokenised when the fallback is first needed.

    Cached because the tokenised index is pure and expensive to rebuild on every degraded call.
    """
    return BM25Fallback(chunk_corpus())
