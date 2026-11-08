"""Cross-encoder reranking of fused candidates, via Pinecone's hosted reranker.

Why rerank after fusion: fusion blends two independent similarity scores, so a keyword-heavy
chunk can outrank the chunk that actually answers the question. A cross-encoder reads the
query and the chunk text together, which is exactly the judgement weighted fusion cannot make.

Why only the top `rerank_candidates`: the hosted model is billed and rate-limited per request
(Starter plan), and it adds one blocking hop of latency, so the cheap fused ranking pre-filters
and the expensive model only judges a small shortlist.

Why the caller falls back to fused order: reranking is an optimisation, not a dependency — the
fused list is already a good answer, so an unavailable reranker degrades quality slightly
instead of failing the request (same degradation style as the dense/sparse searchers).
"""

import asyncio

from langsmith import traceable
from pinecone import PineconeAsyncio

from app.core.config import get_settings
from app.core.errors import VectorStoreError
from app.retrieval.models import RetrievedChunk

# One lazy singleton, like get_store(): the async client holds an HTTP connection pool, and
# building it per query would add a TLS handshake to every search.
_client: PineconeAsyncio | None = None


def _get_client() -> PineconeAsyncio:
    global _client
    if _client is None:
        key = get_settings().pinecone_api_key
        if key is None:
            raise VectorStoreError("PINECONE_API_KEY is not configured")
        _client = PineconeAsyncio(api_key=key.get_secret_value())
    return _client


@traceable(run_type="tool", name="rerank")
async def rerank(query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    """Return `chunks` re-ordered by cross-encoder relevance to `query`, with `rerank_score` set.

    Raises VectorStoreError on any failure (missing key, timeout, SDK error); the caller owns
    the fallback policy, so this module never silently degrades results.
    """
    s = get_settings()
    client = _get_client()
    try:
        async with asyncio.timeout(s.rerank_timeout_s):
            result = await client.inference.rerank(
                model=s.rerank_model,
                query=query,
                documents=[{"id": c.id, "text": c.text} for c in chunks],
                rank_fields=["text"],
                top_n=top_n,
                return_documents=False,
                parameters={"truncate": "END"},
            )
    except TimeoutError as exc:
        raise VectorStoreError(f"rerank timed out after {s.rerank_timeout_s}s") from exc
    except Exception as exc:  # The SDK raises several types; callers only need one
        raise VectorStoreError(f"rerank failed: {exc!r}") from exc

    # `.data` is sorted by descending score; `.index` points back into the list we passed.
    ranked: list[RetrievedChunk] = []
    for doc in result.data:
        chunk = chunks[doc.index]
        ranked.append(chunk.model_copy(update={"rerank_score": round(doc.score, 4)}))
    return ranked


async def aclose_reranker() -> None:
    """Shut down the shared client and reset the singleton (for shutdown hooks; not wired yet)."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
