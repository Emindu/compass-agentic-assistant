"""Sparse (keyword) search: exact terms that embeddings blur, such as IDs and error strings.

Uses Pinecone's integrated sparse model (pinecone-sparse-english-v0, dotproduct), a learned
lexical model in the BM25 family: it weights terms such as `HikariPool`, `INC-2026-0214` or
`handshake_failure` that a dense embedding treats as near-synonyms of generic words.
True BM25 (rank_bm25) is used by the local fallback; see ADR in docs/DESIGN.md.
"""

from langsmith import traceable

from app.core.config import get_settings
from app.retrieval.dense import CANDIDATE_MULTIPLIER
from app.retrieval.models import RetrievalQuery, build_pinecone_filter
from app.retrieval.pinecone_store import get_store


@traceable(run_type="retriever", name="sparse_search")
async def sparse_search(q: RetrievalQuery) -> list[tuple[str, float, dict]]:
    s = get_settings()
    return await get_store().search(
        s.pinecone_sparse_index,
        q.text,
        list(q.departments),
        q.top_k * CANDIDATE_MULTIPLIER,
        build_pinecone_filter(q),
    )
