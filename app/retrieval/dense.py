"""Dense (semantic) search: finds chunks that mean the same thing in different words.

Pinecone embeds the query with the index's integrated model (llama-text-embed-v2, cosine), so
"why did card payments fail because of TLS" can match "handshake_failure ... certificate".
"""

from langsmith import traceable

from app.core.config import get_settings
from app.retrieval.models import RetrievalQuery, build_pinecone_filter
from app.retrieval.pinecone_store import get_store

CANDIDATE_MULTIPLIER = 3  # over-fetch so fusion has candidates from both sides to re-rank


@traceable(run_type="retriever", name="dense_search")
async def dense_search(q: RetrievalQuery) -> list[tuple[str, float, dict]]:
    s = get_settings()
    return await get_store().search(
        s.pinecone_dense_index,
        q.text,
        list(q.departments),
        q.top_k * CANDIDATE_MULTIPLIER,
        build_pinecone_filter(q),
    )
