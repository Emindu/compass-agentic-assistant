"""Retrieval contracts shared by the dense, sparse, hybrid and fallback searchers."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from ingestion.models import AccessLevel, Department, DocumentType

ALL_DEPARTMENTS: tuple[Department, ...] = (
    "payments",
    "platform",
    "security",
    "retail",
    "corporate",
)

# Fields returned from Pinecone for every hit (chunk_text included, so no second lookup).
RETURN_FIELDS = [
    "chunk_text",
    "doc_id",
    "title",
    "section",
    "department",
    "document_type",
    "access_level",
    "created_date",
    "service",
    "status",
    "source_path",
]


class RetrievalQuery(BaseModel):
    """What to search for and, crucially, what the caller is allowed to see.

    `clearance` must come from the authenticated user's role (JWT), never from the prompt or the
    LLM: it becomes a hard metadata filter, so a manipulated model cannot widen access.
    """

    text: str = Field(min_length=1, max_length=2000)
    clearance: AccessLevel
    departments: list[Department] = Field(default_factory=lambda: list(ALL_DEPARTMENTS))
    document_types: list[DocumentType] | None = None
    date_from: date | None = None
    date_to: date | None = None
    exclude_statuses: list[Literal["draft", "superseded"]] = Field(default_factory=list)
    top_k: int = Field(default=8, ge=1, le=50)
    # Without a cap, one long document's sections can fill every slot and hide other sources.
    max_per_doc: int = Field(default=2, ge=1, le=20)
    alpha: float | None = Field(default=None, ge=0.0, le=1.0, description="None = settings")


class RetrievedChunk(BaseModel):
    """One attributed result. Both raw scores are kept so the UI can show how fusion ranked it."""

    id: str
    doc_id: str
    title: str
    section: str
    department: str
    document_type: str
    access_level: str
    created_date: str
    status: str
    source_path: str
    text: str
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float = 0.0
    source: Literal["pinecone", "bm25_fallback"] = "pinecone"

    @property
    def citation(self) -> str:
        return f"{self.doc_id} › {self.section}"


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    degraded: bool = False
    notes: list[str] = Field(default_factory=list, description="Why results may be partial")
    timings_ms: dict[str, float] = Field(default_factory=dict)


def build_pinecone_filter(q: RetrievalQuery) -> dict:
    """Metadata filter applied inside Pinecone. Access is always the first clause."""
    clauses: list[dict] = [{"access_rank": {"$lte": int(q.clearance)}}]
    if q.document_types:
        clauses.append({"document_type": {"$in": list(q.document_types)}})
    if q.date_from:
        clauses.append({"created_ts": {"$gte": int(q.date_from.strftime("%Y%m%d"))}})
    if q.date_to:
        clauses.append({"created_ts": {"$lte": int(q.date_to.strftime("%Y%m%d"))}})
    if q.exclude_statuses:
        clauses.append({"status": {"$nin": list(q.exclude_statuses)}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
