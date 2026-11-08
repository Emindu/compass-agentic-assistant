"""Typed document and chunk models shared by ingestion, retrieval and the BM25 fallback."""

from datetime import date
from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field

Department = Literal["payments", "platform", "security", "retail", "corporate"]
DocumentType = Literal["incident", "runbook", "architecture", "policy", "spec", "meeting_notes"]


class AccessLevel(IntEnum):
    """Ordered so a role's clearance is one number: allowed iff doc.access_rank <= clearance.

    Stored as an integer in Pinecone metadata so retrieval can filter with `$lte` instead of
    listing every allowed level.
    """

    public = 0
    internal = 1
    confidential = 2
    restricted = 3


class DocumentMeta(BaseModel):
    """Front-matter of a corpus file. Validated so a typo can't silently skip RBAC filtering."""

    doc_id: str
    title: str
    department: Department
    document_type: DocumentType
    access_level: Literal["public", "internal", "confidential", "restricted"]
    created_date: date
    service: str
    status: Literal["final", "draft", "superseded"] = "final"
    severity: str | None = None
    version: str | None = None


class Document(BaseModel):
    meta: DocumentMeta
    body: str
    source_path: str


class Chunk(BaseModel):
    """One retrievable unit. `id` is deterministic, so re-ingesting overwrites, never duplicates."""

    id: str
    text: str = Field(description="Contextual header + section text; this is what gets embedded")
    section: str
    chunk_index: int
    meta: DocumentMeta
    source_path: str

    def to_record(self) -> dict:
        """Flat Pinecone record. Integrated inference embeds `chunk_text`; the rest is metadata."""
        m = self.meta
        record = {
            "_id": self.id,
            "chunk_text": self.text,
            "doc_id": m.doc_id,
            "title": m.title,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "department": m.department,
            "document_type": m.document_type,
            "access_level": m.access_level,
            "access_rank": int(AccessLevel[m.access_level]),
            "created_date": m.created_date.isoformat(),
            # YYYYMMDD integer so date ranges can use $gte/$lte filters.
            "created_ts": int(m.created_date.strftime("%Y%m%d")),
            "service": m.service,
            "status": m.status,
            "source_path": self.source_path,
        }
        if m.severity:
            record["severity"] = m.severity
        if m.version:
            record["version"] = m.version
        return record
