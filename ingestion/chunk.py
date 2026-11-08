"""Load markdown corpus files and split them into section-aware chunks.

Why split on `##` sections: the documents are structured (Summary, Root cause, Steps...), so a
section is the natural unit of evidence and gives citations a meaningful anchor
("INC-2026-0214 › Root cause"). Long sections are packed by paragraph up to `max_chars`.

Why a contextual header on every chunk: a chunk like "Retries disabled via feature flag" means
little alone. Prefixing doc id, title and section lets both dense and keyword search match it to
the right document, and lets the model cite it.
"""

import re
from pathlib import Path

import yaml

from ingestion.models import Chunk, Document, DocumentMeta

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
MAX_CHARS = 1200

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)
# Authoring notes for humans (mock-data banner, test-fixture labels) are not indexed.
_AUTHOR_NOTE = re.compile(r"^>\s*(MOCK DATA|SECURITY TEST FIXTURE).*$", re.MULTILINE)


def load_document(path: Path, root: Path = CORPUS_DIR) -> Document:
    match = _FRONT_MATTER.match(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"{path}: missing YAML front matter")
    meta = DocumentMeta.model_validate(yaml.safe_load(match.group(1)))
    body = _AUTHOR_NOTE.sub("", match.group(2)).strip()
    return Document(meta=meta, body=body, source_path=str(path.relative_to(root.parent.parent)))


def load_corpus(root: Path = CORPUS_DIR) -> list[Document]:
    docs = [load_document(p, root) for p in sorted(root.rglob("*.md"))]
    ids = [d.meta.doc_id for d in docs]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate doc_id values: {sorted(duplicates)}")
    return docs


def _sections(body: str) -> list[tuple[str, str]]:
    """(section heading, text) pairs. Text before the first `##` becomes 'Overview'."""
    sections: list[tuple[str, str]] = []
    heading, lines = "Overview", []
    for line in body.splitlines():
        if line.startswith("# "):
            continue  # the H1 duplicates the title, which is already in the header
        if line.startswith("## "):
            if "".join(lines).strip():
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = line[3:].strip(), []
        else:
            lines.append(line)
    if "".join(lines).strip():
        sections.append((heading, "\n".join(lines).strip()))
    return sections


def _pack(text: str, max_chars: int) -> list[str]:
    """Greedy paragraph packing; an oversized paragraph is kept whole, not cut mid-sentence."""
    parts, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        if current and len(current) + len(para) + 2 > max_chars:
            parts.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        parts.append(current)
    return parts


def chunk_document(doc: Document, max_chars: int = MAX_CHARS) -> list[Chunk]:
    m = doc.meta
    status = "" if m.status == "final" else f" [{m.status.upper()}]"
    chunks: list[Chunk] = []
    for section, text in _sections(doc.body):
        for piece in _pack(text, max_chars):
            header = (
                f"[{m.doc_id}] {m.title}{status}\n"
                f"Type: {m.document_type} | Department: {m.department} | "
                f"Service: {m.service} | Date: {m.created_date.isoformat()}\n"
                f"Section: {section}"
            )
            index = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{m.doc_id}#{index:02d}",
                    text=f"{header}\n\n{piece}",
                    section=section,
                    chunk_index=index,
                    meta=m,
                    source_path=doc.source_path,
                )
            )
    return chunks


def chunk_corpus(root: Path = CORPUS_DIR) -> list[Chunk]:
    return [c for doc in load_corpus(root) for c in chunk_document(doc)]
