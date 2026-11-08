import json
from pathlib import Path

from ingestion.chunk import chunk_corpus, chunk_document, load_corpus
from ingestion.models import AccessLevel

GROUND_TRUTH = json.loads((Path(__file__).parent.parent / "evals/ground_truth.json").read_text())


def test_corpus_loads_with_valid_metadata():
    docs = load_corpus()
    assert 20 <= len(docs) <= 30
    assert {d.meta.department for d in docs} == {
        "payments",
        "platform",
        "security",
        "retail",
        "corporate",
    }
    # every access level is represented, so RBAC filtering can be demonstrated
    assert {d.meta.access_level for d in docs} == {level.name for level in AccessLevel}


def test_ground_truth_incidents_exist_in_corpus():
    ids = {d.meta.doc_id for d in load_corpus()}
    rlm = GROUND_TRUTH["rlm_payment_outages"]
    assert set(rlm["in_scope_incidents"]) <= ids
    assert set(rlm["must_exclude"]) <= ids


def test_chunks_have_contextual_header_and_stable_ids():
    doc = next(d for d in load_corpus() if d.meta.doc_id == "INC-2026-0214")
    chunks = chunk_document(doc)
    root_cause = next(c for c in chunks if c.section == "Root cause")
    assert root_cause.text.startswith("[INC-2026-0214]")
    assert "Section: Root cause" in root_cause.text
    assert [c.id for c in chunks] == [c.id for c in chunk_document(doc)]


def test_author_notes_are_not_indexed():
    for chunk in chunk_corpus():
        assert "MOCK DATA" not in chunk.text
        assert "TEST FIXTURE" not in chunk.text


def test_record_metadata_is_pinecone_compatible():
    for chunk in chunk_corpus():
        record = chunk.to_record()
        assert record["_id"] == chunk.id
        assert all(v is not None for v in record.values())
        assert all(isinstance(v, str | int) for v in record.values())
        assert 19000101 < record["created_ts"] < 21000101
