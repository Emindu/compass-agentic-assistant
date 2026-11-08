"""Primitive boundary tests: clearance enforced per primitive, no leaks, budgets honored.

No network: `hybrid_search` and `ainvoke_llm` are monkeypatched, so these exercise only the
dispatch, validation, RBAC and evidence-registry logic that this boundary owns.
"""

import json
from pathlib import Path

import pytest

from app.retrieval.models import RetrievalResult, RetrievedChunk
from app.rlm import primitives
from app.rlm.primitives import PrimitiveSession
from app.security.rbac import Role, UserContext
from ingestion.models import AccessLevel

_GROUND_TRUTH = Path(__file__).parents[1] / "evals" / "ground_truth.json"


def _user(role: Role) -> UserContext:
    return UserContext(user_id="u1", name="Test User", role=role, department="payments")


def _fake_chunk(
    chunk_id: str = "INC-2026-0214#02", doc_id: str = "INC-2026-0214"
) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        doc_id=doc_id,
        title="Card payments retry storm",
        section="Root cause",
        department="payments",
        document_type="incident",
        access_level="internal",
        created_date="2026-02-14",
        status="final",
        source_path="data/corpus/payments/INC-2026-0214.md",
        text="[INC-2026-0214] Root cause: immediate retries without backoff.",
        dense_score=0.5,
        sparse_score=3.2,
        fused_score=0.87,
    )


class _FakeReply:
    text = "fake sub-analyst answer"


async def _fake_ainvoke_llm(role, messages, **kwargs):  # noqa: ANN001, ANN202
    assert role == "fast"
    assert isinstance(messages, list)
    return _FakeReply()


async def test_list_documents_matches_in_scope_incidents_for_viewer():
    truth = json.loads(_GROUND_TRUTH.read_text())
    expected = set(truth["rlm_payment_outages"]["in_scope_incidents"])
    expected.add("INC-2026-0712-DRAFT")  # a draft is still a document the viewer may list

    session = PrimitiveSession(_user(Role.viewer))
    docs = await session.handle(
        "list_documents",
        {"document_type": "incident", "department": "payments", "date_from": "2025-11-08"},
    )

    ids = [d["doc_id"] for d in docs]
    assert set(ids) == expected
    assert "INC-2024-0815" not in ids  # older than the one-year window
    dates = [d["created_date"] for d in docs]
    assert dates == sorted(dates)  # sorted ascending
    assert all("body" not in d and "text" not in d for d in docs)  # metadata only


async def test_list_documents_enforces_clearance_ceiling():
    viewer_ids = {
        d["doc_id"] for d in await PrimitiveSession(_user(Role.viewer)).handle("list_documents", {})
    }
    analyst_ids = {
        d["doc_id"]
        for d in await PrimitiveSession(_user(Role.analyst)).handle("list_documents", {})
    }
    admin_ids = {
        d["doc_id"] for d in await PrimitiveSession(_user(Role.admin)).handle("list_documents", {})
    }

    assert "MTG-PAY-2026-10" not in viewer_ids  # confidential
    assert "POL-SEC-004" not in viewer_ids  # restricted
    assert "MTG-PAY-2026-10" in analyst_ids
    assert "POL-SEC-004" not in analyst_ids  # analyst still cannot see restricted
    assert "POL-SEC-004" in admin_ids


async def test_get_sections_hides_forbidden_document_without_leaking_existence():
    viewer = await PrimitiveSession(_user(Role.viewer)).handle(
        "get_sections", {"doc_id": "MTG-PAY-2026-10"}
    )
    analyst = await PrimitiveSession(_user(Role.analyst)).handle(
        "get_sections", {"doc_id": "MTG-PAY-2026-10"}
    )
    unknown = await PrimitiveSession(_user(Role.analyst)).handle(
        "get_sections", {"doc_id": "NOPE-1"}
    )

    assert viewer == []  # identical to the unknown-doc case: existence is not leaked
    assert unknown == []
    assert analyst  # analyst is cleared for confidential
    assert all(c["doc_id"] == "MTG-PAY-2026-10" for c in analyst)


async def test_get_sections_filters_by_section_name_and_registers_evidence():
    session = PrimitiveSession(_user(Role.viewer))
    chunks = await session.handle(
        "get_sections", {"doc_id": "INC-2026-0214", "sections": ["root cause"]}
    )

    assert chunks
    assert all(c["section"] == "Root cause" for c in chunks)  # case-insensitive match
    assert session.evidence
    for c in chunks:
        retrieved = session.evidence[c["chunk_id"]]
        assert retrieved.doc_id == "INC-2026-0214"
        assert retrieved.section == "Root cause"
        assert retrieved.text == c["text"]


async def test_search_passes_clearance_to_hybrid_and_registers_evidence(monkeypatch):
    captured: dict = {}

    async def fake_hybrid_search(q):
        captured["query"] = q
        return RetrievalResult(chunks=[_fake_chunk()])

    monkeypatch.setattr(primitives, "hybrid_search", fake_hybrid_search)

    viewer = PrimitiveSession(_user(Role.viewer))
    rows = await viewer.handle("search", {"query": "retry storm without backoff", "top_k": 5})

    assert captured["query"].clearance == AccessLevel.internal
    assert captured["query"].text == "retry storm without backoff"
    assert len(rows) == 1
    assert rows[0]["chunk_id"] == "INC-2026-0214#02"
    assert rows[0]["score"] == 0.87
    assert viewer.evidence["INC-2026-0214#02"] == _fake_chunk()  # stored as returned by search


async def test_llm_query_budget_exhausts(monkeypatch):
    monkeypatch.setattr(primitives, "ainvoke_llm", _fake_ainvoke_llm)
    session = PrimitiveSession(_user(Role.viewer), max_llm_calls=2)

    first = await session.handle("llm_query", {"prompt": "What failed?"})
    second = await session.handle("llm_query", {"prompt": "Why did it recur?"})
    assert first == second == "fake sub-analyst answer"
    assert session.llm_calls == 2

    with pytest.raises(ValueError, match="budget exhausted"):
        await session.handle("llm_query", {"prompt": "One question too many"})


async def test_handle_rejects_unknown_primitive_and_maps_positional_args():
    session = PrimitiveSession(_user(Role.viewer))
    with pytest.raises(ValueError, match="unknown primitive"):
        await session.handle("rm_rf", {})

    chunks = await session.handle("get_sections", {"_args": ["INC-2026-0214"]})
    assert chunks  # doc_id was mapped from the first positional argument
    assert all(c["doc_id"] == "INC-2026-0214" for c in chunks)
