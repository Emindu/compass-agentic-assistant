"""RLM subgraph end to end with the real sandbox and primitives; only the LLMs are stubbed."""

import json
import re
from pathlib import Path

from app.agents.nodes import supervisor
from app.agents.state import RouteDecision
from app.rlm import graph as rlm
from app.security.rbac import Role, UserContext

GROUND_TRUTH = json.loads((Path(__file__).parent.parent / "evals/ground_truth.json").read_text())
EXPECTED = set(GROUND_TRUTH["rlm_payment_outages"]["in_scope_incidents"])

PLAN = """
docs = list_documents(document_type="incident", department="payments", date_from="2025-11-08")
docs = [d for d in docs if d["status"] == "final"]
ids = [d["doc_id"] for d in docs]
chunks = {}
for d in ids:
    chunks[d] = [s["chunk_id"] for s in get_sections(d, sections=["Summary", "Root cause"])]
batches = [[c for d in ids[i:i + 3] for c in chunks[d]] for i in range(0, len(ids), 3)]
result = {"documents": ids, "batches": batches, "task": "root cause per incident",
          "data": {"incident_count": len(ids)}}
"""

ANALYST = UserContext(user_id="a", name="Anil", role=Role.analyst, department="payments")


def stub_llms(monkeypatch, plan: str = PLAN):
    async def fake_plan(_question, _today, feedback=None):
        return plan

    async def fake_structured(_role, schema, prompt, **_kw):
        if schema is rlm.BatchAnalysis:
            ids = re.findall(r'<chunk id="([^"]+)">', prompt)
            docs = dict.fromkeys(i.split("#")[0] for i in ids)
            return rlm.BatchAnalysis(
                findings=[
                    rlm.Finding(
                        doc_id=d,
                        summary="s",
                        category="c",
                        evidence=[i for i in ids if i.startswith(d)] + ["FAKE-1#00"],
                    )
                    for d in docs
                ]
            )
        return rlm.Synthesis(summary="themes [INC-2026-0214#03]", recurring_themes=["retries"])

    monkeypatch.setattr(rlm, "write_plan", fake_plan)
    monkeypatch.setattr(rlm, "ainvoke_structured", fake_structured)


async def test_rlm_selects_in_scope_incidents_and_fans_out(monkeypatch):
    stub_llms(monkeypatch)
    out = await rlm.research_graph.ainvoke({"question": "q", "user": ANALYST, "depth": 0})
    assert set(out["documents"]) == EXPECTED  # draft, 2024 and non-payment incidents excluded
    assert len(out["batches"]) == 3  # 8 docs in batches of 3
    assert {f.doc_id for f in out["findings"]} == EXPECTED  # every batch analysed
    # evidence ids the model invented are dropped; real ones are registered evidence
    for f in out["findings"]:
        assert "FAKE-1#00" not in f.evidence
        assert all(e in out["evidence"] for e in f.evidence)
    assert out["data"] == {"incident_count": 8}


async def test_bad_plan_is_retried_then_contained(monkeypatch):
    stub_llms(monkeypatch, plan="import os\nresult = os.listdir('/')")
    out = await rlm.research_graph.ainvoke({"question": "q", "user": ANALYST, "depth": 0})
    assert out["plan_attempts"] == 2  # retried once with the error as feedback
    assert out["plan_error"]
    assert out["synthesis"] is not None  # still produces an (empty-evidence) synthesis


def test_viewer_is_routed_away_from_python_analysis():
    route = RouteDecision(
        intent="deep_research", agents=["research"], search_query="q", reasoning="r"
    )
    viewer = UserContext(user_id="v", name="Vera", role=Role.viewer, department="retail")
    assert supervisor.route_after_supervisor({"route": route, "user": viewer}) == "retrieval"
    assert supervisor.route_after_supervisor({"route": route, "user": ANALYST}) == "research"


def test_finding_without_evidence_is_kept():
    """Models sometimes omit `evidence`; that must not reject the batch it belongs to."""
    batch = rlm.BatchAnalysis.model_validate(
        {"findings": [{"doc_id": "INC-2026-0125", "summary": "s", "category": "c"}]}
    )
    assert batch.findings[0].evidence == []
