"""Graph wiring tests with the LLM and Pinecone stubbed: routing, retry, containment, memory."""

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.graph import build_graph
from app.agents.nodes import response, retrieval, supervisor
from app.agents.state import RouteDecision
from app.retrieval.models import RetrievalResult, RetrievedChunk
from app.security.rbac import Role, UserContext

USER = UserContext(user_id="u1", name="Vera", role=Role.viewer, department="retail")
CHUNK = RetrievedChunk(
    id="INC-2026-0214#03",
    doc_id="INC-2026-0214",
    title="Instant transfer outage",
    section="Root cause",
    department="payments",
    document_type="incident",
    access_level="internal",
    created_date="2026-02-14",
    status="final",
    source_path="p",
    text="Retries had no backoff.",
)


def stub(monkeypatch, *, route: RouteDecision, answers: list[str], search_fails=False):
    async def fake_structured(_role, _schema, _messages, **_kw):
        return route

    async def fake_search(_q):
        if search_fails:
            raise RuntimeError("pinecone exploded")
        return RetrievalResult(chunks=[CHUNK])

    async def fake_stream(_role, _messages, **_kw):
        for word in answers.pop(0).split(" "):
            yield AIMessageChunk(content=word + " ")

    monkeypatch.setattr(supervisor, "ainvoke_structured", fake_structured)
    monkeypatch.setattr(retrieval, "hybrid_search", fake_search)
    monkeypatch.setattr(response, "astream_llm", fake_stream)


QA = RouteDecision(
    intent="knowledge_qa", agents=["retrieval"], search_query="q", reasoning="needs docs"
)


async def run(graph, text: str, thread: str = "t1") -> tuple[dict, list[dict]]:
    config = {"configurable": {"thread_id": thread}}
    inputs = {"messages": [HumanMessage(text)], "user": USER}
    events = [e async for e in graph.astream(inputs, config, stream_mode="custom")]
    return (await graph.aget_state(config)).values, events


async def test_cited_answer_flows_through_validator(monkeypatch):
    stub(monkeypatch, route=QA, answers=["No backoff [INC-2026-0214#03]."])
    state, events = await run(build_graph(InMemorySaver()), "why?")
    assert state["validation"].valid
    assert state["messages"][-1].content == "No backoff [INC-2026-0214#03]."
    types = [e["type"] for e in events]
    assert {"route", "retrieval", "token", "validation", "answer", "memory"} <= set(types)
    answer = next(e for e in events if e["type"] == "answer")
    assert answer["data"]["citations"][0]["doc_id"] == "INC-2026-0214"


async def test_fake_citation_triggers_one_retry(monkeypatch):
    stub(monkeypatch, route=QA, answers=["Made up [FAKE-9999#01].", "Real [INC-2026-0214#03]."])
    state, events = await run(build_graph(InMemorySaver()), "why?")
    validations = [e["data"]["valid"] for e in events if e["type"] == "validation"]
    assert validations == [False, True]
    assert "FAKE" not in state["messages"][-1].content


async def test_smalltalk_skips_retrieval(monkeypatch):
    chat = RouteDecision(intent="smalltalk", agents=[], search_query="hi", reasoning="greeting")
    stub(monkeypatch, route=chat, answers=["Hello! I answer questions from bank documents."])
    _, events = await run(build_graph(InMemorySaver()), "hi")
    assert not any(e["type"] == "retrieval" for e in events)


async def test_retrieval_failure_is_contained(monkeypatch):
    stub(
        monkeypatch,
        route=QA,
        answers=["I couldn't find this in the documents you can access."],
        search_fails=True,
    )
    state, events = await run(build_graph(InMemorySaver()), "why?")
    assert any(e["type"] == "error" and e["node"] == "retrieval" for e in events)
    assert state["errors"][0].node == "retrieval"
    assert state["validation"].valid  # honest "couldn't find" is accepted
    assert state["messages"][-1].type == "ai"  # the turn still completed


async def test_memory_persists_across_turns(monkeypatch):
    stub(monkeypatch, route=QA, answers=["First [INC-2026-0214#03].", "Second [INC-2026-0214#03]."])
    graph = build_graph(InMemorySaver())
    await run(graph, "first question")
    state, events = await run(graph, "follow up")
    assert [m.type for m in state["messages"]] == ["human", "ai", "human", "ai"]
    loaded = next(e for e in events if e["type"] == "memory" and e["node"] == "load_memory")
    assert loaded["data"]["previous_turns"] == 1
    assert state["errors"] == []  # per-turn fields reset
