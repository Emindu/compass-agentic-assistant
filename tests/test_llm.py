"""LLM factory policy tests with scripted fake models (no network)."""

import asyncio
from typing import TypedDict

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.schemas import Run
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph

from app.core import llm
from app.core.config import get_settings
from app.core.errors import LLMError


def http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


class ScriptedModel:
    """Each call consumes the next outcome: an exception to raise, a delay, or a reply text."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def __call__(self, _messages):
        self.calls += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, float):
            await asyncio.sleep(outcome)
            return AIMessage("too late")
        return AIMessage(outcome)


@pytest.fixture
def models(monkeypatch):
    """Route model names to scripted fakes and use fast, deterministic settings."""
    monkeypatch.setenv("LLM_MODEL", "main-model")
    monkeypatch.setenv("FAST_LLM_MODEL", "fast-model")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("LLM_TIMEOUT_S", "5")
    monkeypatch.setenv("FAULT_LLM", "false")
    monkeypatch.setattr(llm, "RETRY_INITIAL_S", 0.0)
    monkeypatch.setattr(llm, "RETRY_MAX_S", 0.0)
    get_settings.cache_clear()

    registry: dict[str, ScriptedModel] = {}
    monkeypatch.setattr(llm, "_base_model", lambda name, _t: RunnableLambda(registry[name]))
    yield registry
    get_settings.cache_clear()


async def test_retries_on_429_then_succeeds(models):
    models["fast-model"] = ScriptedModel(http_error(429), http_error(429), "ok")
    reply = await llm.ainvoke_llm("fast", "hi")
    assert reply.content == "ok"
    assert models["fast-model"].calls == 3


async def test_no_retry_on_400(models):
    models["fast-model"] = ScriptedModel(http_error(400))
    with pytest.raises(httpx.HTTPStatusError):
        await llm.ainvoke_llm("fast", "hi")
    assert models["fast-model"].calls == 1


async def test_timeout_is_retried_then_raises_llm_error(models, monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_S", "0.05")
    get_settings.cache_clear()
    models["fast-model"] = ScriptedModel(1.0)
    with pytest.raises(LLMError):
        await llm.ainvoke_llm("fast", "hi")
    assert models["fast-model"].calls == 3  # 1 attempt + LLM_MAX_RETRIES


async def test_fault_flag_falls_back_to_fast_model(models, monkeypatch):
    monkeypatch.setenv("FAULT_LLM", "true")
    get_settings.cache_clear()
    models["main-model"] = ScriptedModel("from main")
    models["fast-model"] = ScriptedModel("from fast")
    reply = await llm.ainvoke_llm("main", "hi")
    assert reply.content == "from fast"
    assert models["main-model"].calls == 0  # the fault gate fires before the model


async def test_provider_5xx_falls_back_without_retry(models):
    models["main-model"] = ScriptedModel(http_error(503))
    models["fast-model"] = ScriptedModel("from fast")
    reply = await llm.ainvoke_llm("main", "hi")
    assert reply.content == "from fast"
    assert models["main-model"].calls == 1


def test_code_role_chain_dedupes_and_ends_on_main(models, monkeypatch):
    monkeypatch.setenv("CODE_LLM_MODEL", "codestral-latest")
    monkeypatch.setenv("CODE_LLM_FALLBACK", "codestral-latest")
    get_settings.cache_clear()
    assert llm._models_for("code") == ["codestral-latest", "main-model"]


async def test_stream_falls_back_before_first_token(models, monkeypatch):
    monkeypatch.setenv("FAULT_LLM", "true")
    get_settings.cache_clear()
    models["main-model"] = ScriptedModel("from main")
    models["fast-model"] = ScriptedModel("from fast")
    chunks = [c async for c in llm.astream_llm("main", "hi")]
    assert "".join(c.content for c in chunks) == "from fast"


class RecordingTracer(LangChainTracer):
    """LangSmith tracer that records finished runs instead of sending them. `raise_error`
    turns tracer bookkeeping failures ("No indexed run ID") into test failures."""

    raise_error = True
    finished: list[Run] = []

    def _persist_run_single(self, run: Run) -> None:
        pass

    @staticmethod
    def _update_run_single(run: Run) -> None:
        RecordingTracer.finished.append(run)


async def test_llm_runs_inside_graph_nest_under_the_graph_trace(models):
    """Inside a node, the call must extend the node's config, not replace it. Replacing the
    metadata made LangChain fork the tracer, so every LLM run was ended twice and errored."""
    models["main-model"] = ScriptedModel("hello")
    models["fast-model"] = ScriptedModel("unused")

    async def node(_state: dict) -> dict:
        reply = await llm.ainvoke_llm("main", "hi")
        return {"answer": reply.content}

    class State(TypedDict, total=False):
        answer: str

    graph = StateGraph(State)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    RecordingTracer.finished = []
    tracer = RecordingTracer(client=object())  # type: ignore[arg-type]
    config = {
        "configurable": {"thread_id": "user:t1"},  # scoped id, as the chat route sets it
        "metadata": {"thread_id": "t1", "user_id": "user"},
        "callbacks": [tracer],
        "run_name": "compass_chat",
    }
    result = await graph.compile(checkpointer=InMemorySaver()).ainvoke({}, config)

    assert result["answer"] == "hello"
    runs = {str(r.id): r for r in RecordingTracer.finished}
    assert len(runs) == len(RecordingTracer.finished), "a run was finished more than once"
    root = next(r for r in runs.values() if r.parent_run_id is None)
    assert root.name == "compass_chat"
    llm_run = next(r for r in runs.values() if r.name == "llm:main")
    assert llm_run.trace_id == root.trace_id
    assert llm_run.parent_run_id is not None
    metadata = llm_run.extra["metadata"]
    assert metadata["user_id"] == "user" and metadata["llm_role"] == "main"
