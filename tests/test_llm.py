"""LLM factory policy tests with scripted fake models (no network)."""

import asyncio

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

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
