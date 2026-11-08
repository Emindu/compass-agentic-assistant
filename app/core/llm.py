"""LLM factory: three model roles, fallbacks, one shared concurrency limit, retries.

Why three roles instead of one model everywhere:
- ``main`` (quality where it matters): the supervisor's routing, the RLM reduce step and the
  final answer are few calls whose quality the user sees directly, so they get the strongest model.
- ``fast`` (speed and cost for high-volume steps): the input guard runs on every request and the
  RLM map step fans out one call per batch. A small model keeps that latency and cost flat.
- ``code`` (a code-specialised model for the planner): the RLM planner writes a Python search
  plan that a sandbox executes; a code model writes valid Python more reliably.

Model names come from env (see ``Settings``), and models are built with ``init_chat_model``, so
switching provider is an env change, not a code change.

Why one module-level semaphore: Mistral rate limits apply per workspace, across all keys, so
every LLM call in the process (RLM fan-out included) shares one concurrency budget. Retries use
tenacity on transient errors only (429, 5xx, timeouts). A 4xx validation error fails fast,
because sending the same bad request again cannot succeed.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Sequence
from functools import lru_cache
from typing import Any, Literal

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, SystemMessage, convert_to_messages
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, ensure_config
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.config import get_settings
from app.core.errors import LLMError
from app.core.logging import get_logger

log = get_logger(__name__)

Role = Literal["main", "fast", "code"]

_TEMPERATURE: dict[Role, float] = {"main": 0.2, "fast": 0.0, "code": 0.0}

# Backoff bounds (seconds). Module-level so tests can shrink them.
RETRY_INITIAL_S = 1.0
RETRY_MAX_S = 20.0

# Shared by every LLM call in the process, streaming included (one stream = one slot).
_semaphore = asyncio.Semaphore(get_settings().llm_max_concurrency)


class SimulatedProviderError(Exception):
    """Raised by the main model when FAULT_LLM=true, to demo the fallback path."""


# --------------------------------------------------------------------------- error policy


def is_transient(exc: BaseException) -> bool:
    """True for errors worth retrying or falling back on: 429, 5xx, timeouts, simulated faults."""
    if isinstance(exc, SimulatedProviderError | TimeoutError | httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


def _is_rejected_request(exc: BaseException) -> bool:
    """A non-retryable 4xx: the provider understood the request and refused it."""
    return isinstance(exc, httpx.HTTPStatusError) and not is_transient(exc)


# --------------------------------------------------------------------------- model building


def _models_for(role: Role) -> list[str]:
    """Primary model first, then fallbacks in order, without consecutive duplicates."""
    s = get_settings()
    chain = {
        "main": [s.llm_model, s.fast_llm_model],
        "fast": [s.fast_llm_model],
        "code": [s.code_llm_model, s.code_llm_fallback, s.llm_model],
    }[role]
    deduped: list[str] = []
    for name in chain:
        if name not in deduped:
            deduped.append(name)
    return deduped


@lru_cache
def _base_model(model: str, temperature: float) -> BaseChatModel:
    """One client per (model, temperature). Provider retries are off: tenacity owns retries."""
    s = get_settings()
    kwargs: dict[str, Any] = {"temperature": temperature, "timeout": int(s.llm_timeout_s)}
    if s.llm_provider == "mistralai":
        kwargs["max_retries"] = 1  # langchain-mistralai: 1 attempt, no internal retry
        if s.mistral_api_key is not None:
            kwargs["api_key"] = s.mistral_api_key.get_secret_value()
    return init_chat_model(model, model_provider=s.llm_provider, **kwargs)


def _fault_gate(value: Any) -> Any:
    if get_settings().fault_llm:
        raise SimulatedProviderError("FAULT_LLM=true: simulated provider outage on main model")
    return value


def _fallback_marker(role: Role, primary: str, fallback: str) -> Runnable:
    """Pass-through step that logs when a fallback model is actually used."""

    def mark(value: Any) -> Any:
        log.warning("llm_fallback", role=role, primary_model=primary, fallback_model=fallback)
        return value

    return RunnableLambda(mark, name="llm_fallback")


def _build(role: Role, adapt: Callable[[BaseChatModel], Runnable]) -> Runnable:
    """Build primary + fallbacks for a role. `adapt` binds tools / structured output per model,
    because those must be applied to each model before `.with_fallbacks()` wraps them."""
    models = _models_for(role)
    temperature = _TEMPERATURE[role]
    steps: list[Runnable] = []
    for i, name in enumerate(models):
        step = adapt(_base_model(name, temperature))
        if i == 0 and role == "main":
            step = RunnableLambda(_fault_gate, name="fault_gate") | step
        if i > 0:
            step = _fallback_marker(role, models[0], name) | step
        steps.append(step.with_config(metadata={"model": name}))
    primary, *fallbacks = steps
    if not fallbacks:
        return primary
    # LangChain fallbacks take exception types, not a predicate; HTTPStatusError also covers
    # 4xx, which is acceptable: a rejected request just gets a second opinion before failing.
    return primary.with_fallbacks(
        fallbacks,
        exceptions_to_handle=(SimulatedProviderError, TimeoutError, httpx.HTTPError),
    )


def get_llm(role: Role, *, tools: Sequence[Any] | None = None) -> Runnable:
    """Chat model (with fallbacks) for a role, optionally with tools bound."""
    if tools:
        return _build(role, lambda m: m.bind_tools(list(tools)))
    return _build(role, lambda m: m)


def get_structured_llm[SchemaT: BaseModel](role: Role, schema: type[SchemaT]) -> Runnable:
    """Model whose output is parsed into `schema` via the provider's function calling."""
    return _build(role, lambda m: m.with_structured_output(schema))


def _json_mode_llm(role: Role) -> Runnable:
    return _build(role, lambda m: m.bind(response_format={"type": "json_object"}))


# --------------------------------------------------------------------------- call helpers


def _run_config(role: Role, config: RunnableConfig | None) -> RunnableConfig:
    """LangSmith metadata on every call: role, primary model, and a readable run name.

    Built on `ensure_config`, so inside a graph node the call extends the node's config
    (callbacks, metadata, tags, configurable) instead of replacing it. A plain dict would
    drop the inherited metadata while LangChain still inherits `configurable`; LangChain then
    copies the LangSmith tracer to add `thread_id` back as a default, nested runnables end up
    with two tracer copies sharing one run map, and every child run is ended twice
    ("No indexed run ID" warnings, runs detached from the graph trace).
    """
    cfg = ensure_config(config)
    cfg["run_name"] = (config or {}).get("run_name") or f"llm:{role}"
    cfg["metadata"] = {
        **(cfg.get("metadata") or {}),
        "llm_role": role,
        "llm_model": _models_for(role)[0],
        "llm_provider": get_settings().llm_provider,
    }
    cfg["tags"] = [*(cfg.get("tags") or []), f"llm:{role}"]
    return cfg


def _retrying() -> AsyncRetrying:
    def before_sleep(state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        log.warning("llm_retry", attempt=state.attempt_number, error=repr(exc))

    return AsyncRetrying(
        stop=stop_after_attempt(get_settings().llm_max_retries + 1),
        wait=wait_exponential_jitter(initial=RETRY_INITIAL_S, max=RETRY_MAX_S),
        retry=retry_if_exception(is_transient),
        before_sleep=before_sleep,
        reraise=True,
    )


async def _guarded_ainvoke(
    runnable: Runnable, messages: LanguageModelInput, role: Role, config: RunnableConfig | None
) -> Any:
    """Semaphore + per-attempt timeout + transient-only retries around one invocation.

    The slot is taken per attempt, so a call waiting out a backoff doesn't block other calls.
    """
    cfg = _run_config(role, config)
    timeout_s = get_settings().llm_timeout_s
    start = time.perf_counter()
    try:
        async for attempt in _retrying():
            with attempt:
                async with _semaphore, asyncio.timeout(timeout_s):
                    result = await runnable.ainvoke(messages, config=cfg)
    except Exception as exc:
        log.error("llm_failed", role=role, error=repr(exc), transient=is_transient(exc))
        if is_transient(exc):
            raise LLMError(f"{role} model unavailable after retries: {exc!r}") from exc
        raise
    log.info("llm_call", role=role, latency_ms=round((time.perf_counter() - start) * 1000))
    return result


async def ainvoke_llm(
    role: Role,
    messages: LanguageModelInput,
    *,
    tools: Sequence[Any] | None = None,
    config: RunnableConfig | None = None,
) -> AIMessage:
    """Invoke the role's model under the shared limits. Raises LLMError when retries run out."""
    return await _guarded_ainvoke(get_llm(role, tools=tools), messages, role, config)


async def ainvoke_structured[SchemaT: BaseModel](
    role: Role,
    schema: type[SchemaT],
    messages: LanguageModelInput,
    *,
    config: RunnableConfig | None = None,
) -> SchemaT:
    """Structured output via function calling; JSON mode + Pydantic parse if the schema is
    rejected by the provider or the tool-call output cannot be parsed."""
    try:
        return await _guarded_ainvoke(get_structured_llm(role, schema), messages, role, config)
    except Exception as exc:
        if not (_is_rejected_request(exc) or isinstance(exc, OutputParserException)):
            raise
        log.warning(
            "structured_output_fallback", role=role, schema=schema.__name__, error=repr(exc)
        )

    instruction = SystemMessage(
        "Respond with only a JSON object that validates against this JSON schema:\n"
        + json.dumps(schema.model_json_schema())
    )
    msgs = [instruction, *convert_to_messages(messages)]  # type: ignore[arg-type]
    raw: AIMessage = await _guarded_ainvoke(_json_mode_llm(role), msgs, role, config)
    return schema.model_validate_json(raw.text)


async def astream_llm(
    role: Role,
    messages: LanguageModelInput,
    *,
    config: RunnableConfig | None = None,
) -> AsyncIterator[AIMessageChunk]:
    """Stream tokens. The whole stream holds one semaphore slot until it finishes.

    Retries and fallbacks apply only until the first token arrives, because after that the user
    has already seen partial output. Each chunk wait is bounded by LLM_TIMEOUT_S, so a stalled
    stream fails instead of hanging.
    """
    runnable = get_llm(role)
    cfg = _run_config(role, config)
    timeout_s = get_settings().llm_timeout_s
    async with _semaphore:
        stream: AsyncIterator[AIMessageChunk] | None = None
        try:
            async for attempt in _retrying():
                with attempt:
                    stream = aiter(runnable.astream(messages, config=cfg))
                    async with asyncio.timeout(timeout_s):
                        first = await anext(stream)
        except StopAsyncIteration:
            return
        except Exception as exc:
            if is_transient(exc):
                raise LLMError(f"{role} stream unavailable after retries: {exc!r}") from exc
            raise
        yield first
        assert stream is not None
        try:
            while True:
                async with asyncio.timeout(timeout_s):
                    chunk = await anext(stream)
                yield chunk
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            raise LLMError(f"{role} stream stalled for {timeout_s}s") from exc
