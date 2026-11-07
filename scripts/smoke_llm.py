"""Smoke-test the Mistral setup end to end; every call is traced to LangSmith.

Checks: which configured models this key can use (and whether a GLM model is offered), one plain
call per role, a tool call on the main model, a streamed reply, and the FAULT_LLM fallback.

Run: uv run python -m scripts.smoke_llm
"""

import asyncio
import os
import time
from collections.abc import Awaitable
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from mistralai.client import Mistral

from app.core import llm
from app.core.config import get_settings
from app.core.logging import configure_logging


@tool
def lookup_service_owner(service_name: str) -> str:
    """Return the owning team for an internal service (dummy data for the smoke test)."""
    return {"payments-gateway": "Payments Platform"}.get(service_name, "unknown")


async def timed(label: str, call: Awaitable[Any]) -> Any:
    start = time.perf_counter()
    try:
        result = await call
    except Exception as exc:  # report and continue: a smoke test should show every check
        print(f"  FAIL  {label:28} {time.perf_counter() - start:6.2f}s  {exc!r}")
        return None
    print(f"  ok    {label:28} {time.perf_counter() - start:6.2f}s")
    return result


async def check_models() -> None:
    s = get_settings()
    assert s.mistral_api_key is not None, "MISTRAL_API_KEY is not set (.env)"
    async with Mistral(api_key=s.mistral_api_key.get_secret_value()) as client:
        listing = await client.models.list_async()
    ids = sorted({m.id for m in listing.data or []})
    print(f"Models available to this key: {len(ids)}")
    for env_name, model in (
        ("LLM_MODEL", s.llm_model),
        ("FAST_LLM_MODEL", s.fast_llm_model),
        ("CODE_LLM_MODEL", s.code_llm_model),
        ("CODE_LLM_FALLBACK", s.code_llm_fallback),
    ):
        print(f"  {'yes' if model in ids else 'NO ':4} {env_name:18} {model}")
    glm = [i for i in ids if "glm" in i.lower()]
    print(f"  zai-glm-5-2 available: {'zai-glm-5-2' in ids}; GLM models: {glm or 'none'}")


async def main() -> None:
    load_dotenv()  # LangSmith and provider SDKs read os.environ
    configure_logging("WARNING")  # keep output readable; fallback/retry warnings still show
    s = get_settings()
    print(f"Provider: {s.llm_provider}  LangSmith project: {s.langsmith_project}\n")

    await check_models()

    print("\nPlain call per role")
    for role in ("main", "fast", "code"):
        reply = await timed(
            f"{role}", llm.ainvoke_llm(role, "Reply with one short sentence: are you reachable?")
        )
        if reply is not None:
            print(f"        → {reply.text[:100]}")

    print("\nTool calling (main)")
    reply = await timed(
        "main + bind_tools",
        llm.ainvoke_llm(
            "main",
            "Who owns the payments-gateway service? Use the tool.",
            tools=[lookup_service_owner],
        ),
    )
    if reply is not None:
        print(f"        → tool_calls={[(c['name'], c['args']) for c in reply.tool_calls]}")

    print("\nStreaming (main)")

    async def stream() -> int:
        n = 0
        async for _chunk in llm.astream_llm("main", "Count from 1 to 5, comma separated."):
            n += 1
        return n

    chunks = await timed("main astream", stream())
    if chunks is not None:
        print(f"        → {chunks} chunks")

    print("\nFallback with FAULT_LLM=true (expect an llm_fallback warning, answer from fast)")
    os.environ["FAULT_LLM"] = "true"
    get_settings.cache_clear()
    reply = await timed(
        "main with simulated outage",
        llm.ainvoke_llm("main", "Reply with the word: fallback", config={"tags": ["fault-demo"]}),
    )
    if reply is not None:
        print(f"        → served by {reply.response_metadata.get('model_name', '?')}: {reply.text}")
    os.environ["FAULT_LLM"] = "false"
    get_settings.cache_clear()

    print(f"\nDone. Check runs tagged llm:* in LangSmith project '{s.langsmith_project}'.")


if __name__ == "__main__":
    asyncio.run(main())
