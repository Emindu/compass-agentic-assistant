"""One traced LLM call per configured model, to prove keys and LangSmith tracing work.

Run: uv run python -m scripts.smoke_llm   → then check the run in your LangSmith project.
"""

import asyncio

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from app.core.config import get_settings


async def main() -> None:
    # LangSmith/LLM SDKs read their keys from os.environ, so load .env into it first.
    load_dotenv()
    s = get_settings()
    for label, model in (("reasoning", s.llm_model), ("fast", s.fast_llm_model)):
        llm = init_chat_model(model, model_provider=s.llm_provider, timeout=s.llm_timeout_s)
        reply = await llm.ainvoke(
            "Reply with one short sentence confirming you are reachable.",
            config={"run_name": f"smoke_{label}", "tags": ["smoke", label]},
        )
        print(f"{label:9} {model}: {reply.content}")
    print(f"Traces should appear in LangSmith project '{s.langsmith_project}'.")


if __name__ == "__main__":
    asyncio.run(main())
