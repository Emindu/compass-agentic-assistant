"""Application settings, loaded once from environment / `.env`.

Why one settings object: every tunable (models, fusion weight, rate limits, fault flags) lives
here, so the demo can flip behaviour through env vars without code changes, and tests can
override a single object.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "Compass"
    environment: Literal["dev", "demo", "prod"] = "dev"
    log_level: str = "INFO"

    # --- LLMs (provider-agnostic via LangChain init_chat_model; see app/core/llm.py) ---
    llm_provider: str = "mistralai"
    mistral_api_key: SecretStr | None = None
    llm_model: str = "mistral-large-latest"  # main: supervisor, response, RLM reduce
    fast_llm_model: str = "mistral-small-latest"  # fast: input guard, RLM batch map, citations
    code_llm_model: str = "codestral-latest"  # code: RLM planner writes the Python search plan
    code_llm_fallback: str = "codestral-latest"  # used if code_llm_model is unavailable
    # One semaphore for every LLM call: Mistral limits are per workspace, shared across keys.
    llm_max_concurrency: int = Field(default=3, ge=1)
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=4, ge=0)

    # --- Pinecone ---
    pinecone_api_key: SecretStr | None = None
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_dense_index: str = "compass-dense"
    pinecone_sparse_index: str = "compass-sparse"
    pinecone_dense_model: str = "llama-text-embed-v2"
    pinecone_sparse_model: str = "pinecone-sparse-english-v0"
    pinecone_timeout_s: float = 10.0
    # Weight of the dense score in hybrid fusion: fused = alpha*dense + (1-alpha)*sparse.
    hybrid_alpha: float = Field(default=0.6, ge=0.0, le=1.0)

    # --- LangSmith (the SDK reads LANGSMITH_* itself; kept here for visibility/validation) ---
    langsmith_api_key: SecretStr | None = None
    langsmith_tracing: bool = True
    langsmith_project: str = "compass-agentic-assistant"

    # --- Auth / rate limiting ---
    jwt_secret: SecretStr = SecretStr("change-me-in-env")
    jwt_algorithm: str = "HS256"
    jwt_ttl_minutes: int = 120
    rate_limit_capacity: int = 10  # bucket size: burst allowance per user
    rate_limit_refill_per_s: float = 0.2  # tokens added per second (12 requests/minute)

    # --- Fault injection (demo of graceful degradation) ---
    fault_llm: bool = False
    fault_pinecone: bool = False
    fault_mcp: bool = False
    fault_tool_timeout: bool = False


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so settings are parsed once per process."""
    return Settings()
