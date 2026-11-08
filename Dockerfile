# Compass: one image for every service (API, UI). The runtime is the uv-managed project
# virtualenv, so uvicorn, streamlit and the stdio MCP server all share one lock-resolved
# environment.
FROM python:3.12-slim

# uv and uvx copied from the official image, pinned to a real minor tag.
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# util-linux provides `unshare`: the RLM sandbox runs each generated plan under
# `unshare -rn` to give it an empty user+network namespace. Container runtimes that
# forbid user namespaces are fine — the sandbox probes for unshare and degrades
# gracefully (it logs "degraded mode" and runs the plan without the namespace).
RUN apt-get update \
    && apt-get install -y --no-install-recommends util-linux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first and frozen: copying only the manifest and lock keeps this layer
# cached across rebuilds when just the source changed.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application code and demo data. tests/ are dev-only (uv sync --no-dev already skipped
# pytest); .env never enters the image — compose injects it per service via env_file.
COPY app/ app/
COPY ingestion/ ingestion/
COPY mcp_server/ mcp_server/
COPY ui/ ui/
COPY scripts/ scripts/
COPY data/corpus/ data/corpus/
COPY evals/ evals/

# Non-root runtime. The only writable path the API needs is data/checkpoints (a named
# volume); it is pre-created with the right ownership so SQLite can create its file
# there on first boot.
RUN useradd --system --create-home compass \
    && mkdir -p data/checkpoints \
    && chown -R compass:compass /app
USER compass
