# ADR 0001: Mistral models in three roles

## Context
The assistant needs chat quality on visible steps (routing, final answer), cheap high-volume
classification (input guard, RLM batch map), and reliable Python for the RLM planner. The free
Mistral plan has workspace-wide rate limits. The bank values data residency and low lock-in.

## Decision
Use Mistral via `langchain-mistralai` for all three roles, built only through
`init_chat_model`: `main` = `mistral-large-latest` (supervisor, response, RLM reduce),
`fast` = `mistral-small-latest` (guard, batches, summaries), `code` = `codestral-latest`
(planner). Every call shares one semaphore (`LLM_MAX_CONCURRENCY`); retries apply to transient
errors only; main falls back to fast, code to `CODE_LLM_FALLBACK` then main.

## Consequences
- One vendor keeps keys, limits and tracing in one place; open weights allow a future
  self-hosted deployment without prompt or graph redesign.
- Fan-out concurrency is bounded by the shared semaphore — acceptable, predictable latency.
- Free-plan data terms are unsuitable for real bank data; production needs a paid tier or
  self-hosting. Switching providers stays an env change, not a code change.
