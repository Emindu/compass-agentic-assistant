# Compass — design notes

## Model selection rationale

Compass uses Mistral models through `langchain-mistralai`, in three roles set from env
(`app/core/llm.py`):

| Role | Default model | Used for | Why this tier |
| --- | --- | --- | --- |
| main | `mistral-large-latest` | supervisor routing, final response, RLM reduce | Few calls whose quality the user sees directly. |
| fast | `mistral-small-latest` | input guard, RLM batch map, citation check | Runs on every request or once per batch, so latency and cost matter more. |
| code | `codestral-latest` | RLM planner (writes the Python search plan) | A code-specialised model writes valid Python for the sandbox more reliably. |

**Why Mistral**

- Strong function calling, which the supervisor, tools and structured outputs rely on.
- Open-weight lineage: a bank could self-host the same model families for data residency,
  without redesigning prompts or the agent graph.
- One vendor for chat and code models keeps keys, limits and tracing in one place.

**Trade-offs**

- The free plan has tight, workspace-wide rate limits, so every LLM call shares one semaphore
  (`LLM_MAX_CONCURRENCY`) and retries 429/5xx/timeouts with exponential backoff and jitter.
  RLM fan-out is therefore bounded, not unlimited.
- Free-plan data-use terms are not acceptable for real bank data. Production would use a paid
  tier with no training on inputs, or self-hosted models.
- Provider lock-in is low: models are built with `init_chat_model`, so another provider is an
  env change (`LLM_PROVIDER`, model names).

**Resilience:** main falls back to fast, and code falls back to `CODE_LLM_FALLBACK` and then
main. Each fallback is logged as `llm_fallback` and visible in LangSmith. `FAULT_LLM=true`
simulates a main-model outage for the demo.

## Retrieval design

- **Two Pinecone indexes, one per signal.** `compass-dense` (llama-text-embed-v2, cosine) finds
  paraphrases; `compass-sparse` (pinecone-sparse-english-v0, dot product) matches exact terms
  such as `HikariPool` or `INC-2026-0214`. Separate indexes let both run in parallel and let the
  UI show both raw scores.
- **Namespaces per department** (`payments`, `platform`, `security`, `retail`, `corporate`), so
  a query can be scoped and a department re-indexed on its own. Namespaces are not a security
  boundary.
- **Access control is a metadata filter from the JWT role**, always the first clause:
  `access_rank <= clearance` (viewer → internal, analyst → confidential, admin → restricted).
  The LLM never sets it.
- **Fusion:** each list is divided by its top score, then
  `fused = alpha * dense + (1 - alpha) * sparse` (`HYBRID_ALPHA`, default 0.6). Max-normalisation
  keeps relevant low-ranked hits above zero; a single weight is easier to explain than RRF.
- **At most 2 chunks per document** by default, so several sources can be cited instead of one
  long document filling every slot.
- **Degradation:** if one index fails, the other answers alone; if both fail (or
  `FAULT_PINECONE=true`), a local `rank_bm25` index over the same chunks answers with the same
  access filters. Results carry `degraded` and a user-facing note.
- **Latency:** about 0.4 s warm for 10 parallel calls (5 namespaces × 2 indexes); the first call
  pays about 4 s of connection setup, so the API warms the clients at start-up.
