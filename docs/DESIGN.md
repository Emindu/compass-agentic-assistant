# Compass — design notes

## Goals & scope

Compass answers employee questions over the bank's internal documents with citations, while
showing its work. The POC brief weights agent architecture, RAG, LangGraph, RLM, security,
observability, async behaviour, RBAC, code quality and docs. Within that scope the goals were:

- Answers grounded in retrieved, access-filtered evidence — never a bare model claim.
- Every internal step visible: routing, retrieval scores, RLM plan code, tool calls, validations.
- Failures contained: any component can degrade without taking the request down.
- Role-based access enforced in code paths (not prompts) everywhere data is touched.

Deliberately **cut** from the POC, with reasons:

| Cut | Why |
| --- | --- |
| Long-term per-user memory (cross-session profile/preferences) | Conversation memory (SQLite checkpointer) covers the demo requirements; a long-term store adds schema, retention policy and PII review that a POC cannot validate. Next step after the assessment. |
| Answer-quality feedback loop (thumbs up/down, guardrail tuning from real traffic) | No real users yet; feedback with no traffic is unmeasurable. |
| Full eval harness in CI | A LangSmith eval run now exists: `uv run python -m scripts.run_evals` syncs `evals/ground_truth.json` (seven QA cases plus the RLM case with `--include-rlm`) into a dataset and scores it with deterministic evaluators; `--dry-run` scores locally without LangSmith. CI wiring (stable, budgeted LLM runs on a schedule) is the remaining next step. |

## Architecture

Request path, gateway to response:

1. `POST /auth/login` verifies demo credentials (PBKDF2 hashes) and mints an HS256 JWT whose
   claims mirror `UserContext` (user_id, name, role, department).
2. `POST /chat/stream` verifies the bearer token, applies the per-user token-bucket rate limit,
   and starts the graph under a thread id namespaced as `{user_id}:{thread_id}` — one user can
   never load or continue another user's thread.
3. The graph runs under a LangSmith run (`compass_chat`) tagged with role, user, thread and the
   HTTP request id; custom activity events stream to the client over SSE as they happen.
4. On an admin approval, the graph checkpoints its interrupt state; the stream ends with an
   `approval` event, and `POST /chat/resume` (admin only) continues with `Command(resume=...)`.

The graph (`app/agents/graph.py`), nodes and edges exactly:

```
START → load_memory → input_guard ─┬─(blocked)────────────────────────────→ finalize
                                    └─(allowed)→ supervisor

supervisor ─┬─(docs)─────→ retrieval → response
             ├─(research)→ research  → response
             ├─(tools)────→ tools ─┬───→ response
             │                   └─(also docs)→ retrieval → response
             └─(smalltalk/out of scope)→ response

response → validator ─┬─(invalid, retries left)→ response
                      └─(valid / out of retries)→ finalize → update_memory → END
```

Typed state (`AgentState`):

| Field | Scope | Notes |
| --- | --- | --- |
| `messages` | persistent | LangGraph `add_messages` reducer; conversation history |
| `summary` | persistent | Rolling summary of older turns |
| `user` | persistent | `UserContext` from the verified JWT |
| `route: RouteDecision \| None` | per turn | Intent, agents, rewritten search query, departments, reasoning |
| `guard: GuardVerdict \| None` | per turn | Input-guard verdict (typed branch, not parsed text) |
| `retrieved: list[RetrievedChunk]` | per turn | Evidence from retrieval or RLM |
| `retrieval_notes` | per turn | User-facing degradation notes |
| `research: ResearchOutcome \| None` | per turn | Synthesis, themes, findings, data, documents, plan code, depth |
| `tool_results` | per turn | Tool outputs or errors (never both keys) |
| `draft_answer` | per turn | Written by `response`, rewritten by `validator` |
| `validation: ValidationResult \| None` | per turn | Citation check result |
| `response_attempts` | per turn | Retry budget for the response→validator loop |
| `errors: list[AgentError]` | per turn | Appending reducer; a node returning `None` resets the list |

Error containment: every node is wrapped by `instrumented`
(`app/agents/nodes/common.py`). A node that raises records an `AgentError` (node, code, message,
retryable), emits an `error` event, and returns its per-node fallback update, so downstream
nodes still run on whatever state is available. Examples: a supervisor failure degrades routing to
document search with the raw question; a retrieval failure yields empty results plus note; a
response-model failure returns the most relevant sources as text; an MCP failure yields a failed
tool result the answer can mention. `GraphInterrupt` is re-raised so LangGraph checkpoints the
approval pause instead of recording it as an error. Conditional edges branch on typed values
(`GuardVerdict`, `RouteDecision`, `ValidationResult`), never on free text.

## Model selection

Compass uses Mistral models through `langchain-mistralai`, in three roles set from env
(`app/core/llm.py`):

| Role | Default model | Used for | Why this tier |
| --- | --- | --- | --- |
| main | `mistral-large-latest` | supervisor routing, final response, RLM reduce, tools planning | Few calls whose quality the user sees directly. |
| fast | `mistral-small-latest` | input guard, RLM batch map, rolling summary, `llm_query` | Runs on every request or once per batch, so latency and cost matter more. |
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
simulates a main-model outage for the demo. Retries use tenacity on transient errors only
(429, 5xx, timeouts); a non-transient 4xx fails fast. The semaphore slot is taken per attempt, so
a call waiting out a backoff does not block other calls. Structured output falls back to JSON
mode plus a Pydantic parse when the provider rejects the schema. Streaming holds one semaphore
slot for the stream's life; retries and fallbacks apply until the first token arrives, and every
chunk wait is bounded so a stalled stream fails instead of hanging.

## Memory

- The SQLite checkpointer (`data/checkpoints/compass.sqlite`) persists `messages` and `summary`
  per thread; the API scopes thread ids as `{user_id}:{thread_id}`, so memory is per user and per
  conversation. Thread state is also what holds a pending approval interrupt across requests.
- Token cap: once a thread exceeds `MAX_MESSAGES = 12` messages, `update_memory` folds everything
  beyond the last `KEEP_MESSAGES = 6` into a rolling summary written by the fast model and removes
  the originals. The supervisor and response agents see the last 6 messages verbatim plus the
  summary.
- `load_memory` resets per-turn fields at the start of each turn, so a new question never sees
  stale route/guard/validation state.
- User context (`UserContext`, including clearance) comes from the JWT into state each turn,
  never from chat text.
- Every memory read or write emits a `memory` event (loaded / saved / summarised), so the activity
  panel and traces show the memory the answer used.
- Next step: a long-term per-user store (profile, preferred departments, pinned documents) behind
  the same event discipline, with retention policy and PII review.

## RAG

- **Chunking** (`ingestion/chunk.py`): documents split on `##` sections; long sections packed by
  paragraph up to 1200 characters; a contextual header (doc id, title, status, type, department,
  service, date, section) is prefixed to every chunk, and chunk ids are deterministic
  (`{doc_id}#{index:02d}`), so re-ingesting overwrites rather than duplicates.
- **Metadata** (validated YAML front matter): doc_id, title, department, document_type,
  access_level (+ integer `access_rank`), created date (+ `created_ts` int for range filters),
  service, status (final/draft/superseded), severity, version.
- **Namespaces per department** (`payments`, `platform`, `security`, `retail`, `corporate`), so
  a query can be scoped and a department re-indexed on its own. Namespaces are not a security
  boundary.
- **Two Pinecone indexes, one per signal, both with integrated (hosted) embedding models.**
  `compass-dense` (llama-text-embed-v2, cosine) finds paraphrases; `compass-sparse`
  (pinecone-sparse-english-v0, dot product) matches exact terms such as `HikariPool` or
  `INC-2026-0214`. Separate indexes let both run in parallel and let the UI show both raw scores.
  No second embedding vendor key is needed.
- **Access control is a metadata filter from the JWT role**, always the first clause:
  `access_rank <= clearance` (viewer → internal, analyst → confidential, admin → restricted).
  The LLM never sets it, and the prompt cannot widen it.
- **Fusion:** each list is divided by its top score, then
  `fused = alpha * dense + (1 - alpha) * sparse` (`HYBRID_ALPHA`, default 0.6). Max-normalisation
  keeps relevant low-ranked hits above zero; a single weight is easier to explain than RRF.
- **`max_per_doc = 2`** chunks per document by default, so several sources can be cited instead
  of one long document filling every slot. Both searches over-fetch `top_k × 3` candidates so
  fusion has material on both sides.
- **Degradation:** if one index fails, the other answers alone (alpha collapses to 0 or 1); if
  both fail (or `FAULT_PINECONE=true`), a local `rank_bm25` index over the same chunks answers with
  the same access filters — degradation never widens access. Results carry `degraded` and a
  user-facing note, and are marked `source: "bm25_fallback"`.
- **Citations:** answers cite retrieved chunk ids as `[DOC-ID#NN]` (e.g. `[RB-PAY-003-V2#03]`);
  the validator checks every bracketed id against what was actually retrieved. The UI shows
  `doc_id › section` per citation.
- **Latency:** about 0.4 s warm for 10 parallel calls (5 namespaces × 2 indexes); the first call
  pays about 4 s of connection setup, so the API warms the clients at start-up.

## RLM (recursive language model)

Flow (`app/rlm/graph.py`, subgraph `rlm_research`): **plan → execute → Send ×N analyse_batch →
reduce**, with depth-capped recursion and a plan-retry loop.

- **Planner:** the code model writes a short Python "search plan" (≤ 60 lines) that gathers
  evidence — it does not answer the question. The plan must end by assigning a dict to `result`
  with documents, batches of evidence chunk ids, a task description and optional computed data.
- **Sandbox execution:** the plan runs in a disposable child interpreter
  (`app/rlm/sandbox.py`), with defence in depth:
  1. a static AST allow-list (`validate_plan`) — length cap, import allow-list
     (json, re, math, statistics, collections, itertools, datetime, functools), dunder-name
     rejection, banned callables (eval/exec/compile/open/getattr/…), banned attribute prefixes
     that walk frames or code objects, no global/nonlocal;
  2. restricted builtins in the child, with `__import__` re-checking the allow-list at runtime;
  3. a minimal child environment (no inherited env, so no API keys leak) and kernel rlimits
     (CPU 20 s, address space 512 MB, NOFILE 32, file size 1 MB);
  4. `unshare -rn` (user + network namespace) when the kernel allows it, so the child has no
     network stack at all.
- **Primitives run in the parent, not the sandbox** (`app/rlm/primitives.py`): the plan is
  untrusted code, so every capability it reaches must enforce the caller's clearance *outside*
  that code. Each primitive re-derives access from the session's `UserContext` (built from the
  verified JWT), never from plan arguments. `list_documents()` returns metadata only; `get_sections()`
  returns unknown and forbidden doc ids identically as `[]`, so a plan cannot even detect that a
  hidden document exists; `search()` runs hybrid retrieval with clearance inside the query;
  `llm_query()` is a budgeted recursive sub-call to the fast model with sanitised context.
- **Fan-out and recursion:** execute registers every chunk it returned as evidence and accepts
  only those ids in batches (a plan cannot invent evidence). Batches (≤ 8 batches, ≤ 8 chunk ids
  each) are analysed in parallel via LangGraph `Send`, one fast-model call per batch, each seeing
  only its own chunks. `reduce` (main model) aggregates findings into recurring themes citing
  chunk ids, and may ask one follow-up question that re-enters the planner — depth-capped at 2
  (initial pass + one recursion), with `MAX_PLAN_ATTEMPTS = 2` per pass. A failed batch is
  contained and does not sink the analysis.
- **Budgets:** per session, `llm_query` ≤ 12 calls and ≤ 12,000 context characters; per plan run,
  30 s wall clock, ≤ 40 primitive calls, result ≤ 200 KB; each failure mode yields a short error
  message the planner can correct on its retry.
- **Why metadata-first selection:** in testing, plans that filtered candidates by keywords in
  titles (or by service name) silently dropped relevant documents — a card-switch TLS outage is
  still a "payment failure", and titles rarely use the user's words. The planner is therefore
  instructed (in the system prompt *and* in the user-role prompt, because the code model weighs
  it more) to select with `list_documents()` metadata filters and read sections of every
  candidate, using keyword `search()` only when metadata cannot narrow. The evidence-first flow
  also caps context: each batch call sees a few chunks, not the corpus.

## Security

Layers in request order — controls run inside the graph so their verdicts are state, branches and
events, not middleware side effects:

| # | Layer | Mechanism |
| --- | --- | --- |
| 1 | Authentication + rate limit | HS256 JWT (verified claims rebuild `UserContext`); same error for unknown user and wrong password; per-user token bucket (10 burst, 0.2/s refill) applied after auth. |
| 2 | Input guard | Deterministic patterns first (instruction override, role hijack, jailbreak, prompt reveal, bulk dump, exfiltration URLs/hosts, secrets, financial advice, abuse; 4,000-char cap), then a fast-model classifier that blocks only at confidence ≥ 0.75; fail-open to the pattern verdict if the classifier errors. Blocked turns never reach the supervisor. |
| 3 | Role-filtered routing | The supervisor's routing is filtered by `can_use` before tools or analytics are offered; denied attempts emit events (analytics → document search; admin request from non-admin → deterministic refusal). |
| 4 | Tool binding, twice | The LLM is only shown `registry.tools_for(role)`, and proposed tool names are re-checked against the same set; RBAC is re-checked again inside `call_mcp_tool` and `execute_admin_tool`. Tool arguments are validated against Pydantic models (`extra="forbid"`) before any I/O. |
| 5 | Access-level filters | Retrieval always filters `access_rank <= clearance` (first clause), the RLM primitives re-derive clearance from the session, and the BM25 fallback applies the identical filter set. |
| 6 | Sanitiser | Retrieved chunk text and tool results are untrusted data: invisible/control characters, HTML comments, external image beacons, wrapper-forging `<document>` tags and instruction-like lines are stripped before prompts; `<document>` and `<tool_result>` wrappers delimit data from instructions; flagged chunks surface as guardrail events. |
| 7 | Citation validator | Deterministic set difference between bracketed citation ids and retrieved chunk ids — a model cannot invent a source. Retry budget of 2 with the issues fed back, then strip unknown citations with a caveat, or replace an entirely unsupported answer. |
| 8 | Output guard | Regex checks on the drafted answer: block personal investment advice, secret-shaped strings (key prefixes, JWTs, private-key headers — recoverable by redaction) and verbatim system-prompt fragments; warn on profanity, all-caps shouting, exclamation spam and non-bank links. |
| 9 | Sandbox + approval | RLM plan code runs only in the sandbox with parent-side primitives; admin actions pause the graph on `interrupt()` and require a human decision via `POST /chat/resume` (admin-only endpoint) — the model cannot approve its own request, and `execute_admin_tool` re-checks the role and the approval at the effectful boundary. |
| 10 | Data minimisation | The MCP server strips the employee `private` sub-object (salary, national_id) field-by-field; queries are length- and control-character-checked. |

## RBAC matrix

Clearance ceiling per role (document access): viewer → `internal`, analyst → `confidential`,
admin → `restricted`. Tool permissions (`app/security/rbac.py`):

| Tool / capability | viewer | analyst | admin |
| --- | --- | --- | --- |
| Document chat + search (`knowledge_search`) | ✓ | ✓ | ✓ |
| Sandbox Python analysis / RLM research (`python_analysis`) | — | ✓ | ✓ |
| MCP directory: `lookup_employee`, `get_service_status`, `list_open_incidents` | — | ✓ | ✓ |
| Admin: `reindex_department`, `revoke_document` (approval required) | — | — | ✓ |
| Resolve pending approvals (`POST /chat/resume`) | — | — | ✓ |

Unknown tools are denied by default. Every check derives from the verified JWT.

## Observability

- **LangSmith:** every chat run is `compass_chat` with tags `role:{role}`, `chat` and metadata
  `user_id`, `role`, `thread_id`, `request_id`; LLM calls carry run names `llm:{role}` and
  metadata `llm_role`/`llm_model`/`llm_provider`; retrieval functions are `@traceable` retrievers
  whose outputs render as documents; RLM subgraph runs nest under the research node; fallback
  models are visible per call.
- **structlog:** JSON logs on stdout; a `request_id` contextvar set in middleware appears on every
  line for the request (also honoured from an inbound `x-request-id` and returned on the
  response); `user_id` and `role` are bound per request; one `http_request` line per call; node
  failures, retries, fallbacks, guard verdicts and rate-limit hits each have structured events.
- **Activity panel** (`ui/activity.py`, events from `app/agents/events.py`): `node_start`,
  `node_end`, `route`, `retrieval`, `tool_call`, `research`, `memory`, `validation`, `guardrail`,
  `approval`, `token`, `answer`, `error`, `done`. The panel shows routing reasoning, retrieval
  hits with dense/sparse/fused scores per source, the RLM plan code, tool calls with previews and
  denials, validation issues, sanitisation flags, memory actions, node timings, and Approve/Reject
  buttons while an approval is pending.

## Async & failure handling

- Everything async end to end: FastAPI, the graph, retrieval, MCP, sandbox and LLM calls.
- Timeouts: per-attempt LLM timeout (`LLM_TIMEOUT_S`, also bounding each stream chunk wait),
  Pinecone timeout for the whole namespace fan-out (10 s), MCP call timeout (10 s), sandbox wall
  clock (30 s). 
- Retries: tenacity, transient errors only (429, 5xx, timeouts, simulated faults), exponential
  backoff with jitter (1–20 s), bounded attempts; a non-transient 4xx fails fast.
- Concurrency: one process-wide semaphore (`LLM_MAX_CONCURRENCY`) shared by every LLM call
  including streams, so workspace-level rate limits are respected.
- Degraded paths: index failure → single-index results or BM25; model failure → model fallbacks,
  then source-listing fallback; MCP failure → failed tool result the answer reports; supervisor
  failure → document search with the raw question; validation failure → citation stripping or the
  honest "couldn't find" answer.
- Fault flags: `FAULT_LLM`, `FAULT_PINECONE`, `FAULT_MCP`, `FAULT_TOOL_TIMEOUT` flip a specific
  component into its failure path on demand (see README) — the demo shows each degraded answer.

## Assumptions

- The corpus (25 documents, 5 departments) is small, static and fictional; in-memory corpus
  reads (BM25, RLM primitives) are acceptable at this scale.
- One API replica: in-memory rate limiter and SQLite checkpointer are correct only for this.
- Pinecone free tier and Mistral free plan limits are the operating envelope.
- Demo users and passwords are public POC credentials; `JWT_SECRET` is set per deployment.
- Timezone-free dates (ISO dates in metadata) suffice; incident dates are fictional future dates.
- The MCP server and the API run in the same trust boundary for the POC (stdio transport, or
  in-process mode for tests).

## Trade-offs

- **SQLite checkpointer**: simplest durable memory and interrupt store; single writer, not
  multi-replica. Swappable behind LangGraph's `BaseCheckpointSaver`.
- **One semaphore for all LLM calls**: protects the workspace rate limit at the cost of peak
  latency during RLM fan-out — bounded, predictable worst case over fastest possible run.
- **Subprocess sandbox instead of gVisor/Firecracker**: layers (AST, builtins, env, rlimits,
  namespaces) cover the POC threat model; a VM-level sandbox is the production upgrade.
- **Deterministic validators (regex/set-difference) instead of LLM judges**: free, fast,
  reproducible; an LLM evidence-quality judge can layer on top later without changing the
  contract. Looser citation regex deliberately catches malformed invented ids.
- **SSE over WebSockets**: one-way typed event stream, proxy-friendly, simpler; the UI polls
  nothing.
- **Max-normalised weighted fusion instead of RRF**: one explainable knob (`HYBRID_ALPHA`), and
  both raw scores stay visible; rank-based RRF discards score magnitudes.
- **Hardcoded demo users**: assessment scope; only `app/api/auth.py` changes for OIDC.

## Production gaps

| Gap | What production needs |
| --- | --- |
| Identity | Bank IdP via OIDC (e.g. Keycloak): real users, MFA, short-lived tokens, group→role mapping. |
| Sandbox isolation | gVisor / Firecracker microVM per plan run instead of a plain subprocess. |
| Rate limiting | Redis-backed token bucket shared across replicas (interface unchanged). |
| Checkpointing | Postgres checkpointer for multi-replica durability and concurrency. |
| Reranking | Done for chat search (hosted `bge-reranker-v2-m3`, top 20 fused). Next: tune `RERANK_CANDIDATES` and fusion `alpha` against the eval set. |
| Evaluation | The `evals/ground_truth.json` seed wired into CI: regression runs of QA, RLM counts and guardrail cases with budgeted keys. |
| Pinecone tenancy | Per-tenant/department indexes or isolated projects with separate API keys, not just namespaces. |
| Secrets | A secrets manager (vault/KMS) for keys instead of `.env`, plus rotated short-lived credentials. |
| Long-term memory | A per-user store with retention limits and PII review, behind the existing event discipline. |
