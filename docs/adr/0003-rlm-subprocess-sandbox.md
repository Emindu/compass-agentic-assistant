# ADR 0003: RLM plans run in a locked-down subprocess, primitives in the parent

## Context
The RLM planner emits arbitrary Python. Python-level restriction alone is bypassable (dunder
traversal, smuggled imports, C-level escapes), so plans are untrusted code that must never gain
access beyond the calling user's clearance — and that clearance must be enforced where the plan
cannot reach it.

## Decision
Run each plan in a disposable child interpreter with stacked layers: static AST allow-list
(length cap, import allow-list, dunder bans, banned callables/attributes), restricted builtins
with a runtime import re-check, an empty child environment, kernel rlimits (CPU, memory, files)
and `unshare -rn` user+network namespaces when the kernel allows. Plans reach data only through
JSON-RPC primitives (`list_documents`, `get_sections`, `search`, `llm_query`) executed in the
parent process, which re-derive clearance from the session's verified user and never from plan
arguments. Unknown and forbidden documents both return nothing, so plans cannot probe for hidden
documents. Budgets bound each run (30 s, 40 calls, result size; llm_query ≤ 12 calls).

## Consequences
- Defence in depth: any single bypassed layer is not sufficient, and access control survives —
  the sandbox can only shrink what the plan can do, never widen what the user can see.
- A kernel without user namespaces degrades to rlimits-only (logged once per process);
  production hardening moves the child into gVisor / Firecracker.
- Subprocess and RPC plumbing add complexity and a fixed latency floor per research run.
