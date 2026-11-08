# ADR 0004: Guardrails as graph nodes, not middleware

## Context
Input filtering, citation checking and output inspection could all live as FastAPI middleware or
response post-processing. But their verdicts determine control flow (block the turn, retry the
draft, replace the answer), and the UI must show every step like any other agent.

## Decision
Make them graph nodes inside the request path: `input_guard` before the supervisor (branching to
finalize when blocked), `validator` after `response` (retry loop ≤ 2 with issues fed back, then
citation stripping or the honest "couldn't find" answer), and the deterministic output guard
inside the validator. Verdicts are typed objects in state (`GuardVerdict`, `ValidationResult`);
every check emits typed events (`guardrail`, `validation`) to the SSE stream; the wrapper
`instrumented` contains node failures.

## Consequences
- Guard decisions are visible in state, traces and the activity panel, and route on typed values
  rather than parsed text.
- Guards are enforced for every entry point that runs the graph (API, CLI, future channels) —
  no middleware to forget when wiring a new route.
- Feedback-labelled retries make prompts/flows slightly more complex, and guards share the
  latency budget of the request path (kept cheap: pattern/string work plus one small model call).
