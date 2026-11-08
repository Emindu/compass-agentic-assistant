# ADR 0005: Human approval for admin actions via LangGraph interrupt

## Context
Admin actions (re-index a department, revoke a document) mutate shared state. A model that both
proposes and executes them could act on a hallucinated or injected request, and a UI button alone
proves nothing to auditors — the enforcement must be server-side.

## Decision
The tools agent may only *propose* admin actions: it validates arguments, renders a
human-readable description and pauses the whole graph with LangGraph `interrupt()`. The
checkpointer holds the interrupted state; the stream surfaces an `approval` event; only
`POST /chat/resume` (admin-only endpoint) continues with `Command(resume={"approved": ...})`.
`execute_admin_tool` re-checks the role and requires `approved=True` at the effectful boundary
and writes an `admin_action` audit log entry either way. Everything before `interrupt()` is free
of side effects because it re-runs on resume.

## Consequences
- Approval is enforced in the server, visible in traces, and auditable — the model cannot approve
  its own request, and a manipulated tool call still cannot execute without a human decision.
- The turn stays open on the checkpointer until resolved; anything before the interrupt must stay
  idempotent (validate + describe only, which the code does).
- Any future effectful admin tool must follow the same pattern (describe → interrupt → execute).
