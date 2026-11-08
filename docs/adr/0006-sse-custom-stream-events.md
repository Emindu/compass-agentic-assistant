# ADR 0006: Custom SSE stream events from inside the graph

## Context
The UI must show what the agent is doing as it happens: which node runs, the routing decision,
retrieval scores, the RLM plan code, tool calls, validation issues, tokens. LangGraph's default
stream modes expose updates and tokens, not application-level semantics. The demo also needs
update-style delivery through proxies without client polling.

## Decision
Nodes write typed events through LangGraph's custom stream writer (`emit` in
`app/agents/events.py`): `node_start`, `node_end`, `route`, `retrieval`, `tool_call`, `research`,
`memory`, `validation`, `guardrail`, `approval`, `token`, `answer`, `error`, `done`. The chat
endpoint streams with `stream_mode="custom"` and `subgraphs=True` (so RLM subgraph events arrive
too) and relays each event as an SSE message named by its type, ended by one `approval` event if
the thread paused, then `done`. The UI folds events into its view state with pure reducers
(`ui/activity.py`), independent of Streamlit and unit-testable.

## Consequences
- The activity panel is a live, truthful view of the run, and any API client (e.g.
  `scripts/chat_cli`) gets the same event stream without model knowledge.
- Event emission is a cross-node contract: new activity must be added as a typed event, and
  nodes become slightly noisier (kept structured and cheap).
- SSE is one-way server→client; approvals round-trip through a separate POST, which keeps the
  transport simple and proxy-friendly.
