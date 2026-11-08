# ADR 0002: Hybrid retrieval on Pinecone integrated inference

## Context
Bank questions mix paraphrases ("why did card payments fail because of TLS") and exact terms
(`HikariPool`, `INC-2026-0214`). One signal cannot serve both. The bank needs access control the
prompt cannot influence, and the demo must survive a Pinecone outage. Adding an embedding vendor
would add a key, latency and cost.

## Decision
Use two Pinecone indexes with integrated (hosted) embedding models: `compass-dense`
(llama-text-embed-v2) and `compass-sparse` (pinecone-sparse-english-v0), one namespace per
department, queried in parallel. Fuse by max-normalising each list and taking
`fused = alpha * dense + (1 - alpha) * sparse` (alpha = 0.6), keeping at most 2 chunks per
document. Access control is a metadata filter (`access_rank <= clearance`, from the JWT) as the
first clause. A local `rank_bm25` index over the same chunks answers when indexes fail, with the
identical filter set.

## Consequences
- No embedding-vendor key; both raw scores stay visible in the UI; one explainable fusion knob.
- Namespaces allow scoped queries and per-department re-indexing, but are not a security boundary;
  the `access_rank` filter is.
- Degradation (single index, BM25) is flagged to the user and never widens access. Two indexes
  mean two things to provision and monitor (`scripts/setup_pinecone` does both).
