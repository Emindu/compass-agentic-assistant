---
doc_id: RB-PLT-007
title: "Runbook: database connection pool exhaustion"
department: platform
document_type: runbook
access_level: internal
created_date: 2026-06-03
service: postgres-platform
status: final
version: "1.1"
---

# Runbook: database connection pool exhaustion

> MOCK DATA — fictional document created for the Compass assessment POC.

## Symptoms

- Application logs: `Connection is not available, request timed out after 30000ms`
- p99 latency rises sharply while database CPU stays normal
- PgBouncer `cl_waiting` above 0 for more than 1 minute

## Diagnosis

1. Check pool metrics in the "DB Pools" dashboard: active vs max per service.
2. Find long-running queries: `SELECT pid, now() - query_start, query FROM pg_stat_activity
   WHERE state = 'active' ORDER BY 2 DESC LIMIT 20;`
3. Check whether a batch, report or new release started just before the symptoms.

## Mitigation

- Kill clearly non-critical long queries (reports, reconciliation) after agreeing with the owner.
- Do NOT scale out application pods: each pod adds a pool and can exceed max_connections.
- If a recent release is the trigger, turn its feature flag off.

## Prevention

- Run reports and reconciliation on the read replica.
- Size pools so that pods × pool size stays under 80% of PgBouncer's server pool.
