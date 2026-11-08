---
doc_id: ARCH-PLT-002
title: Platform observability architecture
department: platform
document_type: architecture
access_level: internal
created_date: 2025-12-15
service: observability-stack
status: final
version: "2.0"
---

# Platform observability architecture

> MOCK DATA — fictional document created for the Compass assessment POC.

## Overview

All services emit metrics, logs and traces through OpenTelemetry collectors running as a
DaemonSet in each Kubernetes cluster.

## Pipelines

- Metrics: Prometheus remote-write to a long-term store, 13 months retention.
- Logs: structured JSON only, shipped to the log platform, 90 days hot, 7 years archive for
  payment services (regulatory requirement).
- Traces: 10% head sampling, 100% for errors; a correlation id travels in the
  `x-request-id` header across services.

## Alerting

- Alerts route by service ownership to on-call rotations.
- Every SEV1/SEV2 alert must link to a runbook.
- Muted channels are reviewed monthly; alerts may not stay muted for more than 14 days.

## Standards

- No customer PII in logs; account numbers masked to the last 4 digits.
