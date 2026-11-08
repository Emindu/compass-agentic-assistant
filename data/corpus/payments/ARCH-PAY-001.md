---
doc_id: ARCH-PAY-001
title: Payments platform architecture overview
department: payments
document_type: architecture
access_level: internal
created_date: 2026-01-10
service: payments-gateway
status: final
version: "3.2"
---

# Payments platform architecture overview

> MOCK DATA — fictional document created for the Compass assessment POC.

## Purpose

Describes how customer payments flow from digital channels to external switches and the
core ledger. Audience: engineers joining the payments tribe.

## Components

- **payments-gateway**: entry point for all payment requests from mobile, internet banking and
  branch systems. Validates, applies limits, routes to an adapter.
- **payments-ledger**: posts debits and credits; owns the payments database (PostgreSQL 15,
  primary + one read replica, max_connections 300). Accessed through PgBouncer since mid-2026.
- **card-switch-connector**: ISO 8583 connection to the international card switch over mutual TLS.
- **cefts-adapter**: connection to the national instant payment switch (CEFTS) over mutual TLS.
- **biller-gateway**: REST integration with the external biller aggregator for utility bills.
- **swift-gateway**: outward and inward SWIFT messaging.

## Request flow

1. Channel calls payments-gateway with an idempotency key (mandatory from release 4.12).
2. Gateway checks customer limits and fraud score (synchronous call to fraud-service, 150 ms budget).
3. Gateway routes to the adapter for the payment rail.
4. On adapter success, payments-ledger posts the transaction and publishes `payment.posted`.

## Resilience standards

- All outbound calls use the shared resilient client: timeouts, exponential backoff with
  jitter, retry budget of 10%, circuit breaker.
- Every external mutual-TLS connection must be registered in the certificate inventory.

## Ownership

Payments Platform team; on-call rotation "payments-primary".
