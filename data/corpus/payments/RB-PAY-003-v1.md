---
doc_id: RB-PAY-003-V1
title: "Runbook: rotate payment connector certificates (v1, superseded)"
department: payments
document_type: runbook
access_level: internal
created_date: 2025-03-02
service: card-switch-connector
status: superseded
version: "1.0"
---

# Runbook: rotate payment connector certificates (v1)

> MOCK DATA — fictional document created for the Compass assessment POC.
> SUPERSEDED by RB-PAY-003 v2 (February 2026). Kept for audit history.

## When to use

When a connector certificate is close to expiry.

## Steps

1. Request a new certificate from the PKI team by email; allow 5 working days.
2. Copy the new keystore file to each connector server with `scp`.
3. Restart the connector service during the Sunday 02:00 maintenance window.
4. Confirm the service is running.

## Notes

- Certificate expiry dates are tracked in the payments team calendar.
