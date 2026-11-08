---
doc_id: POL-SEC-004
title: Privileged access and break-glass procedure
department: security
document_type: policy
access_level: restricted
created_date: 2026-03-15
service: enterprise
status: final
version: "2.1"
---

# Privileged access and break-glass procedure

> MOCK DATA — fictional document created for the Compass assessment POC.
> RESTRICTED: named administrators only.

## Break-glass accounts

- Two break-glass accounts exist per production environment, stored in the hardware-backed vault.
- Use requires two administrators (four-eyes) and an open SEV1 incident number.
- Every use triggers an alert to the CISO and is reviewed within 24 hours.

## Privileged sessions

- All privileged sessions are recorded by the PAM gateway and kept for 2 years.
- Direct SSH to production hosts is blocked; access only through the PAM gateway.

## User access resets

- Bulk access resets (more than 10 users) need CISO approval.
- The admin tool `reset_user_access` must be run with a change ticket reference.
