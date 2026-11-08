---
doc_id: RB-PAY-003-V2
title: "Runbook: rotate payment connector certificates (v2)"
department: payments
document_type: runbook
access_level: internal
created_date: 2026-02-20
service: card-switch-connector
status: final
version: "2.0"
---

# Runbook: rotate payment connector certificates (v2)

> MOCK DATA — fictional document created for the Compass assessment POC.
> Replaces v1. Written after INC-2025-1207.

## When to use

- The certificate expiry monitor alerts at 30, 14 or 7 days before expiry, or
- A partner announces a CA or certificate change.

## Steps

1. Raise a renewal request in the PKI portal (automated issuance, usually under 1 hour).
2. The new certificate is written to the secrets vault at `pki/payments/<connector>`.
3. Trigger a rolling restart: `kubectl rollout restart deploy/<connector> -n payments`.
   Pods load the keystore from the vault at start-up; do not copy files by hand.
4. Verify: `handshake_failure` count is zero and authorisation success rate is above 99.5%.
5. Update the certificate inventory entry with the new expiry date.

## Partner trust chain changes

When a partner rotates its server certificate or intermediate CA, update the CA bundle in
PKI and rebuild connector images. Never disable certificate validation as a workaround.

## Rollback

`kubectl rollout undo deploy/<connector> -n payments` restores the previous keystore version.
