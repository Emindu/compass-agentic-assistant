---
doc_id: SPEC-PAY-010
title: Instant transfers via CEFTS — product specification
department: payments
document_type: spec
access_level: internal
created_date: 2025-11-18
service: cefts-adapter
status: final
version: "1.4"
---

# Instant transfers via CEFTS — product specification

> MOCK DATA — fictional document created for the Compass assessment POC.

## Overview

Customers can send money instantly to accounts at other local banks, 24/7, through the
national CEFTS switch.

## Limits

- Per transaction: LKR 5,000,000 (personal), LKR 25,000,000 (business)
- Daily: LKR 10,000,000 (personal), configurable for business customers
- Fee: LKR 30 per transfer on mobile, LKR 50 at branches

## Functional requirements

- FR-1: Beneficiary name confirmation before submission
- FR-2: Funds credited to the beneficiary within 15 seconds in 99% of cases
- FR-3: If CEFTS is unavailable for over 10 minutes, offer next-day SLIPS transfer instead
- FR-4: Every request carries an idempotency key; duplicates are rejected, not re-sent

## Non-functional requirements

- Availability target 99.9% per month, excluding switch-side outages
- Duplicate debits: zero tolerance; any duplicate must be reversed within 1 business day
