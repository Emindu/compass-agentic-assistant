---
doc_id: SPEC-RET-005
title: Mobile app card controls — product specification
department: retail
document_type: spec
access_level: internal
created_date: 2026-05-12
service: mobile-app
status: final
version: "1.2"
---

# Mobile app card controls — product specification

> MOCK DATA — fictional document created for the Compass assessment POC.

## Overview

Customers can manage their debit and credit cards from the mobile app without calling the bank.

## Features

- Temporarily lock and unlock a card (takes effect within 5 seconds)
- Turn on or off: e-commerce, international, ATM and contactless payments
- Set a per-transaction e-commerce limit (LKR 1,000 to LKR 500,000)
- Report a card lost or stolen and order a replacement

## Rules

- Unlocking a card or raising limits requires biometric or PIN re-authentication.
- Changes are sent to card-switch-connector as card-control updates; if the switch is
  unavailable, the change is queued and the customer sees "pending".

## Out of scope

Changing the card PIN (handled in a separate release).
