"""Deterministic output guard: checks on the drafted answer before it reaches the user.

Why check the output at all: the answer is assembled from untrusted retrieved text under an
LLM's control, so a poisoned document can resurface in the draft. The input guard never saw
that text - it only checked the user's question. These checks are pure regex/string work so
they run on every answer with no latency or cost, and they are the last line before display.
"""

import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel

SAFE_BLOCK_MESSAGE = (
    "I'm sorry, I can't share that response as drafted. Please rephrase the question, "
    "or contact the IT service desk if you believe this is a mistake."
)


class OutputIssue(BaseModel):
    """One finding on the drafted answer. ``block`` must never be shown to the user;
    ``warn`` may be shown or logged for editorial review."""

    code: str
    message: str
    severity: Literal["block", "warn"]


# --- block-level detectors ---------------------------------------------------------------

# Personal investment advice: a compliance line for a bank assistant, not a style nit.
_FINANCIAL_ADVICE = re.compile(
    r"\byou should (?:buy|sell|invest)\b"
    r"|\bI (?:strongly |highly )?recommend (?:investing|buying|selling)\b"
    r"|\bguaranteed returns?\b"
    r"|\bbuy (?:shares|stocks?)\b"
    r"|\bstock tips?\b"
    r"|\bshould i (?:invest|buy stocks?)\b",
    re.IGNORECASE,
)

# Secret-looking strings ordered from most specific anchors (prefix alphas require long tails
# like sk-..., so ordinary prose cannot accident-shape into them).
_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bpcsk_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\blsv2_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.{1}eyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]+"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)

# --- warn-level detectors ----------------------------------------------------------------

# Small, unambiguous list only - profanity filters notoriously over-match (Scunthorpe).
_PROFANITY = re.compile(r"\b(?:damn|hell|bloody|crap|bugger)\b", re.IGNORECASE)
# Three or more consecutive 4+-letter ALL-CAPS words reads as shouting ("PLEASE FIX THIS NOW").
_CAPS_SHOUT = re.compile(r"\b[A-Z]{4,}(?:\s+[A-Z]{4,}){2,}")
# Exclamation spam is the classic tell of a rushed, non-brand answer.
_EXCLAMATION_SPAM = re.compile(r"!{3,}")
# Markdown links are clickable, so they get surfacing treatment: bank domains only.
_MD_LINK = re.compile(r"\[[^\]]+\]\(\s*(https?://[^)\s]+)")


def _is_bank_host(host: str) -> bool:
    """Links to the bank's own domains are fine; anything else is an untracked external link."""
    return host == "commercialbank.lk" or any(
        host.endswith(suffix) for suffix in (".commercialbank.lk", ".commercialbank.internal")
    )


def check_output(
    answer: str, *, system_prompt_fragments: list[str] | None = None
) -> list[OutputIssue]:
    """Return every issue found in the drafted answer; ``[]`` means it is safe to display.

    Blocks are listed before warns so callers checking ``issues[0]`` see the reason the
    answer must not be shown. Fragments shorter than 30 characters are skipped: a short
    fragment like "the answer" appears in ordinary text and would block at random.
    """
    blocks: list[OutputIssue] = []
    warns: list[OutputIssue] = []

    if _FINANCIAL_ADVICE.search(answer):
        blocks.append(
            OutputIssue(
                code="financial_advice",
                message="The answer contains personal investment advice.",
                severity="block",
            )
        )

    if any(pattern.search(answer) for pattern in _TOKEN_PATTERNS):
        blocks.append(
            OutputIssue(
                code="secret_leak",
                message="The answer contains a string that looks like a credential or API key.",
                severity="block",
            )
        )

    for fragment in system_prompt_fragments or []:
        if len(fragment) >= 30 and fragment in answer:
            blocks.append(
                OutputIssue(
                    code="system_prompt_leak",
                    message="The answer contains verbatim system-prompt text.",
                    severity="block",
                )
            )
            break

    if _PROFANITY.search(answer) or _CAPS_SHOUT.search(answer) or _EXCLAMATION_SPAM.search(answer):
        warns.append(
            OutputIssue(
                code="brand_voice",
                message="The answer uses language outside the brand voice (shouting/profanity).",
                severity="warn",
            )
        )

    for url in _MD_LINK.findall(answer):
        host = urlparse(url).hostname or ""
        if not _is_bank_host(host):
            warns.append(
                OutputIssue(
                    code="external_link",
                    message=f"The answer links to a non-bank domain ({host}).",
                    severity="warn",
                )
            )
            break

    return blocks + warns


def redact_secrets(answer: str) -> str:
    """Replace secret-looking matches with ``[redacted]``.

    Why redact instead of delete: keeping the sentence shape makes the log line remain
    readable and shows a secret *was* there, without preserving the value itself.
    """
    redacted = answer
    for pattern in _TOKEN_PATTERNS:
        redacted = pattern.sub("[redacted]", redacted)
    return redacted
