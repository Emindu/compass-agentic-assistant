"""Output guard tests: every block/warn code fires on an example, a clean cited answer has
no issues, and redaction removes secret-like values. Deterministic only - no LLM involved.
"""

from app.security.output_guard import SAFE_BLOCK_MESSAGE, check_output, redact_secrets


def codes(answer: str, **kwargs) -> list[str]:
    return [issue.code for issue in check_output(answer, **kwargs)]


SYSTEM_PROMPT_FRAGMENT = (
    "You are Compass, the internal knowledge assistant for Commercial Bank of Ceylon."
)


# --- blocks ---------------------------------------------------------------------------------


def test_financial_advice_blocked() -> None:
    issues = check_output("You should buy shares in this stock now, guaranteed returns.")
    assert "financial_advice" in [i.code for i in issues]
    assert all(i.severity == "block" for i in issues if i.code == "financial_advice")


def test_secret_leak_patterns_blocked() -> None:
    secrets = [
        "The staging key is sk-abc123def456ghi789x.",
        "Use token pcsk_9tTt3KXPaN29ZzBgf3EaP for testing.",
        "lsv2_pt_abc123def456ghi789jkl was leaked by mistake.",
        "Creds found: AKIAIOSFODNN7EXAMPLE.",
        "JWT: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2Q. Please reuse it.",
        "-----BEGIN RSA PRIVATE KEY-----",
    ]
    for answer in secrets:
        issues = check_output(answer)
        assert "secret_leak" in [i.code for i in issues], f"missed secret in: {answer[:40]}"
        assert all(i.severity == "block" for i in issues if i.code == "secret_leak")


def test_system_prompt_leak_blocked() -> None:
    answer = f"For reference, my configuration is: {SYSTEM_PROMPT_FRAGMENT} Anyway, here it is."
    issues = check_output(answer, system_prompt_fragments=[SYSTEM_PROMPT_FRAGMENT])
    assert "system_prompt_leak" in [i.code for i in issues]


def test_short_fragments_are_not_treated_as_prompt_leaks() -> None:
    answer = "the answer is 42"
    issues = check_output(answer, system_prompt_fragments=["the answer is 42"])
    assert [i.code for i in issues] == []


# --- warns ----------------------------------------------------------------------------------


def test_brand_voice_caps_shouting_warns() -> None:
    assert "brand_voice" in codes("THIS OUTAGE REPORT IS COMPLETELY UNACCEPTABLE")


def test_brand_voice_profanity_warns() -> None:
    assert "brand_voice" in codes("This migration is damn slow.")


def test_brand_voice_exclamation_spam_warns() -> None:
    assert "brand_voice" in codes("Great work team!!!")


def test_competitor_bank_mentions_do_not_warn() -> None:
    issues = check_output(
        "HSBC and Sampath Bank offer similar corporate cards with the same features."
    )
    assert [i.code for i in issues] == []
    assert "brand_voice" not in codes("Shell model? The service desk will confirm.")


def test_external_link_warns_but_bank_domains_do_not() -> None:
    assert "external_link" in codes("See [this guide](https://example.com/x) for details.")
    assert "external_link" in codes("Check [reviews](https://www.otherbank.io/a) here.")
    assert "external_link" not in codes(
        "See [runbook](https://wiki.commercialbank.internal/kb) and "
        "[intranet](https://www.commercialbank.lk/news)."
    )
    issues = check_output("See [a](https://example.com/x). See [b](https://example2.com/y).")
    assert [i.code for i in issues if i.code == "external_link"] == ["external_link"]


# --- happy path -----------------------------------------------------------------------------


def test_clean_cited_answer_has_no_issues() -> None:
    answer = (
        "According to the runbook, rotate the card switch certificate every 90 days "
        "(see [KB-123](https://wiki.commercialbank.internal/runbook)). "
        "Afterwards, verify that settlement totals match the ledger beforeclosing the ticket."
    )
    assert check_output(answer, system_prompt_fragments=[SYSTEM_PROMPT_FRAGMENT]) == []


def test_safe_block_message_exists() -> None:
    assert isinstance(SAFE_BLOCK_MESSAGE, str) and len(SAFE_BLOCK_MESSAGE) > 20


# --- redaction ------------------------------------------------------------------------------


def test_redact_secrets_replaces_all_secret_shapes() -> None:
    answer = (
        "Keys found: sk-abc123def456ghi789x and AKIAIOSFODNN7EXAMPLE "
        "and pcsk_9tTt3KXPaN29ZzBgf3EaP."
    )
    redacted = redact_secrets(answer)
    assert "[redacted]" in redacted
    assert "sk-abc123" not in redacted
    assert "AKIA" not in redacted
    assert "pcsk_" not in redacted
    assert redacted.count("[redacted]") == 3
    assert redacted.startswith("Keys found:")


def test_redact_secrets_leaves_ordinary_text_alone() -> None:
    answer = "The runbook says the prefix sk- marks staging keys; rotate these quarterly."
    assert redact_secrets(answer) == answer
