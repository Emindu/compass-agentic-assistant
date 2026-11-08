"""Sanitising retrieved chunk text: retrieved documents are untrusted data.

Why this exists: the vector store contains documents written by hundreds of employees (and
occasionally pasted web content). A poisoned page can carry instructions aimed at the model.
Wrapping documents in XML tags helps, but only if the content cannot escape the wrapper or
claim authority inside it. Everything here is pure string work: sanitising runs on every
chunk of every retrieval, so it must stay fast and offline (no model call in the hot path).
"""

import html
import re
from dataclasses import dataclass, field

# Instruction-like text that can appear inside a document and poison the model reading it.
# Line-anchored forms use re.MULTILINE so they hit "### system" at the start of any line.
INJECTION_MARKERS: dict[str, re.Pattern[str]] = {
    "instruction_override": re.compile(
        r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\s+instructions?",
        re.IGNORECASE,
    ),
    "system_instruction": re.compile(
        r"system\s+instructions?\s+for\s+ai\s+assistants?", re.IGNORECASE
    ),
    "you_must_now": re.compile(r"\byou must now\b", re.IGNORECASE),
    "assistant_prefix": re.compile(r"(?im)^\s*assistant\s*:"),
    "im_start": re.compile(r"<\|im_start\|>"),
    "markdown_system": re.compile(r"(?im)^#{1,6}\s*(?:system|assistant)\b"),
    "begin_system_prompt": re.compile(r"begin\s+system\s+prompt", re.IGNORECASE),
    # Hidden text: invisible in rendered docs, fully visible to the model.
    "html_comment": re.compile(r"<!--[\s\S]*?-->"),
    # External markdown images: the model cannot see them, but their URLs are a beacon that
    # something (or someone downstream with logs) can hit. Classic exfiltration channel.
    "external_image": re.compile(r"!\[[^\]]*\]\(\s*https?://[^)]*\)"),
}

# Markers that are deleted outright rather than line-replaced: they hide inside otherwise
# legitimate lines, so replacing the whole line would destroy innocent text.
_DELETE_MARKERS = ("html_comment", "external_image")

# Tags that could close or forge the <document> wrapper the prompt builds around chunks.
# Only the "<" is neutralised so the text stays readable and diffs stay small.
_DOCUMENT_TAG = re.compile(r"</?document(?=[>\s])", re.IGNORECASE)

# Control characters and zero-width chars: keep \t and \n (real document structure),
# drop the rest, which is either invisible padding or a smuggling trick like zero-width splits.
_INVISIBLE_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]"
)

REMOVED_NOTE = "[removed: instruction-like text in source document]"


@dataclass
class SanitizeResult:
    """Sanitised text plus what happened, so callers can surface a notice without ever
    re-rendering the removed content."""

    text: str
    flagged: bool
    findings: list[str] = field(default_factory=list)


def sanitize_chunk_text(text: str) -> SanitizeResult:
    """Clean one retrieved chunk: invisible chars, HTML comments, wrapper-breaking tags,
    and instruction-like lines. Everything else is kept byte-identical - sanitising must
    never paraphrase the bank's documents, which are cited content of record.
    """
    findings: list[str] = []

    cleaned = _INVISIBLE_CHARS.sub("", text)
    if cleaned != text:
        findings.append("invisible_or_control_chars")

    for name in _DELETE_MARKERS:
        pattern = INJECTION_MARKERS[name]
        new = pattern.sub("", cleaned)
        if new != cleaned:
            findings.append(name)
            cleaned = new

    tagged = _DOCUMENT_TAG.sub(lambda m: m.group(0).replace("<", "‹"), cleaned)
    if tagged != cleaned:
        findings.append("document_tag_neutralised")
        cleaned = tagged

    # Line granularity (rather than replacing only the matched span) because one line of a
    # document is roughly one statement - after an injected instruction starts mid-line, the
    # rest of that line is attacker-controlled context too.
    kept_lines: list[str] = []
    for line in cleaned.split("\n"):
        hits = [
            name
            for name, pattern in INJECTION_MARKERS.items()
            if name not in _DELETE_MARKERS and pattern.search(line)
        ]
        if hits:
            for name in hits:
                if name not in findings:
                    findings.append(name)
            kept_lines.append(REMOVED_NOTE)
        else:
            kept_lines.append(line)

    return SanitizeResult(text="\n".join(kept_lines), flagged=bool(findings), findings=findings)


def wrap_untrusted(chunk_id: str, title: str, status: str, date: str, text: str) -> str:
    """Wrap (already-sanitised) chunk text in a <document> element for the prompt.

    Why explicit attributes: the metadata tells the model what status/date the source had,
    so it can weigh drafts differently from approved runbooks. Why html.escape on the
    attributes: document titles are user-entered text, and a title containing `">` would
    otherwise break out of the wrapper exactly like a forged closing tag.
    """
    escaped = {
        key: html.escape(value, quote=True)
        for key, value in (("id", chunk_id), ("title", title), ("status", status), ("date", date))
    }
    return (
        f'<document id="{escaped["id"]}" title="{escaped["title"]}" '
        f'status="{escaped["status"]}" date="{escaped["date"]}">\n{text}\n</document>'
    )
