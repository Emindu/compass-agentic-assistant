"""RLM planner: the code model writes a Python search plan instead of reading documents.

This is the core Recursive Language Model idea: rather than stuffing whole documents into one
context window, the model writes a small program that explores the collection (metadata first),
filters it, pulls only targeted sections, and splits the evidence into batches. The batches are
then analysed separately by sub-calls and aggregated. The program runs in the sandbox; its
primitives run in the parent with the user's clearance.
"""

import re
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import ainvoke_llm
from app.rlm.primitives import PRIMITIVE_DOCS
from app.rlm.sandbox import ALLOWED_IMPORTS

MAX_BATCHES = 8
MAX_CHUNKS_PER_BATCH = 8

PLANNER_SYSTEM = """You are the research planner of Compass, Commercial Bank's internal knowledge
assistant. Write a short Python program (at most 60 lines) that gathers the evidence needed to
answer the question. Do not answer the question yourself.

Available functions (already defined; they enforce the user's access rights):
{primitives}

Rules:
- Explore metadata with list_documents() before reading content; then pull only the sections
  you need with get_sections() or search(). Never fetch more than is needed.
- Exclude drafts and superseded documents unless the question asks for them.
- Select candidates with metadata filters (document_type, department, dates, status), NOT by
  keywords in titles: titles rarely use the user's words (a card-switch TLS outage is still a
  payment failure). Read the relevant sections of every candidate; batch analysis judges
  relevance. Use search() only when metadata cannot narrow the candidates.
- "Last year" / "past 12 months" means created_date on or after {year_ago}. Today is {today}.
- Allowed imports: {imports}. No other imports, no file or network access, no names that start
  with "__", no attributes that start with "_", no str.format() (use f-strings), no class
  definitions, no eval/exec/getattr/open.
- Finish by assigning a dict to the variable `result`:
    result = {{
        "documents": [doc ids you selected],
        "batches": [[chunk_id, ...], ...],   # evidence chunk ids, grouped for parallel analysis
        "task": "what to extract from each batch",
        "data": {{...}},                     # optional computed facts (counts, groupings)
    }}
  Use at most {max_batches} batches of at most {max_chunks} chunk ids, grouping chunks of the
  same document together. For pure counting questions, "batches" may be empty and "data" holds
  the answer.
- Return only the code in one ```python block.

Example (different question: "What went wrong in platform incidents since March, and what
actions were agreed?"):
```python
docs = list_documents(document_type="incident", department="platform", date_from="2026-03-01",
                      status="final")
batches = []
for d in docs:
    ids = [s["chunk_id"] for s in get_sections(d["doc_id"],
                                               sections=["Summary", "Root cause", "Action items"])]
    if ids:
        batches.append(ids)
result = {{
    "documents": [d["doc_id"] for d in docs],
    "batches": batches,
    "task": "For each incident: what failed, the root cause, and the agreed actions.",
    "data": {{"incidents": len(docs)}},
}}
```
Note how it selects by metadata only and reads the same named sections of every candidate.
Incident sections: Summary, Timeline, Root cause, Resolution, Action items."""

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Take the first fenced block; fall back to the raw text if the model omitted fences."""
    match = _CODE_BLOCK.search(text)
    return (match.group(1) if match else text).strip()


async def write_plan(question: str, today: date, feedback: str | None = None) -> str:
    year_ago = today.replace(year=today.year - 1)
    system = PLANNER_SYSTEM.format(
        primitives=PRIMITIVE_DOCS,
        today=today.isoformat(),
        year_ago=year_ago.isoformat(),
        imports=", ".join(sorted(ALLOWED_IMPORTS)),
        max_batches=MAX_BATCHES,
        max_chunks=MAX_CHUNKS_PER_BATCH,
    )
    # Kept next to the question because the code model weighs it more than system-prompt rules:
    # titles rarely contain the user's words, so keyword filters silently drop relevant docs.
    prompt = (
        f"Question: {question}\n\nSelect candidates with list_documents() filters only. Do NOT "
        "filter its results with Python conditions on title, service or text; read the "
        "relevant sections of every candidate instead."
    )
    if feedback:
        prompt += f"\n\nYour previous plan failed: {feedback}\nWrite a corrected plan."
    reply = await ainvoke_llm(
        "code", [SystemMessage(system), HumanMessage(prompt)], config={"run_name": "rlm_planner"}
    )
    return extract_code(reply.text)
