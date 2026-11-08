"""Print attributed hybrid search results with dense, sparse and fused scores.

Run:
  uv run python -m scripts.search_demo "why did card payments fail because of TLS"
  uv run python -m scripts.search_demo "Q3 risk review compensation" --role analyst
  uv run python -m scripts.search_demo "HikariPool" --fault-pinecone
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.retrieval.hybrid import hybrid_search
from app.retrieval.models import RetrievalQuery
from app.retrieval.pinecone_store import get_store
from app.security.rbac import Role, clearance_for


def fmt(score: float | None) -> str:
    return "   -  " if score is None else f"{score:6.3f}"


async def main(args: argparse.Namespace) -> None:
    load_dotenv()
    if args.fault_pinecone:
        os.environ["FAULT_PINECONE"] = "true"
        get_settings.cache_clear()
    configure_logging("WARNING")
    q = RetrievalQuery(
        text=args.query,
        clearance=clearance_for(Role(args.role)),
        top_k=args.top_k,
        alpha=args.alpha,
    )
    try:
        result = await hybrid_search(q)
    finally:
        if not args.fault_pinecone:
            await get_store().aclose()

    alpha = get_settings().hybrid_alpha if args.alpha is None else args.alpha
    print(f'\nQuery: "{q.text}"  role={args.role} (clearance={q.clearance.name})  alpha={alpha}')
    print(f"Timings ms: {result.timings_ms}  degraded={result.degraded}")
    for note in result.notes:
        print(f"  note: {note}")
    print(f"\n{'#':>2}  {'fused':>6} {'dense':>6} {'sparse':>6}  {'access':12} citation")
    for i, c in enumerate(result.chunks, 1):
        print(
            f"{i:>2}  {c.fused_score:6.3f} {fmt(c.dense_score)} {fmt(c.sparse_score)}  "
            f"{c.access_level:12} {c.citation}  [{c.source}]"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--role", default="viewer", choices=[r.value for r in Role])
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--fault-pinecone", action="store_true")
    asyncio.run(main(p.parse_args()))
