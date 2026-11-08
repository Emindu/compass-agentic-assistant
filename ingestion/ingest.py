"""Chunk the corpus and upsert it into the dense and sparse Pinecone indexes.

One namespace per department: queries can target the departments that matter, and a
department's data can be dropped or re-indexed without touching the others. The same records go
to both indexes; Pinecone's integrated inference embeds `chunk_text` with the dense or sparse
model of each index. Both indexes are written concurrently, and record ids are deterministic,
so re-running is idempotent.

Run: uv run python -m ingestion.ingest [--reset]
"""

import argparse
import asyncio
from collections import defaultdict

from pinecone import Pinecone
from pinecone.exceptions import NotFoundException

from app.core.config import get_settings
from ingestion.chunk import chunk_corpus
from ingestion.models import Chunk

BATCH_SIZE = 96  # Pinecone's maximum records per upsert for integrated-inference indexes


async def _upsert_index(pc: Pinecone, index_name: str, by_ns: dict[str, list[dict]], reset: bool):
    host = pc.describe_index(index_name).host
    async with pc.IndexAsyncio(host=host) as index:
        for namespace, records in by_ns.items():
            if reset:
                try:
                    await index.delete(delete_all=True, namespace=namespace)
                except NotFoundException:
                    pass  # namespace doesn't exist yet
            for i in range(0, len(records), BATCH_SIZE):
                await index.upsert_records(namespace=namespace, records=records[i : i + BATCH_SIZE])
            print(f"  {index_name:16} {namespace:10} {len(records):3} records")


def _group(chunks: list[Chunk]) -> dict[str, list[dict]]:
    by_ns: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_ns[c.meta.department].append(c.to_record())
    return dict(by_ns)


async def main(reset: bool) -> None:
    s = get_settings()
    if s.pinecone_api_key is None:
        raise SystemExit("PINECONE_API_KEY is not set (.env)")
    chunks = chunk_corpus()
    by_ns = _group(chunks)
    docs = len({c.meta.doc_id for c in chunks})
    print(f"Corpus: {docs} documents -> {len(chunks)} chunks in {len(by_ns)} namespaces")

    pc = Pinecone(api_key=s.pinecone_api_key.get_secret_value())
    await asyncio.gather(
        _upsert_index(pc, s.pinecone_dense_index, by_ns, reset),
        _upsert_index(pc, s.pinecone_sparse_index, by_ns, reset),
    )

    await asyncio.sleep(5)  # serverless indexes are eventually consistent; let stats catch up
    for name in (s.pinecone_dense_index, s.pinecone_sparse_index):
        stats = pc.Index(name=name).describe_index_stats()
        counts = {ns: v.vector_count for ns, v in stats.namespaces.items()}
        print(f"{name}: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reset", action="store_true", help="delete namespaces before upserting")
    asyncio.run(main(parser.parse_args().reset))
