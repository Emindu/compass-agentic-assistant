"""Create the dense and sparse Pinecone indexes if they don't exist.

Both use Pinecone integrated inference, so Pinecone embeds text on upsert and query and no
second embedding vendor key is needed. Keeping dense and sparse in separate indexes (rather than
one hybrid index) lets us query them in parallel, see both raw scores, and tune the fusion
weight at query time.

Run: uv run python -m scripts.setup_pinecone
"""

import sys

from pinecone import Pinecone

from app.core.config import get_settings

# Records are upserted as {"_id", "chunk_text", ...metadata}; Pinecone embeds `chunk_text`.
TEXT_FIELD = "chunk_text"


def main() -> int:
    s = get_settings()
    if s.pinecone_api_key is None:
        print("PINECONE_API_KEY is not set (.env)", file=sys.stderr)
        return 1

    pc = Pinecone(api_key=s.pinecone_api_key.get_secret_value())
    wanted = {
        s.pinecone_dense_index: s.pinecone_dense_model,
        s.pinecone_sparse_index: s.pinecone_sparse_model,
    }

    for name, model in wanted.items():
        if pc.has_index(name):
            print(f"exists   {name}")
            continue
        pc.create_index_for_model(
            name=name,
            cloud=s.pinecone_cloud,
            region=s.pinecone_region,
            embed={"model": model, "field_map": {"text": TEXT_FIELD}},
            tags={"project": "compass"},
        )
        print(f"created  {name}  (model={model})")

    for name in wanted:
        desc = pc.describe_index(name)
        ready = desc.status.ready
        print(f"{name}: vector_type={desc.vector_type} metric={desc.metric} ready={ready}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
