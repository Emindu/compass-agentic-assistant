"""Async Pinecone access shared by the dense and sparse searchers.

Why one store object: async index clients hold HTTP connection pools; creating them per query
would add a TLS handshake to every search. The store opens each index once, searches all
requested namespaces concurrently, and enforces one timeout for the whole fan-out so a slow
namespace cannot stall the request.
"""

import asyncio
from functools import lru_cache
from typing import Any

from pinecone import Pinecone

from app.core.config import get_settings
from app.core.errors import VectorStoreError
from app.core.logging import get_logger
from app.retrieval.models import RETURN_FIELDS

log = get_logger(__name__)


class PineconeStore:
    def __init__(self) -> None:
        s = get_settings()
        if s.pinecone_api_key is None:
            raise VectorStoreError("PINECONE_API_KEY is not configured")
        self._pc = Pinecone(api_key=s.pinecone_api_key.get_secret_value())
        self._indexes: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def _index(self, name: str) -> Any:
        async with self._lock:
            if name not in self._indexes:
                host = await asyncio.to_thread(lambda: self._pc.describe_index(name).host)
                self._indexes[name] = self._pc.IndexAsyncio(host=host)
            return self._indexes[name]

    async def search(
        self, index_name: str, text: str, namespaces: list[str], top_k: int, flt: dict
    ) -> list[tuple[str, float, dict]]:
        """Search `namespaces` in parallel; returns (id, score, fields) across all of them.

        Scores from one index are comparable across namespaces (same model), so the caller can
        merge them directly.
        """
        s = get_settings()
        if s.fault_pinecone:
            raise VectorStoreError("FAULT_PINECONE=true: simulated Pinecone outage")
        index = await self._index(index_name)

        async def one(namespace: str):
            return await index.search(
                namespace=namespace,
                query={"inputs": {"text": text}, "top_k": top_k, "filter": flt},
                fields=RETURN_FIELDS,
            )

        try:
            async with asyncio.timeout(s.pinecone_timeout_s):
                responses = await asyncio.gather(*(one(ns) for ns in namespaces))
        except TimeoutError as exc:
            raise VectorStoreError(f"{index_name} timed out after {s.pinecone_timeout_s}s") from exc
        except Exception as exc:  # SDK raises several types; callers only need one
            raise VectorStoreError(f"{index_name} search failed: {exc!r}") from exc

        return [(h.id, float(h.score), dict(h.fields)) for r in responses for h in r.result.hits]

    async def aclose(self) -> None:
        for index in self._indexes.values():
            await index.close()
        self._indexes.clear()


@lru_cache
def get_store() -> PineconeStore:
    return PineconeStore()
