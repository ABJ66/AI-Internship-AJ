"""Pinecone vector store — ingest, search, and health checks."""

from __future__ import annotations

import uuid
from typing import Any

from openai import OpenAI
from pinecone import Pinecone

from rag.config import get_settings, pinecone_configured
from rag.embeddings import embed_query, embed_texts

_store: "PineconeStore | None" = None


class PineconeStore:
    """Thin wrapper around Pinecone index + OpenAI embeddings."""

    def __init__(self, openai_client: OpenAI) -> None:
        settings = get_settings()
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")
        if not settings.pinecone_index_name and not settings.pinecone_host:
            raise RuntimeError("Set PINECONE_INDEX_NAME or PINECONE_HOST")

        self.settings = settings
        self.openai_client = openai_client
        pc = Pinecone(api_key=settings.pinecone_api_key)
        if settings.pinecone_host:
            self.index = pc.Index(host=settings.pinecone_host)
        else:
            self.index = pc.Index(settings.pinecone_index_name)

    def upsert_texts(
        self,
        texts: list[str],
        *,
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Embed texts and upsert vectors into Pinecone."""

        if not texts:
            return 0

        vectors = embed_texts(self.openai_client, texts)
        vector_ids = ids or [str(uuid.uuid4()) for _ in texts]
        meta_list = metadata or [{} for _ in texts]

        records = [
            {
                "id": vector_id,
                "values": values,
                "metadata": {**meta, "text": text},
            }
            for vector_id, values, meta, text in zip(vector_ids, vectors, meta_list, texts)
        ]
        self.index.upsert(vectors=records)
        return len(records)

    def search_similar(self, query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
        """Embed a query and return the most similar stored chunks."""

        query_vector = embed_query(self.openai_client, query)
        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )

        matches = []
        for match in response.matches or []:
            matches.append(
                {
                    "id": match.id,
                    "score": match.score,
                    "text": (match.metadata or {}).get("text"),
                    "metadata": match.metadata or {},
                }
            )
        return matches

    def describe_stats(self) -> dict[str, Any]:
        """Return Pinecone index statistics."""

        stats = self.index.describe_index_stats()
        return {
            "dimension": stats.dimension,
            "total_vector_count": stats.total_vector_count,
            "namespaces": {
                name: {"vector_count": ns.vector_count}
                for name, ns in (stats.namespaces or {}).items()
            },
        }


def retrieve_chunks(
    openai_client: OpenAI,
    query: str,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Embed a question and return top matching chunks from Pinecone.
    Does not call the LLM — retrieval only.
    """

    settings = get_settings()
    store = get_pinecone_store(openai_client)
    matches = store.search_similar(query, top_k=top_k)

    chunks = []
    for match in matches:
        metadata = match.get("metadata") or {}
        chunks.append(
            {
                "id": match["id"],
                "score": match["score"],
                "document_id": metadata.get("document_id"),
                "chunk_index": metadata.get("chunk_index"),
                "source": metadata.get("source"),
                "text": match.get("text"),
            }
        )

    return {
        "query": query,
        "embedding_model": settings.embedding_model,
        "top_k": top_k,
        "chunks": chunks,
    }


def get_pinecone_store(openai_client: OpenAI) -> PineconeStore:
    """Return a shared PineconeStore instance for this process."""

    global _store
    if _store is None:
        _store = PineconeStore(openai_client)
    return _store


def pinecone_health(openai_client: OpenAI) -> dict[str, Any]:
    """
    Health/debug check for Pinecone connectivity.
    Safe to call from an HTTP endpoint — never returns secret values.
    """

    settings = get_settings()

    if not pinecone_configured(settings):
        missing = []
        if not settings.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        if not settings.pinecone_index_name and not settings.pinecone_host:
            missing.append("PINECONE_INDEX_NAME or PINECONE_HOST")
        if not settings.openai_api_key:
            missing.append("OPENAI_API_KEY")

        return {
            "status": "not_configured",
            "reachable": False,
            "missing_env": missing,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
        }

    try:
        store = get_pinecone_store(openai_client)
        stats = store.describe_stats()
        return {
            "status": "ok",
            "reachable": True,
            "index_name": settings.pinecone_index_name,
            "index_host": settings.pinecone_host,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "index_stats": stats,
        }
    except Exception as exc:
        return {
            "status": "error",
            "reachable": False,
            "index_name": settings.pinecone_index_name,
            "index_host": settings.pinecone_host,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "error": str(exc),
        }
