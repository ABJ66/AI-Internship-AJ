"""Environment-based configuration for RAG and Pinecone."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RagSettings:
    openai_api_key: str | None
    pinecone_api_key: str | None
    pinecone_index_name: str | None
    pinecone_host: str | None
    embedding_model: str
    embedding_dimension: int
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int


def get_settings() -> RagSettings:
    return RagSettings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME"),
        pinecone_host=os.getenv("PINECONE_HOST"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dimension=int(os.getenv("OPENAI_EMBEDDING_DIMENSION", "1536")),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "100")),
        retrieval_top_k=int(os.getenv("RAG_RETRIEVAL_TOP_K", "5")),
    )


def pinecone_configured(settings: RagSettings | None = None) -> bool:
    cfg = settings or get_settings()
    return bool(cfg.pinecone_api_key and (cfg.pinecone_index_name or cfg.pinecone_host))
