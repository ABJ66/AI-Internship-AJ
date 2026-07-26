"""Document chunking and Pinecone ingest pipeline."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from rag.config import get_settings
from rag.pinecone_store import get_pinecone_store


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks using RecursiveCharacterTextSplitter."""

    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_text(text)


def ingest_document(
    openai_client: OpenAI,
    *,
    text: str,
    document_id: str,
    source: str | None = None,
) -> int:
    """
    Chunk, embed, and upsert a document into Pinecone.
    Returns the number of chunks indexed.
    """

    chunks = chunk_text(text)
    if not chunks:
        return 0

    store = get_pinecone_store(openai_client)
    source_value = source or ""
    vector_ids = [f"{document_id}-{index}" for index in range(len(chunks))]
    metadata = [
        {
            "document_id": document_id,
            "chunk_index": index,
            "source": source_value,
        }
        for index in range(len(chunks))
    ]
    return store.upsert_texts(chunks, ids=vector_ids, metadata=metadata)
