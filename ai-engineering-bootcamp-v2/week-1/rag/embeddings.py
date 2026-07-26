"""OpenAI embedding helpers — same model at ingest and query time."""

from openai import OpenAI

from rag.config import get_settings

# text-embedding-3-small defaults to 1536 dimensions.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSION = 1536


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed one or more text chunks with the configured embedding model."""

    if not texts:
        return []

    settings = get_settings()
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_query(client: OpenAI, query: str) -> list[float]:
    """Embed a single search query (same model as document ingest)."""

    return embed_texts(client, [query])[0]
