"""Week 1 live demo — five stages in one file, built up live in class."""

import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from openai import APIError, OpenAI
from pydantic import BaseModel, Field, ValidationError

from rag.config import get_settings, pinecone_configured
from rag.ingest import ingest_document
from rag.prompts import build_grounding_prompt
from rag.pinecone_store import pinecone_health, retrieve_chunks

# Load .env from this folder so the key is found regardless of shell working directory.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# Reuse one client so TLS handshakes are not repeated on every request.
app = FastAPI()
client = OpenAI()  # Reads OPENAI_API_KEY from the environment; never hardcode keys.

# Stage 4 default — strong general model; swap at request time for the live demo.
DEFAULT_MODEL = "gpt-4o"

# Stage 5 — per-1K-token input/output USD (derived from OpenAI list prices).
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str
    force_bad: bool = False  # Stage 3 demo knob — first attempt breaks schema on purpose.
    model: str | None = Field(
        default=None,
        description="OpenAI model override (e.g. gpt-4o-mini). Leave empty for default.",
    )


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float
    retrieved_chunk_ids: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    """Upload raw text to chunk, embed, and store in Pinecone."""

    text: str
    document_id: str = Field(min_length=1)
    source: str | None = None  # optional filename or URL


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    status: str


class RetrieveChunk(BaseModel):
    id: str
    score: float
    document_id: str | None = None
    chunk_index: int | None = None
    source: str | None = None
    text: str | None = None


class RetrieveResponse(BaseModel):
    query: str
    embedding_model: str
    top_k: int
    chunks: list[RetrieveChunk]


def resolve_model(model: str | None) -> str:
    """Use default model when client omits model or sends Swagger's placeholder value."""

    if not model:
        return DEFAULT_MODEL
    cleaned = model.strip()
    if cleaned.lower() in {"string", "null", "none", ""}:
        return DEFAULT_MODEL
    return cleaned


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost."""

    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    input_per_1k, output_per_1k = prices
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


def call_model_structured(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 2 center: OpenAI structured output forces exactly the Answer schema.
    Returns parsed answer plus token counts from billing metadata.
    """

    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": question}],
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return parsed, total, prompt_tokens, completion_tokens


def call_model_unsafe(question: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 3 demo path: free-form JSON call, then validate locally.
    The bad instruction makes confidence a string so Pydantic rejects it reliably.
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Reply with ONLY a JSON object using keys answer, confidence, sources_needed. "
                    "Set confidence to the string 'very high' (not a number)."
                ),
            }
        ],
    )

    raw = completion.choices[0].message.content or ""
    # Guardrail: refuse malformed output instead of passing it through to clients.
    answer = Answer.model_validate_json(raw)

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return answer, total, prompt_tokens, completion_tokens


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/pinecone")
def debug_pinecone():
    """Confirm Pinecone env vars are set and the index is reachable."""

    return pinecone_health(client)


# curl.exe -s "http://127.0.0.1:8000/debug/retrieve?q=What+is+an+implantable+loop+recorder+used+for"
@app.get("/debug/retrieve", response_model=RetrieveResponse)
def debug_retrieve(
    q: str = Query(..., min_length=1, description="Question to embed and search in Pinecone"),
):
    """Embed the question and return top-5 chunks — no LLM call."""

    question = q.strip()
    if not question:
        raise HTTPException(status_code=400, detail="q must not be empty")

    if not pinecone_configured():
        raise HTTPException(
            status_code=503,
            detail="Pinecone is not configured. Set PINECONE_API_KEY and PINECONE_INDEX_NAME.",
        )

    try:
        top_k = get_settings().retrieval_top_k
        return retrieve_chunks(client, question, top_k=top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc


# curl.exe --% -s -X POST http://127.0.0.1:8000/ingest \
#   -H "Content-Type: application/json" \
#   -d "{\"document_id\": \"doc-001\", \"text\": \"RAG retrieves relevant document chunks before calling the LLM.\", \"source\": \"notes.txt\"}"
@app.post("/ingest", response_model=IngestResponse)
def ingest(body: IngestRequest) -> IngestResponse:
    """Chunk text, embed with text-embedding-3-small, and upsert into Pinecone."""

    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    if not pinecone_configured():
        raise HTTPException(
            status_code=503,
            detail="Pinecone is not configured. Set PINECONE_API_KEY and PINECONE_INDEX_NAME.",
        )

    try:
        chunks_indexed = ingest_document(
            client,
            text=body.text,
            document_id=body.document_id,
            source=body.source,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ingest failed: {exc}") from exc

    if chunks_indexed == 0:
        raise HTTPException(status_code=400, detail="No chunks produced from input text")

    return IngestResponse(
        document_id=body.document_id,
        chunks_indexed=chunks_indexed,
        status="indexed",
    )


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Answer with RAG grounding when Pinecone is configured, else plain generation."""

    model = resolve_model(body.model)
    last_error: str | None = None
    retrieved_chunk_ids: list[str] = []
    prompt = body.question

    # RAG path: embed question, retrieve chunks, build grounding prompt.
    use_rag = pinecone_configured() and not body.force_bad
    if use_rag:
        try:
            top_k = get_settings().retrieval_top_k
            retrieval = retrieve_chunks(client, body.question, top_k=top_k)
            chunks = retrieval["chunks"]
            retrieved_chunk_ids = [chunk["id"] for chunk in chunks]
            prompt = build_grounding_prompt(body.question, chunks)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc

    # Stage 3: one retry keeps the logic legible while still protecting callers.
    try:
        for attempt in range(2):
            try:
                start = time.perf_counter()

                # First attempt with force_bad uses the unsafe path; retry uses structured output.
                use_bad_path = body.force_bad and attempt == 0
                if use_bad_path:
                    answer, tokens_used, prompt_tokens, completion_tokens = call_model_unsafe(
                        body.question, model
                    )
                else:
                    answer, tokens_used, prompt_tokens, completion_tokens = call_model_structured(
                        prompt, model
                    )

                latency_ms = int((time.perf_counter() - start) * 1000)
                cost_usd = compute_cost_usd(model, prompt_tokens, completion_tokens)

                return AskResponse(
                    answer=answer,
                    tokens_used=tokens_used,
                    model=model,
                    latency_ms=latency_ms,
                    cost_usd=round(cost_usd, 6),
                    retrieved_chunk_ids=retrieved_chunk_ids,
                )
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                continue

        # Clean failure — never leak a half-parsed response to the client.
        raise HTTPException(
            status_code=502,
            detail=f"Model response failed schema validation after retry: {last_error}",
        )
    except APIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc
