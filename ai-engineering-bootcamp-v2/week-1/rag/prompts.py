"""Grounding prompts for retrieval-augmented generation."""

from __future__ import annotations

from typing import Any

# Template shown to the model — context block is appended per request.
GROUNDING_PROMPT_TEMPLATE = """You are a grounded Q&A assistant. Answer the user's question using ONLY the context chunks below.

Rules:
1. Use ONLY information from the provided context. Do not use outside knowledge.
2. For every fact you state, cite the document_id of each chunk you used inline in your answer (format: [document_id: <id>]).
3. If the context does not contain enough information to answer the question, clearly refuse and explain that the available context is insufficient. Use low confidence and set sources_needed to true.
4. Respond as structured output with fields: answer (string), confidence (number 0-1), sources_needed (boolean).

Context chunks:
{context_block}

User question: {question}
"""


def format_context_block(chunks: list[dict[str, Any]]) -> str:
    """Render retrieved chunks for insertion into the grounding prompt."""

    if not chunks:
        return "(No context chunks were retrieved.)"

    parts = []
    for chunk in chunks:
        parts.append(
            "\n".join(
                [
                    "---",
                    f"Chunk ID: {chunk.get('id', 'unknown')}",
                    f"document_id: {chunk.get('document_id', 'unknown')}",
                    f"source: {chunk.get('source', '')}",
                    f"similarity_score: {chunk.get('score', 0)}",
                    f"text:\n{chunk.get('text', '')}",
                ]
            )
        )
    return "\n".join(parts)


def build_grounding_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Build the full RAG prompt sent to the Session 1 structured generation path."""

    context_block = format_context_block(chunks)
    # Use replace (not str.format) so chunk text with { or } cannot break the prompt.
    return (
        GROUNDING_PROMPT_TEMPLATE.replace("{context_block}", context_block).replace(
            "{question}", question.strip()
        )
    )
