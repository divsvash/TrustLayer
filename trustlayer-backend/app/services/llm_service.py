"""
llm_service.py — Grounded answer generation for TrustLayer AI.

Uses the exact prompt specified in the PRD to ensure the LLM
answers ONLY from retrieved context and never from prior knowledge.

Temperature is set to 0.0 for deterministic, factual responses.
"""

import logging
from typing import List

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.models.schemas import SourceChunk

logger   = logging.getLogger(__name__)
settings = get_settings()

# Shared async client (connection-pooled)
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


# ── Prompts (from PRD — do not modify without version bump) ───────────────

ANSWER_SYSTEM_PROMPT = """You are a strict AI assistant.

Rules:
- Answer ONLY using the provided context.
- Do NOT use prior knowledge.
- If the answer is not clearly in the context, say:
  "I don't have enough information to answer this."
"""

ANSWER_USER_TEMPLATE = """Context:
{context}

Question:
{question}

Answer:"""


# ── Public API ─────────────────────────────────────────────────────────────

async def generate_answer(question: str, chunks: List[SourceChunk]) -> str:
    """
    Generate a grounded answer from retrieved context chunks.

    If no chunks are provided, returns the standard "no information"
    response without making an API call.

    Args:
        question: The user's original question.
        chunks:   Top-k retrieved context chunks from FAISS.

    Returns:
        A grounded answer string.
    """
    if not chunks:
        logger.info("No context chunks available — returning withheld response")
        return "I don't have enough information to answer this."

    # Concatenate chunk texts into a single context block
    context = _build_context(chunks)

    user_message = ANSWER_USER_TEMPLATE.format(
        context=context,
        question=question,
    )

    logger.debug("Calling LLM for answer generation (model=%s)", settings.llm_model)

    client   = _get_client()
    response = await client.chat.completions.create(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    answer = response.choices[0].message.content or ""
    answer = answer.strip()
    logger.debug("LLM answer generated (%d chars)", len(answer))
    return answer


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_context(chunks: List[SourceChunk]) -> str:
    """
    Format retrieved chunks into a numbered context block.

    Each chunk is prefixed with its source filename and chunk ID
    so the verifier can reference specific passages.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Source: {chunk.filename} (chunk {chunk.chunk_id})\n"
            f"{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)