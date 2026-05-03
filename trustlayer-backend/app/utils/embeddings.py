"""
embeddings.py — OpenAI embedding wrapper for TrustLayer AI.

Provides batch embedding generation with automatic retry on
transient API errors. All embedding calls are routed through
this module so the model can be swapped in one place.
"""

import asyncio
import logging
from typing import List

from openai import AsyncOpenAI

from app.core.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

# Module-level client (re-used across requests for connection pooling)
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def embed_texts(texts: List[str], retries: int = 3) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts:   List of strings to embed.
        retries: Number of retry attempts on transient errors.

    Returns:
        List of embedding vectors (same order as input).

    Raises:
        RuntimeError: If all retry attempts are exhausted.
    """
    if not texts:
        return []

    client = _get_client()

    for attempt in range(1, retries + 1):
        try:
            response = await client.embeddings.create(
                model=settings.embedding_model,
                input=texts,
            )
            # OpenAI guarantees same ordering as input
            vectors = [item.embedding for item in response.data]
            logger.debug("embed_texts: embedded %d texts (attempt %d)", len(texts), attempt)
            return vectors

        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"Embedding failed after {retries} attempts: {exc}"
                ) from exc
            wait = 2 ** attempt  # exponential back-off
            logger.warning("Embedding attempt %d failed (%s). Retrying in %ds…", attempt, exc, wait)
            await asyncio.sleep(wait)

    return []  # unreachable, satisfies type checker


async def embed_query(text: str) -> List[float]:
    """
    Embed a single query string.

    Convenience wrapper around embed_texts for single strings.
    """
    results = await embed_texts([text])
    return results[0]