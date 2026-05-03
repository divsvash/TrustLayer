"""
query.py — /ask endpoint for TrustLayer AI.

Orchestrates the full RAG + verification pipeline:
  1. Retrieve relevant chunks from FAISS
  2. Generate a grounded answer (LLM call #1)
  3. Verify the answer against context (LLM call #2) — skipped in fast mode
  4. Return structured QueryResponse

Caching: identical questions return cached results (LRU, 256 entries).
Rate limiting: slowapi limits to 30 req/min per IP.
"""

import hashlib
import logging
from functools import lru_cache
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.models.schemas import QueryMode, QueryRequest, QueryResponse
from app.services.llm_service import _build_context, generate_answer
from app.services.rag_service import faiss_store
from app.services.verifier_service import fast_mode_verdict, verify_answer

logger   = logging.getLogger(__name__)
settings = get_settings()
router   = APIRouter()

# ── Rate limiter ───────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Query result cache ─────────────────────────────────────────────────────
# Keyed by SHA-256(question + mode) → QueryResponse dict
# Using a plain dict instead of lru_cache because we need async compatibility.

_query_cache: dict[str, dict] = {}
_CACHE_MAX = settings.cache_max_size


def _cache_key(question: str, mode: QueryMode) -> str:
    raw = f"{question.strip().lower()}|{mode.value}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_cached(key: str) -> Optional[dict]:
    return _query_cache.get(key)


def _set_cached(key: str, value: dict) -> None:
    if len(_query_cache) >= _CACHE_MAX:
        # Evict oldest entry (FIFO approximation)
        oldest = next(iter(_query_cache))
        del _query_cache[oldest]
    _query_cache[key] = value


# ── Endpoint ───────────────────────────────────────────────────────────────

@router.post(
    "/ask",
    response_model=QueryResponse,
    summary="Ask a question against indexed documents",
    description=(
        "Submit a natural-language question. TrustLayer retrieves relevant "
        "context from uploaded documents, generates a grounded answer, and "
        "(in trust mode) verifies it with a second LLM pass."
    ),
)
@limiter.limit(f"{settings.rate_limit_calls}/{settings.rate_limit_period}seconds")
async def ask_question(
    request: Request,       # required by slowapi
    body: QueryRequest,
) -> QueryResponse:
    """Full RAG + verification pipeline."""

    question = body.question.strip()
    mode     = body.mode

    # ── Cache check ───────────────────────────────────────────
    if settings.enable_cache:
        key    = _cache_key(question, mode)
        cached = _get_cached(key)
        if cached:
            logger.info("Cache hit for question (mode=%s): %r", mode, question[:80])
            return QueryResponse(**{**cached, "cached": True})

    logger.info("Processing query (mode=%s): %r", mode, question[:80])

    # ── Step 1: Retrieve ──────────────────────────────────────
    try:
        chunks = await faiss_store.retrieve(
            question,
            top_k=body.top_k,
        )
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {exc}",
        )

    if faiss_store.chunk_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been indexed yet. Upload a document first via POST /upload.",
        )

    # ── Step 2: Generate answer ────────────────────────────────
    try:
        answer = await generate_answer(question, chunks)
    except Exception as exc:
        logger.exception("Answer generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM error during answer generation: {exc}",
        )

    # ── Step 3: Verify (trust mode) or fast-path ───────────────
    context = _build_context(chunks) if chunks else ""

    try:
        if mode == QueryMode.FAST:
            verdict, explanation, confidence = fast_mode_verdict(answer)
        else:
            verdict, explanation, confidence = await verify_answer(answer, context)
    except Exception as exc:
        logger.exception("Verification failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM error during verification: {exc}",
        )

    # ── Step 4: Build response ────────────────────────────────
    result = QueryResponse(
        answer=answer,
        confidence=confidence,
        verdict=verdict,
        explanation=explanation,
        sources=chunks,
        cached=False,
        mode=mode,
    )

    # ── Cache store ───────────────────────────────────────────
    if settings.enable_cache:
        _set_cached(key, result.model_dump())

    logger.info(
        "Query complete | verdict=%s confidence=%.2f sources=%d",
        verdict, confidence, len(chunks),
    )
    return result