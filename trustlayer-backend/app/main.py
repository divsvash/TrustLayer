"""
main.py — FastAPI application entrypoint for TrustLayer AI.

Responsibilities:
  - Create the FastAPI app instance
  - Register middleware (CORS, rate limiting, request logging)
  - Mount all routers
  - Expose /health endpoint
  - Configure structured logging

Run with:
    uvicorn app.main:app --reload
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.core.config import get_settings
from app.models.schemas import HealthResponse
from app.routes.query import limiter, router as query_router
from app.routes.upload import router as upload_router
from app.services.rag_service import faiss_store

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger   = logging.getLogger(__name__)
settings = get_settings()


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before the server starts accepting requests."""
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(os.path.dirname(settings.faiss_index_path), exist_ok=True)
    logger.info("TrustLayer AI backend starting up…")
    logger.info(
        "FAISS index: %d vectors across %d documents",
        faiss_store.chunk_count,
        faiss_store.doc_count,
    )
    yield
    logger.info("TrustLayer AI backend shutting down.")


# ── App factory ────────────────────────────────────────────────────────────
app = FastAPI(
    title="TrustLayer AI",
    description=(
        "Hallucination-aware RAG backend. "
        "Answers ONLY from trusted documents with a verification layer."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Rate limiter state attached to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS — tighten origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Replace with your frontend origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ──────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(upload_router, tags=["Documents"])
app.include_router(query_router,  tags=["Query"])


# ── Health ─────────────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        faiss_loaded=faiss_store.chunk_count > 0,
        doc_count=faiss_store.doc_count,
    )


# ── Root ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["System"], include_in_schema=False)
async def root():
    return {
        "service": "TrustLayer AI",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }