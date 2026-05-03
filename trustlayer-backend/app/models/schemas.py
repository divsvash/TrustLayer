"""
schemas.py — All Pydantic request/response models for TrustLayer AI.

Every API contract is defined here so routes stay thin and
serialization is validated automatically by FastAPI.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    SUPPORTED            = "SUPPORTED"
    PARTIALLY_SUPPORTED  = "PARTIALLY_SUPPORTED"
    NOT_SUPPORTED        = "NOT_SUPPORTED"


class QueryMode(str, Enum):
    TRUST = "trust"   # full pipeline: retrieval + generation + verification
    FAST  = "fast"    # skip verification, return answer + sources only


# ── Upload ─────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    filename: str      = Field(..., description="Original uploaded filename")
    doc_id:   str      = Field(..., description="UUID assigned to this document")
    chunks:   int      = Field(..., description="Number of chunks indexed into FAISS")
    message:  str      = Field(..., description="Human-readable status message")


# ── Query ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str       = Field(..., min_length=3, max_length=2000,
                                description="The user's natural-language question")
    mode:     QueryMode = Field(QueryMode.TRUST,
                                description="'trust' runs full verification; 'fast' skips it")
    top_k:    Optional[int] = Field(None, ge=1, le=10,
                                description="Override default top-k retrieval count")


class SourceChunk(BaseModel):
    chunk_id:  str   = Field(..., description="Unique chunk identifier")
    doc_id:    str   = Field(..., description="Parent document UUID")
    filename:  str   = Field(..., description="Source filename")
    text:      str   = Field(..., description="Raw text of the retrieved chunk")
    score:     float = Field(..., description="FAISS similarity score (lower = more similar)")


class QueryResponse(BaseModel):
    answer:      str           = Field(..., description="LLM-generated answer grounded in context")
    confidence:  float         = Field(..., ge=0.0, le=1.0,
                                       description="Confidence score: 0.9 / 0.6 / 0.2")
    verdict:     Verdict       = Field(..., description="Hallucination verdict")
    explanation: str           = Field(..., description="One-line verification reasoning")
    sources:     List[SourceChunk] = Field(default_factory=list,
                                           description="Retrieved context chunks")
    cached:      bool          = Field(False, description="True if result was served from cache")
    mode:        QueryMode     = Field(QueryMode.TRUST, description="Pipeline mode used")


# ── Health ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:       str  = "ok"
    faiss_loaded: bool = False
    doc_count:    int  = 0


# ── Error ──────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str