"""
upload.py — /upload endpoint for TrustLayer AI.

Accepts a multipart file upload, runs the full ingest pipeline
(extract → chunk → embed → FAISS index), and returns a summary.

Supported file types: .pdf, .txt, .md
Max file size: 20 MB (enforced by FastAPI before reaching this handler)
"""

import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, status

from app.models.schemas import UploadResponse
from app.services.rag_service import faiss_store

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
ALLOWED_EXTS  = {".pdf", ".txt", ".md"}


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
    description=(
        "Upload a PDF or text file. The file is extracted, chunked, "
        "embedded via OpenAI, and stored in the FAISS vector index."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, TXT, or MD file to index"),
) -> UploadResponse:
    """Ingest an uploaded document into the FAISS vector store."""

    # ── Validation ────────────────────────────────────────────
    filename = file.filename or "unknown"
    ext      = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTS)}",
        )

    raw = await file.read()

    if len(raw) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE // 1_048_576} MB.",
        )

    logger.info("Received upload: '%s' (%d bytes)", filename, len(raw))

    # ── Ingest ────────────────────────────────────────────────
    try:
        result = await faiss_store.add_document(raw, filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error during document ingestion")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )

    return UploadResponse(
        filename=result["filename"],
        doc_id=result["doc_id"],
        chunks=result["chunks"],
        message=(
            f"Successfully indexed '{filename}' into {result['chunks']} chunks. "
            "The document is now searchable."
        ),
    )