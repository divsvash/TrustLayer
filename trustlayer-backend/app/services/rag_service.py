"""
rag_service.py — FAISS-based vector store for TrustLayer AI.

Responsibilities:
  - Maintain an in-memory FAISS index (persisted to disk).
  - Add documents (embed chunks → store vectors + metadata).
  - Retrieve top-k most similar chunks for a query.

The index is a module-level singleton so it survives across
FastAPI requests without being re-loaded on every call.
"""

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

from app.core.config import get_settings
from app.models.schemas import SourceChunk
from app.utils.chunking import extract_text_from_bytes, split_into_chunks
from app.utils.embeddings import embed_query, embed_texts

logger   = logging.getLogger(__name__)
settings = get_settings()


# ── Chunk metadata ─────────────────────────────────────────────────────────

@dataclass
class ChunkMeta:
    chunk_id: str
    doc_id:   str
    filename: str
    text:     str


# ── Singleton FAISS store ──────────────────────────────────────────────────

class FAISSStore:
    """
    Wraps a FAISS flat L2 index with a parallel metadata list.

    FAISS stores vectors; metadata (text, doc_id, filename) is kept
    in a plain Python list at matching indices.
    """

    def __init__(self) -> None:
        self._index:    Optional[faiss.IndexFlatL2] = None
        self._metadata: List[ChunkMeta]             = []
        self._dim:      int                         = 0
        self._load_or_create()

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def doc_count(self) -> int:
        """Number of unique documents indexed."""
        return len({m.doc_id for m in self._metadata})

    @property
    def chunk_count(self) -> int:
        return len(self._metadata)

    async def add_document(self, raw: bytes, filename: str) -> Dict:
        """
        Full ingest pipeline for one uploaded document.

        1. Extract text
        2. Chunk text
        3. Embed chunks
        4. Add vectors + metadata to FAISS
        5. Persist index to disk

        Returns a dict with doc_id, filename, chunk_count.
        """
        doc_id = str(uuid.uuid4())

        # 1. Extract
        text = extract_text_from_bytes(raw, filename)
        logger.info("Extracted %d chars from '%s'", len(text), filename)

        # 2. Chunk
        chunks = split_into_chunks(
            text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError(f"No text could be extracted from '{filename}'.")

        logger.info("Split '%s' into %d chunks", filename, len(chunks))

        # 3. Embed (batch call — one API round-trip)
        vectors = await embed_texts(chunks)
        matrix  = np.array(vectors, dtype="float32")

        # 4. Initialise index on first document
        if self._index is None:
            self._dim   = matrix.shape[1]
            self._index = faiss.IndexFlatL2(self._dim)
            logger.info("Created FAISS index with dim=%d", self._dim)

        self._index.add(matrix)

        for i, chunk_text in enumerate(chunks):
            self._metadata.append(ChunkMeta(
                chunk_id=f"{doc_id}_{i}",
                doc_id=doc_id,
                filename=filename,
                text=chunk_text,
            ))

        # 5. Persist
        self._save()
        logger.info("Indexed doc '%s' (%s) with %d chunks", filename, doc_id, len(chunks))

        return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks)}

    async def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> List[SourceChunk]:
        """
        Embed *question* and return the top-k most relevant chunks.

        Returns an empty list (not an error) if the index is empty.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("retrieve called but FAISS index is empty")
            return []

        k = min(top_k or settings.top_k_chunks, self._index.ntotal)

        # Embed query
        vec = await embed_query(question)
        q   = np.array([vec], dtype="float32")

        # FAISS search
        distances, indices = self._index.search(q, k)

        results: List[SourceChunk] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            results.append(SourceChunk(
                chunk_id=meta.chunk_id,
                doc_id=meta.doc_id,
                filename=meta.filename,
                text=meta.text,
                score=float(dist),
            ))

        logger.debug("Retrieved %d chunks for query", len(results))
        return results

    # ── Persistence ────────────────────────────────────────────────────────

    def _index_path(self) -> Path:
        return Path(settings.faiss_index_path + ".index")

    def _meta_path(self) -> Path:
        return Path(settings.faiss_index_path + ".meta.json")

    def _load_or_create(self) -> None:
        """Load existing index + metadata from disk, or start fresh."""
        idx_path  = self._index_path()
        meta_path = self._meta_path()

        if idx_path.exists() and meta_path.exists():
            try:
                self._index = faiss.read_index(str(idx_path))
                self._dim   = self._index.d
                with open(meta_path, "r", encoding="utf-8") as f:
                    raw_meta = json.load(f)
                self._metadata = [ChunkMeta(**m) for m in raw_meta]
                logger.info(
                    "Loaded FAISS index (%d vectors, %d docs)",
                    self._index.ntotal,
                    self.doc_count,
                )
            except Exception as exc:
                logger.error("Failed to load FAISS index: %s. Starting fresh.", exc)
                self._index    = None
                self._metadata = []
        else:
            logger.info("No existing FAISS index found — will create on first upload.")

    def _save(self) -> None:
        """Persist index and metadata to disk."""
        idx_path  = self._index_path()
        meta_path = self._meta_path()

        idx_path.parent.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            faiss.write_index(self._index, str(idx_path))

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in self._metadata], f, ensure_ascii=False, indent=2)

        logger.debug("FAISS index persisted to %s", idx_path)


# ── Module-level singleton ─────────────────────────────────────────────────
# Instantiated once at import time; shared across all requests.
faiss_store = FAISSStore()