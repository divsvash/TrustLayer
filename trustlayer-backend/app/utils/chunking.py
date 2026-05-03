"""
chunking.py — Splits raw document text into overlapping chunks
suitable for embedding and FAISS indexing.

Strategy: character-level sliding window with sentence-boundary
awareness. This keeps chunks semantically coherent without
requiring a heavy NLP tokeniser.
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def split_into_chunks(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[str]:
    """
    Split *text* into overlapping chunks.

    Args:
        text:          Raw document text (pre-cleaned).
        chunk_size:    Target character length of each chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of non-empty chunk strings.
    """
    if not text or not text.strip():
        return []

    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Final (possibly short) chunk
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to break at a sentence boundary within the last 20% of the window
        boundary_zone = text[end - chunk_size // 5 : end]
        sentence_end  = _last_sentence_end(boundary_zone)

        if sentence_end != -1:
            end = end - (len(boundary_zone) - sentence_end)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Slide forward, respecting overlap
        start = end - chunk_overlap

    logger.debug("split_into_chunks: produced %d chunks from %d chars", len(chunks), len(text))
    return chunks


def _last_sentence_end(text: str) -> int:
    """
    Return index of the last sentence-ending character (.!?) in *text*,
    or -1 if none found.
    """
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ".!?\n":
            return i + 1
    return -1


def extract_text_from_bytes(raw: bytes, filename: str) -> str:
    """
    Extract plain text from uploaded file bytes.

    Supports:
        - .txt  / .md : decoded as UTF-8 (with fallback to latin-1)
        - .pdf        : extracted via PyMuPDF (fitz)

    Args:
        raw:      Raw file bytes.
        filename: Original filename (used to determine parser).

    Returns:
        Extracted plain text string.

    Raises:
        ValueError: If the file type is unsupported or extraction fails.
    """
    lower = filename.lower()

    if lower.endswith((".txt", ".md")):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")

    if lower.endswith(".pdf"):
        return _extract_pdf(raw, filename)

    raise ValueError(
        f"Unsupported file type: '{filename}'. "
        "Accepted: .pdf, .txt, .md"
    )


def _extract_pdf(raw: bytes, filename: str) -> str:
    """Use PyMuPDF to extract text from a PDF byte stream."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF parsing. "
            "Install it with: pip install pymupdf"
        ) from exc

    import io
    doc  = fitz.open(stream=io.BytesIO(raw), filetype="pdf")
    pages: List[str] = []

    for page in doc:
        pages.append(page.get_text("text"))  # type: ignore[attr-defined]

    doc.close()
    text = "\n\n".join(pages)

    if not text.strip():
        raise ValueError(
            f"PDF '{filename}' appears to contain no extractable text "
            "(scanned image PDF). Use an OCR tool first."
        )

    return text