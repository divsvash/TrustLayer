"""
config.py — Central configuration for TrustLayer AI.

All settings are loaded from environment variables (via .env).
Never hardcode secrets here.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── OpenAI ────────────────────────────────────────────────
    openai_api_key: str

    # ── LLM behaviour ─────────────────────────────────────────
    llm_model: str = "gpt-4o-mini"          # swap to gpt-4o for better accuracy
    llm_temperature: float = 0.0            # 0 = deterministic, grounded answers
    llm_max_tokens: int = 1024

    # ── Retrieval ─────────────────────────────────────────────
    top_k_chunks: int = 4                   # number of chunks returned by FAISS
    chunk_size: int = 512                   # characters per chunk
    chunk_overlap: int = 64                 # overlap between adjacent chunks

    # ── Embedding model ───────────────────────────────────────
    embedding_model: str = "text-embedding-3-small"

    # ── Storage ───────────────────────────────────────────────
    faiss_index_path: str = "data/faiss_index"
    upload_dir: str = "data/uploads"

    # ── Cache ─────────────────────────────────────────────────
    enable_cache: bool = True
    cache_max_size: int = 256               # LRU entries

    # ── Rate limiting ─────────────────────────────────────────
    rate_limit_calls: int = 30              # requests per window
    rate_limit_period: int = 60            # seconds

    # ── Confidence thresholds ─────────────────────────────────
    confidence_supported: float = 0.9
    confidence_partial: float = 0.6
    confidence_not_supported: float = 0.2

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()