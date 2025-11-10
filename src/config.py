"""Configuration helpers for the RAG application."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _load_environment() -> None:
    """Load environment variables from a .env file if present."""

    # Only call load_dotenv once; subsequent calls are inexpensive no-ops.
    load_dotenv()


_load_environment()


def _env_int(name: str, default: int) -> int:
    """Return an integer environment value with a safe fallback."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Return a float environment value with a safe fallback."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class AppConfig:
    """Runtime configuration for the RAG pipeline and UI."""

    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    temperature: float = _env_float("OPENAI_TEMPERATURE", 0.2)
    chunk_size: int = _env_int("CHUNK_SIZE", 1500)  # Increased for better context in large docs
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 300)  # Increased overlap for continuity
    top_k: int = _env_int("TOP_K", 8)  # Increased for better recall
    max_context_chars: int = _env_int("MAX_CONTEXT_CHARS", 2000)  # Increased for large docs
    history_turns: int = _env_int("MAX_HISTORY_TURNS", 6)
    score_threshold: float = _env_float("SCORE_THRESHOLD", 0.5)  # More lenient threshold for better recall
    max_context_sections: int = _env_int("MAX_CONTEXT_SECTIONS", 12)  # Increased for better recall
    use_mmr: bool = os.getenv("USE_MMR", "false").lower() in ("true", "1", "yes")  # Disabled by default for better relevance
    mmr_diversity: float = _env_float("MMR_DIVERSITY", 0.7)  # More relevance-focused when MMR is used
    enable_query_expansion: bool = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() in ("true", "1", "yes")
    fallback_source_label: str = "Unknown document"
    projects_path: str = os.getenv("PROJECTS_PATH", "projects")


__all__ = ["AppConfig"]

