"""Configuration helpers for the RAG application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

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
    chunk_size: int = _env_int("CHUNK_SIZE", 1200)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 200)
    top_k: int = _env_int("TOP_K", 6)
    max_context_chars: int = _env_int("MAX_CONTEXT_CHARS", 1600)
    history_turns: int = _env_int("MAX_HISTORY_TURNS", 6)
    score_threshold: float = _env_float("SCORE_THRESHOLD", 0.35)
    max_context_sections: int = _env_int("MAX_CONTEXT_SECTIONS", 6)
    fallback_source_label: str = "Unknown document"

    def as_dict(self) -> Dict[str, Any]:
        """Return the configuration as a serialisable dictionary."""

        return {
            "chat_model": self.chat_model,
            "embedding_model": self.embedding_model,
            "temperature": self.temperature,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "max_context_chars": self.max_context_chars,
            "history_turns": self.history_turns,
            "score_threshold": self.score_threshold,
            "max_context_sections": self.max_context_sections,
        }


__all__ = ["AppConfig"]

