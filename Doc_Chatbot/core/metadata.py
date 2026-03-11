"""
metadata.py — Enrich document chunks with structured metadata.

Called after document_loader produces raw chunks; before vectorstore ingestion.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_hash(text: str, source: str, chunk_id: int) -> str:
    """Stable, unique ID for a chunk — used as ChromaDB document ID."""
    raw = f"{source}::{chunk_id}::{text[:64]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _file_size_kb(path: str | Path) -> float:
    try:
        return Path(path).stat().st_size / 1024
    except OSError:
        return 0.0


def _infer_language(text: str) -> str:
    """Very lightweight language hint (extend with langdetect if needed)."""
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    return "en" if ascii_ratio > 0.85 else "mixed"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def enrich_chunk(
    chunk: dict[str, Any],
    file_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Add rich metadata to a single chunk dict.

    Parameters
    ----------
    chunk     : raw chunk from document_loader.load_and_chunk()
    file_path : original file path (used for file-level stats)
    extra     : any additional key/value pairs to merge in

    Returns
    -------
    Enriched chunk dict (mutates in place and returns).
    """
    source   = chunk.get("source", "unknown")
    chunk_id = chunk.get("chunk_id", 0)
    text     = chunk.get("text", "")

    chunk["chunk_uid"]      = _chunk_hash(text, source, chunk_id)
    chunk["ingested_at"]    = datetime.now(timezone.utc).isoformat()
    chunk["char_count"]     = len(text)
    chunk["word_count"]     = len(text.split())
    chunk["language_hint"]  = _infer_language(text)

    if file_path is not None:
        chunk["file_size_kb"] = _file_size_kb(file_path)
        chunk["file_path"]    = str(Path(file_path).resolve())

    if extra:
        chunk.update(extra)

    return chunk


def enrich_chunks(
    chunks: list[dict[str, Any]],
    file_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Enrich a list of chunks (convenience wrapper)."""
    return [enrich_chunk(c, file_path, extra) for c in chunks]


def build_chroma_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    Extract a flat, ChromaDB-compatible metadata dict from a chunk.
    ChromaDB metadata values must be str | int | float | bool.
    """
    safe: dict[str, Any] = {}
    allowed_types = (str, int, float, bool)

    # Explicitly include important fields
    for key in (
        "source", "doc_type", "page", "chunk_id", "chunk_uid",
        "ingested_at", "char_count", "word_count", "language_hint",
        "char_start", "char_end", "file_size_kb", "global_chunk_id",
    ):
        val = chunk.get(key)
        if isinstance(val, allowed_types):
            safe[key] = val
        elif val is None:
            safe[key] = ""   # ChromaDB does not support None

    return safe


def format_citation(chunk: dict[str, Any]) -> str:
    """Human-readable citation string for a retrieved chunk."""
    source   = chunk.get("source", "unknown")
    chunk_id = chunk.get("chunk_id", "?")
    page     = chunk.get("page")
    if page:
        return f"[Source: {source}, page {page}, chunk {chunk_id}]"
    return f"[Source: {source}, chunk {chunk_id}]"
