"""
utils.py — Shared helper functions used across the project.

Includes:
  • Logging setup
  • Text-cleaning utilities
  • Chunk merging / deduplication
  • Token / character budget helpers
  • File-system helpers
"""
from __future__ import annotations

import hashlib
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", fmt: str | None = None) -> None:
    """Configure root logger with sensible defaults."""
    from config import LOG_FORMAT
    logging.basicConfig(
        level   = getattr(logging, level.upper(), logging.INFO),
        format  = fmt or LOG_FORMAT,
        stream  = sys.stdout,
        force   = True,
    )
    # Quieten noisy third-party loggers
    for noisy in ("httpx", "chromadb", "urllib3", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

def strip_non_printable(text: str) -> str:
    """Remove non-printable / control characters, preserving newlines."""
    return "".join(
        c for c in text
        if c == "\n" or unicodedata.category(c)[0] != "C"
    )


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces / tabs; normalise multiple blank lines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 300, ellipsis: str = "…") -> str:
    """Truncate *text* to *max_chars*, appending *ellipsis* if trimmed."""
    if len(text) <= max_chars:
        return text
    # Try to cut at a word boundary
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + ellipsis


def estimate_tokens(text: str) -> int:
    """
    Rough token count estimate (≈4 chars per token for English).
    Used for budget checks — not suitable for exact tokenisation.
    """
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────────────────────
# Chunk utilities
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove duplicate chunks based on a hash of their text content.
    Preserves order; keeps first occurrence.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for c in chunks:
        h = hashlib.md5(c.get("text", "").encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(c)
    return unique


def merge_adjacent_chunks(
    chunks: list[dict[str, Any]],
    max_chars: int = 1024,
) -> list[dict[str, Any]]:
    """
    Merge consecutive chunks from the *same source / same page* when their
    combined length is within *max_chars*.  Useful for building longer context
    windows after retrieval.
    """
    if not chunks:
        return []

    merged: list[dict[str, Any]] = []
    buf = dict(chunks[0])

    for c in chunks[1:]:
        same_doc  = c.get("source") == buf.get("source")
        same_page = c.get("page")   == buf.get("page")
        fits      = len(buf["text"]) + len(c["text"]) + 1 <= max_chars

        if same_doc and same_page and fits:
            buf["text"]     = buf["text"] + " " + c["text"]
            buf["char_end"] = c.get("char_end", buf.get("char_end", 0))
            buf["word_count"] = len(buf["text"].split())
        else:
            merged.append(buf)
            buf = dict(c)

    merged.append(buf)
    return merged


def build_context_block(
    chunks: list[dict[str, Any]],
    max_total_chars: int = 4000,
) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM prompt.
    Respects *max_total_chars* budget by truncating from the end.
    """
    lines: list[str] = []
    total = 0

    for i, c in enumerate(chunks, start=1):
        source   = c.get("source", "unknown")
        chunk_id = c.get("chunk_id", "?")
        page     = c.get("page")
        loc      = f"page {page}, chunk {chunk_id}" if page else f"chunk {chunk_id}"
        header   = f"[{i}] {source} ({loc})"
        body     = c.get("text", "").strip()
        entry    = f"{header}\n{body}"

        if total + len(entry) > max_total_chars:
            remaining = max_total_chars - total
            if remaining > len(header) + 20:
                entry = f"{header}\n{body[:remaining - len(header) - 4]}…"
            else:
                break

        lines.append(entry)
        total += len(entry)

    return "\n\n---\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# File helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_filename(name: str) -> str:
    """Sanitise a filename, replacing unsafe characters with underscores."""
    return re.sub(r"[^\w\-. ]", "_", name).strip()


def get_file_hash(path: str | Path, algo: str = "sha256") -> str:
    """Return hex digest of a file — useful for deduplication on re-upload."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def human_size(num_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"
