"""
document_loader.py — Load PDF / DOCX / TXT files, clean text, and chunk per document.

Each chunk is returned as a dict:
  {
    "text":      str,          # chunk content
    "chunk_id":  int,          # zero-based index within the document
    "source":    str,          # original filename
    "doc_type":  str,          # "pdf" | "docx" | "txt"
    "page":      int | None,   # page number for PDFs; None otherwise
    "char_start": int,         # start character offset in cleaned text
    "char_end":   int,         # end character offset
  }
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── optional heavy imports (graceful degradation in tests) ─────────────────────
try:
    import pdfplumber  # preferred over PyPDF2 — better text extraction
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

try:
    from docx import Document as DocxDocument
    _HAS_DOCX = True
except ImportError:
    _HAS_DOCX = False


# ─────────────────────────────────────────────────────────────────────────────
# Low-level readers
# ─────────────────────────────────────────────────────────────────────────────

def _read_pdf(path: Path) -> list[dict[str, Any]]:
    """Return a list of {text, page} dicts — one entry per PDF page."""
    if not _HAS_PDFPLUMBER:
        raise ImportError("pdfplumber is required for PDF support: pip install pdfplumber")
    pages: list[dict] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            pages.append({"text": raw, "page": i})
    return pages


def _read_docx(path: Path) -> str:
    """Return full text from a DOCX file."""
    if not _HAS_DOCX:
        raise ImportError("python-docx is required for DOCX support: pip install python-docx")
    doc = DocxDocument(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _read_txt(path: Path) -> str:
    """Return full text from a plain-text file."""
    return path.read_text(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise whitespace, remove control characters, collapse blank lines."""
    # remove non-printable control characters (keep \n and \t)
    text = re.sub(r"[^\S\n\t]+", " ", text)       # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)          # at most two newlines
    text = re.sub(r"[ \t]+\n", "\n", text)          # trailing spaces before newline
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Chunker
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict[str, int]]:
    """
    Split *text* into overlapping fixed-size character chunks.

    Returns list of dicts: {text, char_start, char_end, chunk_id}
    Tries to break on sentence / paragraph boundaries when possible.
    """
    if not text:
        return []

    # Prefer splitting on paragraph then sentence boundaries
    # We'll use a greedy word-boundary approach for cleaner chunks.
    words = text.split()
    chunks: list[dict] = []
    # Build a word→char_offset index so char_start is exact
    word_offsets: list[int] = []
    pos = 0
    for w in words:
        idx = text.find(w, pos)
        word_offsets.append(idx)
        pos = idx + len(w)

    word_idx = 0
    chunk_id = 0

    while word_idx < len(words):
        # Build chunk up to chunk_size characters
        chunk_words: list[str] = []
        char_count = 0
        w = word_idx

        while w < len(words):
            word_len = len(words[w]) + (1 if chunk_words else 0)  # +1 for space
            if char_count + word_len > chunk_size and chunk_words:
                break
            chunk_words.append(words[w])
            char_count += word_len
            w += 1

        chunk_text_str = " ".join(chunk_words)
        char_start = word_offsets[word_idx]
        char_end   = word_offsets[w - 1] + len(words[w - 1])

        chunks.append({
            "text":       chunk_text_str,
            "char_start": char_start,
            "char_end":   char_end,
            "chunk_id":   chunk_id,
        })

        chunk_id += 1

        # If all words fit in one chunk, we're done
        if w >= len(words):
            break

        # Advance word index, stepping back by overlap characters
        overlap_chars = 0
        step_back = 0
        for ow in range(w - 1, word_idx, -1):
            overlap_chars += len(words[ow]) + 1
            if overlap_chars >= chunk_overlap:
                break
            step_back += 1

        word_idx = max(word_idx + 1, w - step_back)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_and_chunk(
    file_path: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict[str, Any]]:
    """
    Load *file_path* (PDF / DOCX / TXT), clean text, chunk, and return
    a list of chunk dicts ready for embedding + storage.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    filename = path.name
    logger.info("Loading %s (%s)", filename, suffix)

    if suffix == ".pdf":
        pages = _read_pdf(path)
        all_chunks: list[dict] = []
        for page_info in pages:
            cleaned = clean_text(page_info["text"])
            if not cleaned:
                continue
            raw_chunks = chunk_text(cleaned, chunk_size, chunk_overlap)
            for c in raw_chunks:
                all_chunks.append({**c, "source": filename, "doc_type": "pdf", "page": page_info["page"]})
        return all_chunks

    elif suffix == ".docx":
        raw = _read_docx(path)
        cleaned = clean_text(raw)
        raw_chunks = chunk_text(cleaned, chunk_size, chunk_overlap)
        return [{**c, "source": filename, "doc_type": "docx", "page": None} for c in raw_chunks]

    elif suffix == ".txt":
        raw = _read_txt(path)
        cleaned = clean_text(raw)
        raw_chunks = chunk_text(cleaned, chunk_size, chunk_overlap)
        return [{**c, "source": filename, "doc_type": "txt", "page": None} for c in raw_chunks]

    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .docx, .txt")


def load_multiple(
    file_paths: list[str | Path],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict[str, Any]]:
    """Load and chunk multiple documents; assign globally unique chunk ids."""
    all_chunks: list[dict] = []
    global_id = 0
    for fp in file_paths:
        try:
            chunks = load_and_chunk(fp, chunk_size, chunk_overlap)
            for c in chunks:
                c["global_chunk_id"] = global_id
                global_id += 1
            all_chunks.extend(chunks)
            logger.info("  → %d chunks from %s", len(chunks), Path(fp).name)
        except Exception as exc:
            logger.error("Failed to load %s: %s", fp, exc)
    return all_chunks
