"""
components.py — Reusable Streamlit UI components for DocMind.

All visual building blocks live here; main.py only calls these functions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# CSS Injection
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* ── Typography & base ───────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    letter-spacing: -0.02em;
}

/* ── App background ──────────────────────────────────────────────────── */
.stApp {
    background: #0e1117;
    color: #e8e9ed;
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #13151c;
    border-right: 1px solid #1f2330;
}
section[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.82rem;
    color: #8b92a5;
}

/* ── Chat bubble: user ───────────────────────────────────────────────── */
.chat-user {
    background: linear-gradient(135deg, #1a4fd4 0%, #1238a8 100%);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 8px 0 8px 20%;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 2px 12px rgba(26, 79, 212, 0.25);
}

/* ── Chat bubble: assistant ──────────────────────────────────────────── */
.chat-assistant {
    background: #181c27;
    border: 1px solid #252b3b;
    color: #dde0ea;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    margin: 8px 20% 8px 0;
    font-size: 0.92rem;
    line-height: 1.65;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.chat-assistant code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    background: #0e1117;
    padding: 2px 5px;
    border-radius: 4px;
    color: #7dd3fc;
}

/* ── Citation badges ─────────────────────────────────────────────────── */
.citation-badge {
    display: inline-block;
    background: #1b2540;
    border: 1px solid #2d3a5c;
    color: #7dd3fc;
    border-radius: 6px;
    font-size: 0.73rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 2px 7px;
    margin: 2px 3px;
}

/* ── Source card ─────────────────────────────────────────────────────── */
.source-card {
    background: #141720;
    border: 1px solid #1e2436;
    border-left: 3px solid #1a4fd4;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 0.8rem;
    color: #9da5bb;
}
.source-card strong {
    color: #c8cdd8;
    font-weight: 600;
}
.source-card .source-score {
    float: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #4ade80;
    background: #0f2218;
    padding: 1px 6px;
    border-radius: 4px;
}

/* ── Upload area ─────────────────────────────────────────────────────── */
.upload-hint {
    border: 1px dashed #2a3350;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    color: #5a6480;
    font-size: 0.82rem;
    margin: 8px 0;
}

/* ── Doc pill ────────────────────────────────────────────────────────── */
.doc-pill {
    display: inline-flex;
    align-items: center;
    background: #0f1624;
    border: 1px solid #1d2540;
    border-radius: 20px;
    padding: 4px 10px 4px 8px;
    font-size: 0.78rem;
    color: #9da8c0;
    margin: 3px 3px;
}
.doc-pill-icon {
    margin-right: 5px;
    font-size: 0.9rem;
}

/* ── Status indicators ───────────────────────────────────────────────── */
.status-ok   { color: #4ade80; font-weight: 600; }
.status-warn { color: #fbbf24; font-weight: 600; }
.status-err  { color: #f87171; font-weight: 600; }

/* ── Thinking spinner ────────────────────────────────────────────────── */
.thinking-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #1a4fd4;
    margin: 0 2px;
    animation: blink 1.2s infinite;
}
.thinking-dot:nth-child(2) { animation-delay: 0.2s; }
.thinking-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
    0%, 80%, 100% { opacity: 0.15; transform: scale(0.8); }
    40%            { opacity: 1;    transform: scale(1.1); }
}
</style>
"""


def inject_css() -> None:
    """Inject global custom CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header / branding
# ─────────────────────────────────────────────────────────────────────────────

def render_header() -> None:
    """Top-of-page branding bar."""
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:4px 0 18px;">
        <span style="font-size:2.1rem;">🧠</span>
        <div>
            <h1 style="margin:0;font-size:1.7rem;color:#e8e9ed;">DocMind</h1>
            <p style="margin:0;font-size:0.8rem;color:#5a6480;letter-spacing:0.08em;
               text-transform:uppercase;font-weight:500;">
               Local Document Intelligence
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chat bubbles
# ─────────────────────────────────────────────────────────────────────────────

def render_user_bubble(text: str) -> None:
    """Render a user chat bubble."""
    safe = text.replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f'<div class="chat-user">{safe}</div>', unsafe_allow_html=True)


def render_assistant_bubble(text: str) -> None:
    """Render an assistant chat bubble (supports markdown via st.markdown)."""
    with st.container():
        st.markdown(f'<div class="chat-assistant">', unsafe_allow_html=True)
        st.markdown(text)
        st.markdown("</div>", unsafe_allow_html=True)


def render_thinking() -> None:
    """Animated 'thinking' indicator."""
    st.markdown(
        '<div style="padding:10px 0;">'
        '<span class="thinking-dot"></span>'
        '<span class="thinking-dot"></span>'
        '<span class="thinking-dot"></span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Source / citation rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_source_cards(chunks: list[dict[str, Any]], max_shown: int = 5) -> None:
    """
    Render retrieved source chunks as expandable cards below an answer.
    """
    if not chunks:
        return

    with st.expander(f"📄 View {min(len(chunks), max_shown)} source passage(s)", expanded=False):
        for c in chunks[:max_shown]:
            source   = c.get("source", "unknown")
            chunk_id = c.get("chunk_id", "?")
            page     = c.get("page")
            score    = c.get("score", 0.0)
            text     = c.get("text", "")[:320]

            loc = f"page {page}, chunk {chunk_id}" if page else f"chunk {chunk_id}"
            score_str = f"{score:.2f}" if isinstance(score, float) else str(score)

            st.markdown(
                f"""<div class="source-card">
                    <strong>📄 {source}</strong>
                    <span class="source-score">sim {score_str}</span>
                    <br><span style="font-size:0.73rem;color:#5a6480;">{loc}</span>
                    <p style="margin:6px 0 0;font-size:0.8rem;line-height:1.5;">{text}…</p>
                </div>""",
                unsafe_allow_html=True,
            )


def render_citation_badges(citations: list[str]) -> None:
    """Render inline citation badges for a list of citation strings."""
    if not citations:
        return
    badges = "".join(
        f'<span class="citation-badge">{c}</span>' for c in citations
    )
    st.markdown(
        f'<div style="margin-top:6px;">{badges}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Document management UI
# ─────────────────────────────────────────────────────────────────────────────

def render_doc_pill(filename: str) -> None:
    """Render a single document pill (sidebar doc list)."""
    ext  = Path(filename).suffix.lower()
    icon = {"pdf": "📕", "docx": "📘", "txt": "📄"}.get(ext.lstrip("."), "📎")
    st.markdown(
        f'<div class="doc-pill"><span class="doc-pill-icon">{icon}</span>{filename}</div>',
        unsafe_allow_html=True,
    )


def render_document_list(sources: list[str]) -> None:
    """Render all loaded document pills in the sidebar."""
    if not sources:
        st.markdown(
            '<div class="upload-hint">No documents loaded yet</div>',
            unsafe_allow_html=True,
        )
        return
    for src in sources:
        render_doc_pill(src)


# ─────────────────────────────────────────────────────────────────────────────
# Status / health indicators
# ─────────────────────────────────────────────────────────────────────────────

def render_health_status(health: dict[str, bool]) -> None:
    """Show Ollama service health indicators in the sidebar."""
    st.markdown("**Service Status**")
    rows = {
        "ollama_llm":       "🤖 LLM (phi3:mini)",
        "ollama_embedding": "🔢 Embeddings (nomic)",
    }
    for key, label in rows.items():
        ok = health.get(key, False)
        css_class = "status-ok" if ok else "status-err"
        symbol    = "●" if ok else "●"
        status    = "Online" if ok else "Offline"
        st.markdown(
            f'<span class="{css_class}">{symbol} {label}: {status}</span>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# File uploader wrapper
# ─────────────────────────────────────────────────────────────────────────────

def render_file_uploader() -> list | None:
    """
    Styled file uploader.
    Returns list of uploaded UploadedFile objects or None.
    """
    from app.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_MB

    st.markdown("#### 📂 Upload Documents")
    st.markdown(
        '<div class="upload-hint">'
        f'PDF · DOCX · TXT &nbsp;|&nbsp; Max {MAX_UPLOAD_MB} MB each'
        '</div>',
        unsafe_allow_html=True,
    )
    files = st.file_uploader(
        label      = "Choose files",
        type       = [e.lstrip(".") for e in ALLOWED_EXTENSIONS],
        accept_multiple_files = True,
        label_visibility = "collapsed",
    )
    return files if files else None


# ─────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_empty_chat_hint() -> None:
    """Shown in the chat area before any conversation starts."""
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#3a4060;">
        <div style="font-size:3rem;margin-bottom:12px;">📚</div>
        <p style="font-size:1.05rem;font-weight:500;color:#5a6480;">
            Upload documents and start asking questions
        </p>
        <p style="font-size:0.82rem;color:#3a4060;">
            Answers will be grounded in your documents with source citations.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_stats_bar(chunk_count: int, doc_count: int) -> None:
    """Small stats bar shown at the top of the chat area."""
    if chunk_count == 0:
        return
    st.markdown(
        f'<div style="font-size:0.75rem;color:#4a5268;padding:4px 0 10px;">'
        f'📊 <strong style="color:#6b7594;">{doc_count}</strong> doc(s) · '
        f'<strong style="color:#6b7594;">{chunk_count}</strong> indexed chunks'
        f'</div>',
        unsafe_allow_html=True,
    )
