"""
main.py — Streamlit entry point for DocMind: Local Document Intelligence Chatbot.

Run with:
    streamlit run app/main.py
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import logging
from pathlib import Path

import streamlit as st

from config import (
    APP_ICON, APP_TITLE, CHUNK_OVERLAP, CHUNK_SIZE,
    TOP_K, UPLOADS_DIR,
)
from components import (
    inject_css,
    render_assistant_bubble,
    render_citation_badges,
    render_document_list,
    render_empty_chat_hint,
    render_file_uploader,
    render_header,
    render_health_status,
    render_source_cards,
    render_stats_bar,
    render_user_bubble,
)
from core.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title            = APP_TITLE,
    page_icon             = APP_ICON,
    layout                = "wide",
    initial_sidebar_state = "expanded",
)
inject_css()


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "messages":        [],
        "pipeline":        None,
        "indexed_sources": [],
        "chunk_count":     0,
        "upload_key":      0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_pipeline():
    from core.rag_pipeline import get_pipeline as _get
    return _get()


def get_pipeline():
    if st.session_state["pipeline"] is None:
        with st.spinner("⚙️ Initialising AI pipeline…"):
            st.session_state["pipeline"] = _load_pipeline()
    return st.session_state["pipeline"]


# ─────────────────────────────────────────────────────────────────────────────
# Document ingestion
# ─────────────────────────────────────────────────────────────────────────────

def _save_uploaded_file(uploaded_file) -> Path:
    dest = UPLOADS_DIR / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def _ingest_files(uploaded_files) -> None:
    pipeline  = get_pipeline()
    new_files: list[str] = []

    for uf in uploaded_files:
        if uf.name in st.session_state["indexed_sources"]:
            st.toast(f"⏭️ {uf.name} already indexed — skipped.", icon="ℹ️")
            continue
        dest = _save_uploaded_file(uf)
        new_files.append(str(dest))

    if not new_files:
        return

    with st.spinner(f"🔍 Processing {len(new_files)} file(s)…"):
        result = pipeline.ingest_documents(
            file_paths    = new_files,
            chunk_size    = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
        )

    st.session_state["indexed_sources"].extend(result["sources"])
    st.session_state["chunk_count"] = pipeline.vectorstore.count()
    st.success(
        f"✅ Indexed **{result['chunks_added']}** chunks from "
        f"**{result['files_processed']}** file(s).",
        icon="📄",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📁 Documents")

        uploaded = render_file_uploader()
        if uploaded:
            if st.button("⬆️ Index Documents", use_container_width=True, type="primary"):
                _ingest_files(uploaded)
                st.session_state["upload_key"] += 1

        st.divider()

        st.markdown("**Loaded Documents**")
        render_document_list(st.session_state["indexed_sources"])

        st.divider()

        st.markdown("## ⚙️ Settings")
        new_topk = st.slider(
            "Top-K retrieval", min_value=1, max_value=10, value=TOP_K,
            help="Number of document chunks retrieved per query.",
        )
        if st.session_state["pipeline"] and new_topk != st.session_state["pipeline"].top_k:
            st.session_state["pipeline"].top_k = new_topk

        st.divider()

        if st.button("🔄 Check Service Health", use_container_width=True):
            render_health_status(get_pipeline().health())

        st.divider()

        with st.expander("⚠️ Danger Zone"):
            if st.button("🗑️ Clear All Documents", use_container_width=True):
                pipeline = get_pipeline()
                pipeline.vectorstore.clear()
                st.session_state["indexed_sources"] = []
                st.session_state["chunk_count"]     = 0
                st.session_state["messages"]        = []
                st.success("Vector store cleared.")
                st.rerun()

        st.markdown(
            '<p style="font-size:0.72rem;color:#3a4060;text-align:center;margin-top:12px;">'
            'DocMind · Qdrant + phi3:mini + nomic-embed-text'
            '</p>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────

def _render_chat() -> None:
    render_header()
    render_stats_bar(
        st.session_state["chunk_count"],
        len(st.session_state["indexed_sources"]),
    )

    if not st.session_state["messages"]:
        render_empty_chat_hint()
    else:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                render_user_bubble(msg["content"])
            else:
                render_assistant_bubble(msg["content"])
                if msg.get("sources"):
                    render_source_cards(msg["sources"])
                if msg.get("citations"):
                    render_citation_badges(msg["citations"])

    user_input = st.chat_input(
        "Ask a question about your documents…",
        disabled = st.session_state["chunk_count"] == 0,
    )

    if user_input:
        _handle_query(user_input)


def _handle_query(query: str) -> None:
    pipeline = get_pipeline()

    st.session_state["messages"].append({"role": "user", "content": query})
    render_user_bubble(query)

    lc_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["messages"][:-1]
        if m["role"] in ("user", "assistant")
    ]

    placeholder  = st.empty()
    full_answer  = ""
    sources: list = []

    try:
        token_gen, sources = pipeline.answer_stream(query, chat_history=lc_history)
        with placeholder.container():
            st.markdown('<div class="chat-assistant">', unsafe_allow_html=True)
            answer_placeholder = st.empty()
            for token in token_gen:
                full_answer += token
                answer_placeholder.markdown(full_answer + " ▌")
            answer_placeholder.markdown(full_answer)
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception as exc:
        full_answer = f"❌ Error: {exc}"
        placeholder.error(full_answer)

    from core.metadata import format_citation
    citations = list({format_citation(c) for c in sources})

    st.session_state["messages"].append({
        "role":      "assistant",
        "content":   full_answer,
        "sources":   sources,
        "citations": citations,
    })

    render_source_cards(sources)
    render_citation_badges(citations)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _render_sidebar()
    _render_chat()


if __name__ == "__main__":
    main()

