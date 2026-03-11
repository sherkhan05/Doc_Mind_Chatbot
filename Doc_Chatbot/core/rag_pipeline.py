"""
rag_pipeline.py — LangChain-style RAG orchestration for multi-document retrieval.

Pipeline flow:
  1. User query  →  embed with nomic-embed-text (Ollama)
  2. ChromaDB semantic search  →  top-k chunks
  3. Build numbered context block with source headers
  4. phi3:mini (Ollama) generates a grounded, cited answer
  5. Return RAGResponse (answer + source chunks) for UI rendering

The pipeline is stateless per call; conversation history is passed in by the
caller (Streamlit session_state) and injected into the chat messages list.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Generator

import requests

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Response dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    answer:   str
    sources:  list[dict[str, Any]] = field(default_factory=list)
    query:    str = ""
    model:    str = ""

    @property
    def citation_list(self) -> list[str]:
        """Unique, human-readable citation strings for all retrieved sources."""
        from core.metadata import format_citation
        seen: set[str] = set()
        cites: list[str] = []
        for chunk in self.sources:
            c = format_citation(chunk)
            if c not in seen:
                seen.add(c)
                cites.append(c)
        return cites


# ─────────────────────────────────────────────────────────────────────────────
# Ollama LLM client
# ─────────────────────────────────────────────────────────────────────────────

class OllamaChatClient:
    """
    Thin HTTP wrapper around Ollama's /api/chat endpoint.
    Supports both blocking and streaming generation.
    """

    def __init__(
        self,
        model: str = "phi3:mini",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        temperature: float = 0.1,
    ) -> None:
        self.model       = model
        self.base_url    = base_url.rstrip("/")
        self.timeout     = timeout
        self.temperature = temperature

    def _build_messages(
        self,
        system_prompt: str,
        user_message: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Assemble the full messages list: system → history → user."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Blocking generation — returns complete response string."""
        payload = {
            "model":    self.model,
            "messages": self._build_messages(system_prompt, user_message, chat_history),
            "stream":   False,
            "options":  {"temperature": self.temperature},
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json    = payload,
                timeout = self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Make sure Ollama is running "
                "(check system tray or run `ollama serve`)."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama timed out after {self.timeout}s. "
                "Try a smaller model or increase OLLAMA_TIMEOUT in config.py."
            )

    def chat_stream(
        self,
        system_prompt: str,
        user_message: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> Generator[str, None, None]:
        """
        Streaming generation — yields text tokens as they arrive from Ollama.
        The caller iterates this generator to display tokens in real time.
        """
        payload = {
            "model":    self.model,
            "messages": self._build_messages(system_prompt, user_message, chat_history),
            "stream":   True,
            "options":  {"temperature": self.temperature},
        }
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json    = payload,
                stream  = True,
                timeout = self.timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data  = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except requests.exceptions.ConnectionError:
            yield "\n\n❌ Cannot connect to Ollama. Please check it is running."
        except requests.exceptions.Timeout:
            yield f"\n\n❌ Ollama timed out after {self.timeout}s."

    def health_check(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Orchestrates the full RAG cycle:
      embed query → retrieve chunks → build prompt → generate answer

    Parameters
    ----------
    vectorstore   : VectorStore (ChromaDB) — must already hold indexed chunks
    embedding_fn  : NomicEmbeddings — Ollama nomic-embed-text wrapper
    llm_client    : OllamaChatClient — phi3:mini wrapper
    top_k         : number of chunks to retrieve per query
    system_prompt : RAG prompt template; must contain {context} placeholder
    """

    def __init__(
        self,
        vectorstore=None,
        embedding_fn=None,
        llm_client: OllamaChatClient | None = None,
        top_k: int | None = None,
        system_prompt: str | None = None,
    ) -> None:
        from config import (
            LLM_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT,
            TOP_K, RAG_SYSTEM_PROMPT,
        )

        # Lazy-initialise dependencies if not provided
        if embedding_fn is None:
            from core.embeddings import get_embeddings
            embedding_fn = get_embeddings()

        if vectorstore is None:
            from core.vectorstore import get_vectorstore
            vectorstore = get_vectorstore(embedding_fn)

        if llm_client is None:
            llm_client = OllamaChatClient(
                model    = LLM_MODEL,
                base_url = OLLAMA_BASE_URL,
                timeout  = OLLAMA_TIMEOUT,
            )

        self.vectorstore   = vectorstore
        self.embedding_fn  = embedding_fn
        self.llm           = llm_client
        self.top_k         = top_k         or TOP_K
        self.system_prompt = system_prompt or RAG_SYSTEM_PROMPT

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _retrieve(self, query: str) -> list[dict[str, Any]]:
        """
        Embed the query with nomic-embed-text and fetch top-k chunks from ChromaDB.
        Embedding happens inside vectorstore.query() via the embedding_fn.
        """
        return self.vectorstore.query(query_text=query, top_k=self.top_k)

    def _build_system_prompt(self, chunks: list[dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a numbered context block and inject
        into the RAG system prompt template.
        """
        from core.utils import build_context_block
        context = build_context_block(chunks)
        return self.system_prompt.format(context=context)

    def _trim_history(
        self, chat_history: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Keep only the last 6 messages (3 turns) to avoid context overflow."""
        return chat_history[-6:]

    # ── Public API ────────────────────────────────────────────────────────────

    def answer(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> RAGResponse:
        """
        Full RAG cycle — blocking mode.

        Steps:
          1. Check vectorstore is non-empty
          2. Retrieve top-k chunks via nomic-embed-text + ChromaDB
          3. Build context-injected system prompt
          4. Generate answer with phi3:mini (Ollama)
          5. Return RAGResponse with answer + source chunks

        Returns RAGResponse (answer, sources, query, model).
        """
        if self.vectorstore.count() == 0:
            return RAGResponse(
                answer = (
                    "⚠️ No documents have been uploaded yet.\n\n"
                    "Please upload PDF, DOCX, or TXT files using the sidebar "
                    "and click **Index Documents** first."
                ),
                sources = [],
                query   = query,
            )

        logger.info("RAG query: %.80s", query)

        # Step 1 — Retrieve relevant chunks
        chunks = self._retrieve(query)
        logger.info("Retrieved %d chunks for query.", len(chunks))

        # Step 2 — Build grounded system prompt
        system_msg = self._build_system_prompt(chunks)

        # Step 3 — Generate answer
        history = self._trim_history(chat_history or [])
        try:
            answer = self.llm.chat(
                system_prompt = system_msg,
                user_message  = query,
                chat_history  = history,
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            answer = f"❌ LLM error: {exc}"

        return RAGResponse(
            answer  = answer,
            sources = chunks,
            query   = query,
            model   = self.llm.model,
        )

    def answer_stream(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> tuple[Generator[str, None, None], list[dict[str, Any]]]:
        """
        Full RAG cycle — streaming mode.

        Retrieval is synchronous (fast); generation streams token-by-token.

        Returns
        -------
        (token_generator, source_chunks)
        Iterate the generator to display tokens in real time.
        Source chunks are available immediately for citation rendering.
        """
        if self.vectorstore.count() == 0:
            def _empty_gen():
                yield (
                    "⚠️ No documents uploaded yet.\n\n"
                    "Upload PDF, DOCX, or TXT files in the sidebar and click "
                    "**Index Documents**."
                )
            return _empty_gen(), []

        # Retrieval (synchronous)
        chunks     = self._retrieve(query)
        system_msg = self._build_system_prompt(chunks)
        history    = self._trim_history(chat_history or [])

        # Generation (streaming)
        token_gen = self.llm.chat_stream(
            system_prompt = system_msg,
            user_message  = query,
            chat_history  = history,
        )
        return token_gen, chunks

    def ingest_documents(
        self,
        file_paths: list[str],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict[str, Any]:
        """
        End-to-end ingestion: load → chunk → enrich → embed → store.

        Parameters
        ----------
        file_paths    : list of absolute paths to uploaded documents
        chunk_size    : override config CHUNK_SIZE
        chunk_overlap : override config CHUNK_OVERLAP

        Returns
        -------
        Summary dict: {files_processed, chunks_added, sources}
        """
        from pathlib import Path as _Path
        from config import CHUNK_SIZE, CHUNK_OVERLAP
        from core.document_loader import load_multiple
        from core.metadata import enrich_chunks

        cs = chunk_size    or CHUNK_SIZE
        co = chunk_overlap or CHUNK_OVERLAP

        # Load + chunk all files
        all_chunks = load_multiple(file_paths, chunk_size=cs, chunk_overlap=co)

        # Enrich each file's chunks with metadata
        enriched: list[dict[str, Any]] = []
        for fp in file_paths:
            fname  = _Path(fp).name
            subset = [c for c in all_chunks if c.get("source") == fname]
            enriched.extend(enrich_chunks(subset, file_path=fp))

        # Embed + store via nomic-embed-text → ChromaDB
        if enriched:
            self.vectorstore.add_chunks(enriched)

        return {
            "files_processed": len(file_paths),
            "chunks_added":    len(enriched),
            "sources":         sorted({c["source"] for c in enriched}),
        }

    def health(self) -> dict[str, bool]:
        """Check connectivity to Ollama LLM and embedding services."""
        emb_ok = False
        if hasattr(self.embedding_fn, "_client"):
            emb_ok = self.embedding_fn._client.health_check()

        return {
            "ollama_llm":       self.llm.health_check(),
            "ollama_embedding": emb_ok,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_pipeline(vectorstore=None, embedding_fn=None) -> RAGPipeline:
    """Return a fully initialised RAGPipeline using config defaults."""
    return RAGPipeline(vectorstore=vectorstore, embedding_fn=embedding_fn)
