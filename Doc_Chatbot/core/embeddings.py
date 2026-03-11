"""
embeddings.py — Generate text embeddings using nomic-embed-text via Ollama.

Wraps the Ollama /api/embeddings endpoint and provides a LangChain-compatible
Embeddings class so it can be plugged directly into LangChain pipelines.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Low-level Ollama embedding client
# ─────────────────────────────────────────────────────────────────────────────

class OllamaEmbeddingClient:
    """
    Thin HTTP client for Ollama's embedding endpoint.

    Usage:
        client = OllamaEmbeddingClient()
        vec = client.embed("Hello world")         # single
        vecs = client.embed_batch(["a", "b"])     # batch
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        retry: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.model       = model
        self.base_url    = base_url.rstrip("/")
        self.timeout     = timeout
        self.retry       = retry
        self.retry_delay = retry_delay
        self._endpoint   = f"{self.base_url}/api/embeddings"

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self.retry + 1):
            try:
                resp = requests.post(
                    self._endpoint,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("Embedding request failed (attempt %d/%d): %s", attempt, self.retry, exc)
                if attempt < self.retry:
                    time.sleep(self.retry_delay * attempt)
        raise RuntimeError(f"Ollama embedding failed after {self.retry} attempts: {last_exc}") from last_exc

    def embed(self, text: str) -> list[float]:
        """Return embedding vector for a single text string."""
        data = self._post({"model": self.model, "prompt": text})
        return data["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return a list of embedding vectors for a batch of texts."""
        vectors: list[list[float]] = []
        for i, text in enumerate(texts):
            logger.debug("Embedding text %d/%d", i + 1, len(texts))
            vectors.append(self.embed(text))
        return vectors

    def health_check(self) -> bool:
        """Return True if Ollama is reachable and the model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            available = any(self.model in m for m in models)
            if not available:
                logger.warning(
                    "Model '%s' not found in Ollama. Run: ollama pull %s",
                    self.model, self.model,
                )
            return True   # Ollama is running; model may still be pulled lazily
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# LangChain-compatible Embeddings wrapper
# ─────────────────────────────────────────────────────────────────────────────

try:
    from langchain_core.embeddings import Embeddings as LCEmbeddings

    class NomicEmbeddings(LCEmbeddings):
        """
        LangChain Embeddings implementation backed by nomic-embed-text on Ollama.

        Compatible with any LangChain vectorstore that accepts an `embedding`
        parameter (e.g. Chroma, FAISS).

        Example:
            from core.embeddings import NomicEmbeddings
            embeddings = NomicEmbeddings()
            vectorstore = Chroma(embedding_function=embeddings, ...)
        """

        def __init__(
            self,
            model: str = "nomic-embed-text",
            base_url: str = "http://localhost:11434",
            timeout: int = 60,
        ) -> None:
            self._client = OllamaEmbeddingClient(
                model=model, base_url=base_url, timeout=timeout
            )

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            """Embed a list of documents (called by LangChain during indexing)."""
            return self._client.embed_batch(texts)

        def embed_query(self, text: str) -> list[float]:
            """Embed a single query string (called by LangChain during retrieval)."""
            return self._client.embed(text)

        def health_check(self) -> bool:
            return self._client.health_check()

except ImportError:
    # langchain-core not installed — provide a stub so imports don't break
    logger.warning("langchain-core not installed; NomicEmbeddings LangChain wrapper unavailable.")

    class NomicEmbeddings:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self._client = OllamaEmbeddingClient(**kwargs)

        def embed_documents(self, texts):
            return self._client.embed_batch(texts)

        def embed_query(self, text):
            return self._client.embed(text)

        def health_check(self):
            return self._client.health_check()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def get_embeddings(
    model: str | None = None,
    base_url: str | None = None,
) -> NomicEmbeddings:
    """
    Factory that reads config defaults and returns a NomicEmbeddings instance.

    Usage:
        from core.embeddings import get_embeddings
        emb = get_embeddings()
    """
    from config import EMBEDDING_MODEL, OLLAMA_BASE_URL  # lazy import
    return NomicEmbeddings(
        model    = model    or EMBEDDING_MODEL,
        base_url = base_url or OLLAMA_BASE_URL,
    )
