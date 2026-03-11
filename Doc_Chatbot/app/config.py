"""
config.py — Centralised configuration for DocMind Local RAG Chatbot.
All tuneable parameters live here. Import from anywhere in the project.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = BASE_DIR / "data"
UPLOADS_DIR    = DATA_DIR / "uploads"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Ollama / LLM ───────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL       = os.environ.get("LLM_MODEL", "phi3:mini")
OLLAMA_TIMEOUT  = 300

# ── Embedding model ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM   = 768   # nomic-embed-text output dimension

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K = 5

# ── Qdrant ─────────────────────────────────────────────────────────────────────
QDRANT_HOST       = os.environ.get("QDRANT_HOST", None)   # None = embedded/local
QDRANT_PORT       = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_COLLECTION = "doc_intelligence"
CHROMA_COLLECTION = QDRANT_COLLECTION   # alias kept for compatibility
CHROMA_PERSIST_DIR = str(EMBEDDINGS_DIR)

# ── RAG prompt ─────────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are a precise document-intelligence assistant.
Answer the user's question using ONLY the provided context passages.

Rules:
1. Base every claim strictly on the context below — do not invent facts.
2. After each factual statement cite the source: [Source: <filename>, chunk <id>]
3. If context is insufficient say: "I don't have enough information in the uploaded documents."
4. Be concise but thorough. Use bullet points when listing multiple items.

Context:
{context}
"""

# ── Streamlit UI ───────────────────────────────────────────────────────────────
APP_TITLE          = "DocMind — Local Document Intelligence"
APP_ICON           = "🧠"
MAX_UPLOAD_MB      = 50
ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt"]

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
