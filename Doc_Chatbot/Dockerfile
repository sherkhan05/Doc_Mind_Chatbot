# ── DocMind RAG Chatbot — Dockerfile ──────────────────────────────────────────
# Python 3.11 slim to match the project's compiled .pyc files
FROM python:3.11-slim

# Metadata
LABEL maintainer="DocMind"
LABEL description="DocMind — Local Document Intelligence Chatbot (Streamlit + Ollama + Qdrant)"

# Prevent Python from buffering stdout/stderr (important for Docker logs)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the full project source
COPY . .

# Create required data directories
RUN mkdir -p data/uploads data/embeddings

# Expose Streamlit's default port
EXPOSE 8501

# Health check — confirms Streamlit is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
