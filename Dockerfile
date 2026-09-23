# Single-container image: API, MCP tool server, embedded vector index, and the
# tools database are all built into the image so the deployment has no external
# dependencies. The knowledge base is indexed at build time, not at startup.
# Thread and allocator caps keep resident memory under the 512 MB of a free-tier host.

# Stage 1: build the single-page app. Nothing from this stage runs at runtime; only the
# static files are copied forward, so the image gains no Node process and no memory.
FROM node:20-alpine AS frontend
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/models_cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    DATA_DIR=/app/data \
    EMBEDDING_BACKEND=onnx \
    TOOLS_BACKEND=sqlite \
    SQLITE_PATH=/app/data/darkstore.sqlite \
    QDRANT_PATH=/app/qdrant_local \
    FRONTEND_DIST=/app/static \
    EVAL_RESULTS_DIR=/app/eval/results \
    QDRANT_URL= \
    AGENT_MODE=agentic \
    PORT=7860

WORKDIR /app

# The runtime dependencies deliberately exclude torch: embeddings run through
# onnxruntime, which is what keeps the container inside a 512 MB instance.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Hugging Face Spaces runs containers as uid 1000. Everything the service writes
# is created as that user from here on, so no recursive chown layer is needed.
RUN useradd -m -u 1000 app \
 && mkdir -p /app/data /app/qdrant_local /app/models_cache \
 && chown app:app /app /app/data /app/qdrant_local /app/models_cache
COPY --chown=app:app data ./data
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app eval/results/*.json ./eval/results/
COPY --from=frontend --chown=app:app /ui/dist ./static
USER app

# Build the tools database and the vector index into the image. This is the only
# point at which the embedding model is downloaded.
RUN python -m qc_copilot.tools.seed --sqlite /app/data/darkstore.sqlite \
 && python -m qc_copilot.ingest.pipeline --recreate \
 && python -c "from qc_copilot.config import get_settings; from qc_copilot.retrieval.embeddings import build_embedder; from qc_copilot.retrieval.store import VectorStore; s=get_settings(); st=VectorStore.from_settings(s, build_embedder(s).dimension); print('indexed vectors:', st.count()); st.client.close()"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4).status==200 else 1)"

CMD ["sh", "-c", "uvicorn qc_copilot.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
