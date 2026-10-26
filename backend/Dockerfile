# Base shared by the dev and production images: same Python, same dependencies.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
COPY pyproject.toml ./
COPY src ./src
RUN pip install -e ".[ml]"

# Development: source is mounted over /app and uvicorn reloads on change.
# Data and the model cache live in named volumes (see docker-compose.yml).
FROM base AS dev
ENV HF_HOME=/models \
    NBA_RAG_DATA_DIR=/data
RUN pip install -e ".[ml,dev]"
EXPOSE 8000
CMD ["uvicorn", "nba_rag_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "src"]

# Production (Google Cloud Run, whose source deploys build the last stage).
# The Corpus, index and embedding model are baked in from the manifest at build time,
# so the container starts serving without network access or a rebuild.
FROM base AS prod
ARG NBA_RAG_CONTACT=https://github.com/milanmos13/RagProgram/issues
ENV HF_HOME=/opt/nba-rag/models \
    NBA_RAG_DATA_DIR=/opt/nba-rag/data
COPY manifest.json ./
RUN python -m nba_rag.build --manifest manifest.json --contact "$NBA_RAG_CONTACT"
ENV HF_HUB_OFFLINE=1
# Committed eval results, served by GET /results.
COPY results ./results
# Run as an unprivileged user; the baked files only need to be readable.
USER 1000
EXPOSE 7860
# Shell form so ${PORT} expands: Cloud Run sets PORT, anywhere else it falls back to 7860.
CMD uvicorn nba_rag_api.main:app --host 0.0.0.0 --port ${PORT:-7860}
