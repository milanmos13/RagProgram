# Base shared by the dev image (here) and the production image (added with deployment).
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    NBA_RAG_DATA_DIR=/data
WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
COPY pyproject.toml ./
COPY src ./src
RUN pip install -e ".[ml]"

# Development: source is mounted over /app and uvicorn reloads on change.
FROM base AS dev
RUN pip install -e ".[ml,dev]"
EXPOSE 8000
CMD ["uvicorn", "nba_rag_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "src"]
