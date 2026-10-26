"""Composition root: load the built Corpus and real Embedder, then serve.

uvicorn nba_rag_api.main:app
"""

import os

from nba_rag.configurations import load_configurations
from nba_rag.settings import data_dir, results_dir
from nba_rag.store import load_built
from nba_rag_api.app import create_app

try:
    corpus, matrix = load_built(data_dir())
except FileNotFoundError as error:
    raise SystemExit(
        f"No built Corpus in {data_dir()}. Build it first: python -m nba_rag.build"
    ) from error

app = create_app(
    load_configurations(corpus, matrix),
    allowed_origins=os.environ.get("NBA_RAG_ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    results_dir=results_dir(),
)
