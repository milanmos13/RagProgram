"""Settings shared by the build command and the API, read from the environment."""

import os
from pathlib import Path

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 200


def data_dir() -> Path:
    return Path(os.environ.get("NBA_RAG_DATA_DIR", "data"))
