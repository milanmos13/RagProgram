"""Persist a built Corpus and its embedding matrix so searches never re-embed."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from nba_rag.corpus import Chunk, Corpus

CHUNKS_FILE = "chunks.jsonl"
EMBEDDINGS_DIR = "embeddings"
"""One ``<embedding setup>.npy`` matrix per embedding setup, one row per Chunk."""
META_FILE = "corpus.json"


def save_built(directory: Path, corpus: Corpus, matrices: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / META_FILE).write_text(
        json.dumps({"manifest_version": corpus.manifest_version}), encoding="utf-8"
    )
    with (directory / CHUNKS_FILE).open("w", encoding="utf-8") as f:
        for chunk in corpus.chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    (directory / EMBEDDINGS_DIR).mkdir(exist_ok=True)
    for setup, matrix in matrices.items():
        np.save(directory / EMBEDDINGS_DIR / f"{setup}.npy", matrix)


def load_built(directory: Path) -> tuple[Corpus, dict[str, np.ndarray]]:
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    with (directory / CHUNKS_FILE).open(encoding="utf-8") as f:
        chunks = [_chunk_from_json(json.loads(line)) for line in f]
    matrices = {
        path.stem: np.load(path) for path in sorted((directory / EMBEDDINGS_DIR).glob("*.npy"))
    }
    if not matrices:
        raise FileNotFoundError(f"No embedding matrices in {directory / EMBEDDINGS_DIR}")
    return Corpus(manifest_version=meta["manifest_version"], chunks=chunks), matrices


def _chunk_from_json(data: dict) -> Chunk:
    return Chunk(**{**data, "section_path": tuple(data["section_path"])})
