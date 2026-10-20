"""Retrieval configurations and the single ``retrieve`` entry point."""

from dataclasses import dataclass

import numpy as np

from nba_rag.corpus import Chunk, Corpus
from nba_rag.embedders import Embedder


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Configuration:
    name: str
    chunks: list[Chunk]
    embedder: Embedder
    matrix: np.ndarray
    """One embedding per Chunk, aligned with ``chunks``."""


def build_baseline(
    corpus: Corpus, embedder: Embedder, matrix: np.ndarray | None = None
) -> Configuration:
    """Row 1: the tutorial's vector search, cosine similarity over un-normalized embeddings.

    Pass a precomputed ``matrix`` to skip embedding the Corpus again.
    """
    if matrix is None:
        matrix = embedder.embed_chunks([c.text for c in corpus.chunks])
    return Configuration(
        name="baseline",
        chunks=corpus.chunks,
        embedder=embedder,
        matrix=matrix,
    )


def retrieve(configuration: Configuration, query: str, k: int) -> list[ScoredChunk]:
    scores = _cosine(configuration.matrix, configuration.embedder.embed_query(query))
    order = np.argsort(-scores, kind="stable")
    return _dedupe((ScoredChunk(configuration.chunks[i], float(scores[i])) for i in order), k)


def _cosine(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    dots = matrix @ query
    return np.divide(dots, norms, out=np.zeros_like(dots, dtype=float), where=norms > 0)


def _dedupe(ranked, k: int) -> list[ScoredChunk]:
    """Keep the first occurrence of each Chunk text until ``k`` results are collected."""
    seen: set[str] = set()
    results: list[ScoredChunk] = []
    for scored in ranked:
        key = scored.chunk.text.strip()
        if key in seen:
            continue
        seen.add(key)
        results.append(scored)
        if len(results) == k:
            break
    return results
