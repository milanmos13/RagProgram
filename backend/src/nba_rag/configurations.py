"""The retrieval configurations the commands and the API know by name."""

import numpy as np

from nba_rag.corpus import Corpus
from nba_rag.embedders import TutorialEmbedder
from nba_rag.retrieval import Configuration, build_baseline
from nba_rag.settings import EMBEDDING_MODEL

CONFIGURATION_NAMES = ("baseline",)
"""Cheap to read, so a command can offer them as choices before loading any model."""


def load_configurations(corpus: Corpus, matrix: np.ndarray) -> dict[str, Configuration]:
    """Every configuration over a built Corpus, reusing its stored embeddings."""
    baseline = build_baseline(corpus, TutorialEmbedder(EMBEDDING_MODEL), matrix)
    return {baseline.name: baseline}
