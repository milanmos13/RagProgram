"""Slow tier: uses the real bge-small model."""

import numpy as np
import pytest

from nba_rag.settings import EMBEDDING_MODEL as MODEL

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tutorial_embedder():
    from nba_rag.embedders import TutorialEmbedder

    return TutorialEmbedder(MODEL)


def tutorial_vector(text: str) -> np.ndarray:
    """The tutorial's original code path, one text at a time."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        return model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()


def test_batched_chunks_match_the_tutorials_unbatched_mean_pooling(tutorial_embedder):
    texts = ["Curry was born in Akron, Ohio.", "LeBron James blocked a layup by Andre Iguodala."]

    matrix = tutorial_embedder.embed_chunks(texts)

    for row, text in zip(matrix, texts, strict=True):
        np.testing.assert_allclose(row, tutorial_vector(text), atol=1e-5)


def test_query_gets_no_instruction_and_no_normalization(tutorial_embedder):
    query = "Who won the 2016 NBA Finals?"

    vector = tutorial_embedder.embed_query(query)

    np.testing.assert_allclose(vector, tutorial_vector(query), atol=1e-5)
    assert not np.isclose(np.linalg.norm(vector), 1.0)


def test_counts_tokens_with_the_model_tokenizer(tutorial_embedder):
    assert tutorial_embedder.count_tokens("The basketball player") == 3
