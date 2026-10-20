"""Embedders turn text into vectors. The real ones load models lazily via transformers."""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed_chunks(self, texts: list[str]) -> np.ndarray:
        """Return one row per text."""
        ...

    def embed_query(self, text: str) -> np.ndarray: ...


class TutorialEmbedder:
    """The tutorial's embedding: mean of the last hidden state over all tokens.

    No normalization and no query instruction; queries and Chunks are embedded the
    same way. Batches use an attention-masked mean so padding doesn't change the result
    the tutorial got by embedding one text at a time.
    """

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).eval()
        self._batch_size = batch_size

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.tokenize(text))

    def embed_chunks(self, texts: list[str]) -> np.ndarray:
        batches = [
            self._embed(texts[i : i + self._batch_size])
            for i in range(0, len(texts), self._batch_size)
        ]
        return np.vstack(batches) if batches else np.zeros((0, self._dimension()))

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> np.ndarray:
        import torch

        inputs = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            hidden = self._model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return ((hidden * mask).sum(dim=1) / mask.sum(dim=1)).numpy()

    def _dimension(self) -> int:
        return self._model.config.hidden_size
