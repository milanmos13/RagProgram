"""Test doubles for the nba_rag adapters."""

from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).parent / "fixtures" / "html"


class FixtureArticleSource:
    """Article source that serves saved HTML fixtures instead of calling Wikipedia."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, int]] = []

    def fetch_html(self, title: str, revision_id: int) -> str:
        self.requests.append((title, revision_id))
        return (FIXTURES / f"{title}.html").read_text(encoding="utf-8")


def count_words(text: str) -> int:
    """Token counter that treats each whitespace-separated word as one token."""
    return len(text.split())


class KeywordEmbedder:
    """Deterministic Embedder: each dimension counts one keyword in the text."""

    VOCABULARY = ("irving", "james", "curry", "mvp", "born")

    def __init__(self) -> None:
        self.chunks_embedded = 0

    def count_tokens(self, text: str) -> int:
        return count_words(text)

    def embed_chunks(self, texts: list[str]) -> np.ndarray:
        self.chunks_embedded += len(texts)
        return np.array([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        words = [w.strip(".,?!()").lower() for w in text.split()]
        return [float(words.count(keyword)) for keyword in self.VOCABULARY]
