"""Test doubles for the nba_rag adapters."""

from pathlib import Path

import numpy as np

from nba_rag.corpus import ManifestEntry

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


class FakeCatalogue:
    """Catalogue serving fixed lists; every title's current revision is its position + 1000."""

    def __init__(self, franchises: list[str], anniversary_team: list[str]) -> None:
        self._franchises = franchises
        self._anniversary_team = anniversary_team
        self.revisions_requested: list[str] = []

    def franchise_titles(self) -> list[str]:
        return self._franchises

    def anniversary_team_titles(self) -> list[str]:
        return self._anniversary_team

    def current_revisions(self, titles: list[str]) -> list[ManifestEntry]:
        self.revisions_requested.extend(titles)
        return [ManifestEntry(title, 1000 + i) for i, title in enumerate(titles)]
