"""An on-disk cache of Article HTML, so parser changes never re-download the Corpus."""

from pathlib import Path
from urllib.parse import quote

from nba_rag.corpus import ArticleSource


class CachedArticleSource:
    """Serves each pinned revision from disk, fetching from ``source`` only on a miss.

    A pinned revision never changes, so cached HTML never goes stale.
    """

    def __init__(self, source: ArticleSource, directory: Path) -> None:
        self._source = source
        self._directory = directory

    def fetch_html(self, title: str, revision_id: int) -> str:
        path = self._directory / f"{quote(title.replace(' ', '_'), safe='')}@{revision_id}.html"
        if path.exists():
            return path.read_text(encoding="utf-8")
        html = self._source.fetch_html(title, revision_id)
        self._directory.mkdir(parents=True, exist_ok=True)
        # Write then rename, so an interrupted build never leaves a truncated file behind.
        partial = path.with_suffix(".partial")
        partial.write_text(html, encoding="utf-8")
        partial.replace(path)
        return html
