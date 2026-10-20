"""The Corpus: pinned Articles parsed into Chunks."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import quote

from nba_rag.chunking import TokenCounter, baseline_chunks
from nba_rag.parsing import extract_prose
from nba_rag.settings import CHUNK_SIZE

Origin = Literal["prose", "infobox", "table"]


@dataclass(frozen=True)
class ManifestEntry:
    title: str
    revision_id: int


@dataclass(frozen=True)
class Manifest:
    version: str
    articles: list[ManifestEntry]


def load_manifest(path: Path) -> Manifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Manifest(
        version=data["version"],
        articles=[ManifestEntry(**entry) for entry in data["articles"]],
    )


@dataclass(frozen=True)
class Chunk:
    id: str
    article_title: str
    revision_id: int
    section_path: tuple[str, ...]
    origin: Origin
    text: str
    position: int

    @property
    def revision_url(self) -> str:
        title = quote(self.article_title.replace(" ", "_"))
        return f"https://en.wikipedia.org/w/index.php?title={title}&oldid={self.revision_id}"


@dataclass(frozen=True)
class Corpus:
    manifest_version: str
    chunks: list[Chunk]


class ArticleSource(Protocol):
    def fetch_html(self, title: str, revision_id: int) -> str: ...


class EmptyArticleError(ValueError):
    """An Article's parse produced no prose, which means the parser has regressed."""


def build_corpus(
    manifest: Manifest,
    *,
    source: ArticleSource,
    count_tokens: TokenCounter,
    chunk_size: int = CHUNK_SIZE,
) -> Corpus:
    chunks: list[Chunk] = []
    for entry in manifest.articles:
        chunks.extend(_chunk_article(entry, source, count_tokens, chunk_size))
    return Corpus(manifest_version=manifest.version, chunks=chunks)


def _chunk_article(
    entry: ManifestEntry,
    source: ArticleSource,
    count_tokens: TokenCounter,
    chunk_size: int,
) -> list[Chunk]:
    paragraphs = extract_prose(source.fetch_html(entry.title, entry.revision_id))
    if not paragraphs:
        raise EmptyArticleError(
            f"{entry.title} (revision {entry.revision_id}) has no prose; the parser may be broken"
        )
    chunks: list[Chunk] = []
    for paragraph in paragraphs:
        for text in baseline_chunks(paragraph.text, count_tokens, chunk_size):
            position = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{entry.title}#{position}",
                    article_title=entry.title,
                    revision_id=entry.revision_id,
                    section_path=paragraph.section_path,
                    origin="prose",
                    text=text,
                    position=position,
                )
            )
    return chunks
