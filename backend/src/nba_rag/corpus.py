"""The Corpus: pinned Articles parsed into Chunks."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import quote

from nba_rag.chunking import TokenCounter, baseline_chunks
from nba_rag.parsing import extract_infobox, extract_prose
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


def save_manifest(path: Path, manifest: Manifest) -> None:
    """Write one Article per line, so manifest diffs read as per-Article changes."""
    entries = ",\n".join(
        f"    {json.dumps(asdict(entry), ensure_ascii=False)}" for entry in manifest.articles
    )
    path.write_text(
        f'{{\n  "version": {json.dumps(manifest.version)},\n  "articles": [\n{entries}\n  ]\n}}\n',
        encoding="utf-8",
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
    html = source.fetch_html(entry.title, entry.revision_id)
    paragraphs = extract_prose(html)
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
    chunks.extend(_infobox_chunks(entry, html, count_tokens, chunk_size, len(chunks)))
    return chunks


def _infobox_chunks(
    entry: ManifestEntry,
    html: str,
    count_tokens: TokenCounter,
    chunk_size: int,
    first_position: int,
) -> list[Chunk]:
    """Pack infobox lines into Chunks, keeping each line whole unless it alone is too long.

    Every Chunk starts with the Article title, because infobox lines like ``Position: Point
    guard`` never name their subject and would otherwise be unfindable by it. The title
    counts against the token budget. These Chunks follow the prose Chunks, so prose Chunk
    ids do not depend on the infobox.
    """
    title = entry.title

    def fits(lines: list[str]) -> bool:
        return count_tokens("\n".join([title, *lines])) <= chunk_size

    bodies: list[list[str]] = []
    current: list[str] = []
    for line in extract_infobox(html):
        if not fits([line]):
            if current:
                bodies.append(current)
                current = []
            budget = max(1, chunk_size - count_tokens(title))
            bodies.extend([piece] for piece in baseline_chunks(line, count_tokens, budget))
        elif current and not fits([*current, line]):
            bodies.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        bodies.append(current)
    texts = ["\n".join([title, *body]) for body in bodies]
    return [
        Chunk(
            id=f"{entry.title}#{first_position + i}",
            article_title=entry.title,
            revision_id=entry.revision_id,
            section_path=(),
            origin="infobox",
            text=text,
            position=first_position + i,
        )
        for i, text in enumerate(texts)
    ]
