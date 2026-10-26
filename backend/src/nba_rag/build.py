"""One-off command: fetch the manifest's Articles, chunk and embed them into the data dir.

    python -m nba_rag.build --manifest manifest.json --contact you@example.com

The contact can also come from NBA_RAG_CONTACT.
"""

import argparse
import os
from pathlib import Path

from nba_rag.corpus import build_corpus, load_manifest
from nba_rag.embedders import TutorialEmbedder
from nba_rag.retrieval import build_baseline
from nba_rag.settings import CHUNK_SIZE, EMBEDDING_MODEL, data_dir
from nba_rag.store import save_built
from nba_rag.wikipedia import WikipediaArticleSource


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifest.json"))
    parser.add_argument("--out", type=Path, default=None, help="defaults to NBA_RAG_DATA_DIR")
    parser.add_argument(
        "--contact",
        default=os.environ.get("NBA_RAG_CONTACT") or None,
        help="email or URL for the Wikimedia user agent (default: NBA_RAG_CONTACT)",
    )
    args = parser.parse_args(argv)
    if not args.contact:
        parser.error("a contact is required: pass --contact or set NBA_RAG_CONTACT")

    manifest = load_manifest(args.manifest)
    embedder = TutorialEmbedder(EMBEDDING_MODEL)
    corpus = build_corpus(
        manifest,
        source=WikipediaArticleSource(contact=args.contact),
        count_tokens=embedder.count_tokens,
        chunk_size=CHUNK_SIZE,
    )
    baseline = build_baseline(corpus, embedder)
    out = args.out or data_dir()
    save_built(out, corpus, baseline.matrix)
    print(f"Built {len(corpus.chunks)} Chunks from {len(manifest.articles)} Articles into {out}")


if __name__ == "__main__":
    main()
