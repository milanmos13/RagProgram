"""One-off command: generate the manifest from the Corpus inclusion rule.

    python -m nba_rag.generate_manifest --version full-2026-10 --last-season 2025

Articles already in the output manifest keep their pinned revision, so existing Eval
Evidence stays valid; delete the file first to re-pin everything to the current revisions.
The contact can also come from NBA_RAG_CONTACT.
"""

import argparse
import os
from pathlib import Path

from nba_rag.corpus import load_manifest, save_manifest
from nba_rag.manifest import generate_manifest
from nba_rag.wikipedia import WikipediaCatalogue


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="the new manifest's version label")
    parser.add_argument(
        "--last-season",
        type=int,
        required=True,
        help="start year of the latest completed season (2025 for 2025–26)",
    )
    parser.add_argument("--out", type=Path, default=Path("manifest.json"))
    parser.add_argument(
        "--contact",
        default=os.environ.get("NBA_RAG_CONTACT") or None,
        help="email or URL for the Wikimedia user agent (default: NBA_RAG_CONTACT)",
    )
    args = parser.parse_args(argv)
    if not args.contact:
        parser.error("a contact is required: pass --contact or set NBA_RAG_CONTACT")

    previous = load_manifest(args.out) if args.out.exists() else None
    manifest = generate_manifest(
        WikipediaCatalogue(contact=args.contact),
        version=args.version,
        last_season=args.last_season,
        previous=previous,
    )
    save_manifest(args.out, manifest)
    print(f"Wrote {len(manifest.articles)} Articles to {args.out}")


if __name__ == "__main__":
    main()
