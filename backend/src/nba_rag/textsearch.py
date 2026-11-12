"""Plain text search over the Corpus, for finding Evidence without the system's retriever.

A Chunk matches when it contains every word of the query, ignoring case and whitespace.
There is no ranking and no embedding: hits come back in Corpus order, so labelling an
Eval question never depends on what any retrieval configuration would have returned.
"""

from nba_rag.corpus import Chunk, Corpus


def search_text(corpus: Corpus, query: str, limit: int = 20) -> list[Chunk]:
    words = query.lower().split()
    if not words:
        return []
    hits: list[Chunk] = []
    for chunk in corpus.chunks:
        text = " ".join(chunk.text.lower().split())
        if all(word in text for word in words):
            hits.append(chunk)
            if len(hits) == limit:
                break
    return hits
