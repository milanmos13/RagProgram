"""The tutorial's baseline chunker: greedy word accumulation within a paragraph."""

from collections.abc import Callable

TokenCounter = Callable[[str], int]


def baseline_chunks(paragraph: str, count_tokens: TokenCounter, chunk_size: int) -> list[str]:
    """Split a paragraph into chunks of at most ``chunk_size`` tokens.

    Words are added one at a time while the chunk stays within budget. A single word
    longer than the budget becomes a chunk on its own. There is no overlap.

    The tutorial also merged a first chunk under 40 characters into the next one, to
    glue a plain-text title line onto its paragraph. Article HTML has no title
    paragraph, and the merge would join text across paragraphs, so it is not kept.
    """
    chunks: list[str] = []
    current = ""
    for word in paragraph.split(" "):
        candidate = f"{current} {word}" if current else word
        if count_tokens(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks
