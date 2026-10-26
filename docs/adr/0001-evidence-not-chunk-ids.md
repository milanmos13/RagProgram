# Label Eval questions with Evidence quotes, not chunk IDs

Each Eval question is labelled with Evidence — a verbatim sentence of prose, or one table/infobox row in its text form, from a named Article — and a retrieved Chunk counts as a hit when it contains that Evidence after whitespace normalization. We rejected labelling with gold chunk IDs because every change to chunking (size, overlap, section headers, splitting out infoboxes and tables) would invalidate the labels, and comparing chunking strategies is one of the experiments this project exists to run.

## Consequences

- Evidence must be short (one sentence or one row) so it rarely straddles a Chunk boundary; when it does, the miss is accepted rather than adding fuzzy matching.
- Multi-hop questions carry several Evidence spans and are scored both as all-evidence and any-evidence.
- The validator rejects Evidence that runs over more than one sentence, since that is how a span would cross a paragraph and never fit in any Chunk.
- Evidence must exist verbatim in the parsed Corpus, so Eval questions are validated against the Corpus whenever the manifest's pinned revisions change.
