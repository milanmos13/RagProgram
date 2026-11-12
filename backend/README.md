# NBA History RAG: backend

FastAPI search API for [NBA History RAG](https://github.com/milanmos13/RagProgram),
deployed to Google Cloud Run (see [docs/deploy.md](../docs/deploy.md)). The production
image builds the Corpus and index from `manifest.json` at build time.

`POST /search` with `{"question": "...", "k": 5}` returns ranked Chunks. Interactive
docs are at `/docs`.

Article text comes from Wikipedia and is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Building the Eval set

`python -m nba_rag.eval_tools` (see its `--help`) covers the whole flow: `search` finds
Evidence by plain text over the Corpus without using any retrieval configuration;
`generate` samples Chunks stratified by Article type and Origin, asks the LLM for a
paraphrased question plus verbatim Evidence, and drops anything whose Evidence isn't in
the Corpus; `review` keeps, edits or drops each survivor in the JSONL review file;
`finalize` appends the kept ones to `eval_set.jsonl`; `split` assigns dev/test stratified by
question type and source. Web questions the Corpus can't answer go in the out-of-corpus
log (`log-out-of-corpus`): one JSON object per line with `question`, `found_at`,
`suggested_article` and `note`.
