# NBA History RAG: backend

FastAPI search API for [NBA History RAG](https://github.com/milanmos13/RagProgram),
deployed to Google Cloud Run (see [docs/deploy.md](../docs/deploy.md)). The production
image builds the Corpus and index from `manifest.json` at build time.

`POST /search` with `{"question": "...", "k": 5}` returns ranked Chunks. Interactive
docs are at `/docs`.

Article text comes from Wikipedia and is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
