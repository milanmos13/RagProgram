---
title: NBA History RAG
emoji: 🏀
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
short_description: Search API over pinned NBA-history Wikipedia Articles
license: mit
---

# NBA History RAG: backend

FastAPI search API for [NBA History RAG](https://github.com/milanmos13/RagProgram). The
image builds the Corpus and index from `manifest.json` at build time.

`POST /search` with `{"question": "...", "k": 5}` returns ranked Chunks. Interactive
docs are at `/docs`.

Article text comes from Wikipedia and is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Code is MIT.
