# NBA History RAG

Question answering over a fixed, reproducible Corpus of pinned NBA-history Wikipedia
Articles, with core retrieval built from scratch and judged by an Eval set.

> **Status: tracer slice.** The Corpus is 5 Articles (prose only) searched with the
> tutorial's baseline embeddings. See the [spec](https://github.com/milanmos13/RagProgram/issues/1)
> for where this is heading; domain terms are defined in [CONTEXT.md](CONTEXT.md).

## Layout

```
backend/
  manifest.json      Articles and pinned revision IDs: the only Corpus artifact in git
  src/nba_rag/       the library: Corpus building, Embedders, retrieval
  eval_set.jsonl     the Eval set
  results/           eval results files, one per configuration, split and manifest
  src/nba_rag_api/   thin FastAPI layer (POST /search, GET /results)
  tests/             pytest, with saved HTML fixtures and a fake Embedder
frontend/            Next.js (App Router, TypeScript, Tailwind, shadcn/ui)
docker-compose.yml   development stack
```

## Running locally

Requires Docker.

```bash
# One-off: fetch the manifest's Articles, chunk and embed them into the `data` volume.
# Wikimedia requires a contact address in the user agent.
NBA_RAG_CONTACT=you@example.com docker compose run --rm backend python -m nba_rag.build

# Backend on :8000 (auto-reload), frontend on :3000 (dev server).
docker compose up
```

Search directly:

```bash
curl -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"question": "Who blocked Andre Iguodala in Game 7?", "k": 3}'
```

## Evaluating retrieval

`backend/eval_set.jsonl` holds the Eval set: one Eval question per line, with its
question type, source, split and verbatim Evidence spans. A retrieved Chunk is a hit
when it contains an Evidence span after whitespace normalization; a span split across
two Chunks counts as a miss ([ADR 0001](docs/adr/0001-evidence-not-chunk-ids.md)).

```bash
# Score a configuration on a split (dev, test or all) and write backend/results/.
docker compose run --rm backend python -m nba_rag.evaluate --configuration baseline --split all
```

The command first checks that every Evidence span is still in the built Corpus. Results
files are committed; the API serves the latest at `GET /results` and the frontend shows
it at `/results`. Commit new results before deploying, because the production image
ships the committed files.

## Tests

```bash
cd backend && pip install -e ".[dev]" && pytest   # fast suite, no model needed
pip install -e ".[ml]" && pytest -m slow          # real bge-small model
cd frontend && npm ci && npm test                  # Vitest + React Testing Library + MSW
```

CI runs ruff, eslint, tsc and the unit tests on every push, and checks that the
production image builds and serves a search offline.

## Deploying

The backend runs on Google Cloud Run and the frontend on Vercel. See
[docs/deploy.md](docs/deploy.md).

## Baseline retrieval

Row 1 of the planned experiment table reproduces the original tutorial exactly: BGE
embeddings with mean pooling, no normalization and no query instruction, ranked by
cosine similarity. Chunks are built by greedy word accumulation up to 200 tokens within a
paragraph, with no overlap.

## License

Code is MIT. Article text comes from Wikipedia and is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Originally based on [Building Retrieval Augmented Generation (RAG) From
Scratch](https://medium.com/red-buffer/building-retrieval-augmented-generation-rag-from-scratch-74c1cd7ae2c0);
the generic version lives in git history.
