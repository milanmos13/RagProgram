"""Thin HTTP layer: translates requests into nba_rag calls and nothing more."""

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StringConstraints

from nba_rag.corpus import Origin
from nba_rag.eval import load_latest_results
from nba_rag.retrieval import Configuration, ScoredChunk, retrieve


class SearchRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    configuration: str = "baseline"
    k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    id: str
    article_title: str
    section_path: list[str]
    origin: Origin
    text: str
    score: float
    revision_url: str

    @classmethod
    def from_scored(cls, scored: ScoredChunk) -> "SearchResult":
        chunk = scored.chunk
        return cls(
            id=chunk.id,
            article_title=chunk.article_title,
            section_path=list(chunk.section_path),
            origin=chunk.origin,
            text=chunk.text,
            score=scored.score,
            revision_url=chunk.revision_url,
        )


class SearchResponse(BaseModel):
    configuration: str
    results: list[SearchResult]


def create_app(
    configurations: dict[str, Configuration],
    *,
    allowed_origins: list[str],
    results_dir: Path,
) -> FastAPI:
    app = FastAPI(title="NBA History RAG")
    app.add_middleware(
        CORSMiddleware,
        # Browsers send Origin without a trailing slash; settings are often pasted with one.
        allow_origins=[o.strip().rstrip("/") for o in allowed_origins if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.post("/search")
    def search(request: SearchRequest) -> SearchResponse:
        configuration = configurations.get(request.configuration)
        if configuration is None:
            raise HTTPException(404, f"Unknown configuration: {request.configuration}")
        results = retrieve(configuration, request.question, request.k)
        return SearchResponse(
            configuration=configuration.name,
            results=[SearchResult.from_scored(r) for r in results],
        )

    @app.get("/results")
    def results() -> dict:
        latest = load_latest_results(results_dir)
        if latest is None:
            raise HTTPException(404, "No eval results yet")
        return latest

    return app
