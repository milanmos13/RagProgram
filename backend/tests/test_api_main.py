"""The API composition root: it loads the built Corpus when the module is run."""

import runpy

import pytest
from fastapi.testclient import TestClient

from fakes import KeywordEmbedder
from nba_rag import configurations
from nba_rag.retrieval import build_baseline
from nba_rag.store import save_built

VERCEL = "https://nba-rag.vercel.app"


def load_main() -> dict:
    return runpy.run_module("nba_rag_api.main")


@pytest.fixture
def built(corpus, tmp_path, monkeypatch):
    save_built(tmp_path / "data", corpus, build_baseline(corpus, KeywordEmbedder()).matrix)
    monkeypatch.setenv("NBA_RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NBA_RAG_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.delenv("NBA_RAG_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setattr(configurations, "TutorialEmbedder", lambda model_name: KeywordEmbedder())


def preflight(client, origin):
    return client.options(
        "/search", headers={"Origin": origin, "Access-Control-Request-Method": "POST"}
    )


def test_serves_searches_over_the_built_corpus(built):
    client = TestClient(load_main()["app"])

    response = client.post("/search", json={"question": "Was Curry the unanimous MVP?", "k": 1})

    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "Stephen Curry#3"


def test_a_missing_corpus_exits_with_the_command_that_builds_it(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_RAG_DATA_DIR", str(tmp_path / "nothing-here"))

    with pytest.raises(SystemExit) as exit_info:
        load_main()

    message = str(exit_info.value)
    assert str(tmp_path / "nothing-here") in message
    assert "python -m nba_rag.build" in message


def test_only_the_local_frontend_is_allowed_by_default(built):
    client = TestClient(load_main()["app"])

    assert preflight(client, "http://localhost:3000").headers["access-control-allow-origin"] == (
        "http://localhost:3000"
    )
    assert "access-control-allow-origin" not in preflight(client, VERCEL).headers


def test_allowed_origins_come_from_a_comma_separated_environment_variable(built, monkeypatch):
    monkeypatch.setenv("NBA_RAG_ALLOWED_ORIGINS", f"http://localhost:3000,{VERCEL}")
    client = TestClient(load_main()["app"])

    assert preflight(client, VERCEL).headers["access-control-allow-origin"] == VERCEL
    assert "access-control-allow-origin" in preflight(client, "http://localhost:3000").headers
    assert "access-control-allow-origin" not in preflight(client, "https://evil.example").headers
