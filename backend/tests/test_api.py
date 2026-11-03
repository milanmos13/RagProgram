from datetime import UTC, datetime
from unittest.mock import ANY

import pytest
from fastapi.testclient import TestClient

from fakes import KeywordEmbedder
from nba_rag.corpus import Corpus
from nba_rag.eval import EvalQuestion, Evidence, evaluate, save_results
from nba_rag.retrieval import build_baseline
from nba_rag_api.app import create_app

FRONTEND = "http://localhost:3000"


@pytest.fixture
def results_dir(tmp_path):
    return tmp_path / "results"


@pytest.fixture
def client(corpus: Corpus, results_dir) -> TestClient:
    baseline = build_baseline(corpus, KeywordEmbedder())
    return TestClient(
        create_app({"baseline": baseline}, allowed_origins=[FRONTEND], results_dir=results_dir)
    )


def test_search_returns_ranked_chunks_in_the_agreed_shape(client):
    response = client.post(
        "/search",
        json={"question": "Was Curry the unanimous MVP?", "configuration": "baseline", "k": 2},
    )

    assert response.status_code == 200
    assert response.json() == {
        "configuration": "baseline",
        "results": [
            {
                "id": "Stephen Curry#3",
                "article_title": "Stephen Curry",
                "section_path": [
                    "Professional career",
                    "Golden State Warriors (2009–present)",
                    "2015–16 season: Unanimous MVP",
                ],
                "origin": "prose",
                "text": "Curry became the first unanimous MVP in league history.",
                "score": pytest.approx(1.0),
                "revision_url": "https://en.wikipedia.org/w/index.php?title=Stephen_Curry&oldid=111",
            },
            {
                "id": "Stephen Curry#4",
                "article_title": "Stephen Curry",
                "section_path": [],
                "origin": "infobox",
                "text": ANY,
                "score": pytest.approx(0.7071, abs=1e-4),
                "revision_url": "https://en.wikipedia.org/w/index.php?title=Stephen_Curry&oldid=111",
            },
        ],
    }


def test_search_defaults_to_baseline_and_five_results(client):
    response = client.post("/search", json={"question": "Curry"})

    assert response.status_code == 200
    assert response.json()["configuration"] == "baseline"
    assert len(response.json()["results"]) == 5


def test_unknown_configuration_is_not_found(client):
    response = client.post("/search", json={"question": "Curry", "configuration": "nope"})

    assert response.status_code == 404


@pytest.mark.parametrize("body", [{"question": "   "}, {"question": "Curry", "k": 0}, {}])
def test_invalid_search_requests_are_rejected(client, body):
    assert client.post("/search", json=body).status_code == 422


def test_cors_allows_only_the_frontend_origin(client):
    allowed = client.options(
        "/search",
        headers={"Origin": FRONTEND, "Access-Control-Request-Method": "POST"},
    )
    blocked = client.options(
        "/search",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )

    assert allowed.headers["access-control-allow-origin"] == FRONTEND
    assert "access-control-allow-origin" not in blocked.headers


def test_cors_allow_list_tolerates_how_origins_are_typed_into_settings(corpus, tmp_path):
    vercel = "https://nba-rag.vercel.app"
    app = create_app(
        {"baseline": build_baseline(corpus, KeywordEmbedder())},
        allowed_origins=[f" {vercel}/", "http://localhost:3000", ""],
        results_dir=tmp_path,
    )
    client = TestClient(app)

    response = client.post("/search", json={"question": "Curry"}, headers={"Origin": vercel})
    blocked = client.post(
        "/search", json={"question": "Curry"}, headers={"Origin": "https://nba-rag-fork.vercel.app"}
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == vercel
    assert "access-control-allow-origin" not in blocked.headers


def test_results_serves_the_latest_eval_run(client, corpus, results_dir):
    mvp = EvalQuestion(
        id="q1",
        text="Was Curry the unanimous MVP?",
        question_type="lookup",
        source="hand-written",
        split="dev",
        evidence=[Evidence("Stephen Curry", "Curry became the first unanimous MVP")],
    )
    results = evaluate(build_baseline(corpus, KeywordEmbedder()), [mvp], split="dev")
    save_results(results, results_dir, created_at=datetime(2026, 10, 21, tzinfo=UTC))

    response = client.get("/results")

    assert response.status_code == 200
    body = response.json()
    assert {k: body[k] for k in ("configuration", "split", "manifest_version", "created_at")} == {
        "configuration": "baseline",
        "split": "dev",
        "manifest_version": "test",
        "created_at": "2026-10-21T00:00:00+00:00",
    }
    assert body["question_count"] == 1
    assert body["metrics"] == {
        "recall_at_1": 1.0,
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "mrr_at_10": 1.0,
    }
    assert body["multi_hop"]["question_count"] == 0
    assert body["questions"][0]["evidence_ranks"] == [1]


def test_results_is_not_found_before_any_eval_run(client):
    assert client.get("/results").status_code == 404


def test_search_reports_infobox_origin_for_infobox_chunks(client):
    response = client.post("/search", json={"question": "Curry", "k": 10})

    by_origin = {r["origin"]: r for r in response.json()["results"]}
    assert set(by_origin) == {"prose", "infobox"}
    assert "Draft: 2009 / Round: 1 / Pick: 7th overall" in by_origin["infobox"]["text"]
