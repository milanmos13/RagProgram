import pytest
from fastapi.testclient import TestClient

from fakes import KeywordEmbedder
from nba_rag.corpus import Corpus
from nba_rag.retrieval import build_baseline
from nba_rag_api.app import create_app

FRONTEND = "http://localhost:3000"


@pytest.fixture
def client(corpus: Corpus) -> TestClient:
    baseline = build_baseline(corpus, KeywordEmbedder())
    return TestClient(create_app({"baseline": baseline}, allowed_origins=[FRONTEND]))


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
                "id": "Stephen Curry#2",
                "article_title": "Stephen Curry",
                "section_path": ["Early life"],
                "origin": "prose",
                "text": "Curry was born in Akron, Ohio, while his father Dell Curry was playing "
                "for the Cleveland Cavaliers.",
                "score": pytest.approx(0.6325, abs=1e-4),
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
