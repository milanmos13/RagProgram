import httpx
import pytest

from nba_rag.wikipedia import WikipediaArticleSource


def test_requests_pinned_revision_html_with_descriptive_user_agent():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="<html>pinned</html>")

    source = WikipediaArticleSource(
        contact="dev@example.com",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        delay_seconds=0,
    )

    html = source.fetch_html("2015–16 NBA season", 1373015657)

    assert html == "<html>pinned</html>"
    [request] = seen
    assert request.url == (
        "https://en.wikipedia.org/api/rest_v1/page/html/2015%E2%80%9316_NBA_season/1373015657"
    )
    user_agent = request.headers["user-agent"]
    assert user_agent.startswith("nba-rag/")
    assert "dev@example.com" in user_agent


def test_failed_fetch_raises():
    source = WikipediaArticleSource(
        contact="dev@example.com",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(403))),
        delay_seconds=0,
    )

    with pytest.raises(httpx.HTTPStatusError):
        source.fetch_html("LeBron James", 1)


def test_waits_the_delay_after_every_request():
    waits: list[float] = []
    source = WikipediaArticleSource(
        contact="dev@example.com",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="x"))),
        delay_seconds=0.5,
        sleep=waits.append,
    )

    source.fetch_html("LeBron James", 1)
    source.fetch_html("Stephen Curry", 2)

    assert waits == [0.5, 0.5]
