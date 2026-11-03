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


def _source_answering(responses: list[httpx.Response | Exception], waits: list[float]):
    """A source whose requests get ``responses`` in order, raising any that are exceptions."""
    seen: list[httpx.Request] = []
    remaining = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        response = next(remaining)
        if isinstance(response, Exception):
            raise response
        return response

    source = WikipediaArticleSource(
        contact="dev@example.com",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        delay_seconds=0.5,
        sleep=waits.append,
    )
    return source, seen


def test_a_throttled_fetch_waits_as_long_as_the_server_asks_then_retries():
    waits: list[float] = []
    source, seen = _source_answering(
        [httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, text="ok")], waits
    )

    assert source.fetch_html("LeBron James", 1) == "ok"
    assert len(seen) == 2
    assert waits == [7, 0.5]


def test_temporary_failures_back_off_longer_each_time_without_a_retry_after():
    waits: list[float] = []
    source, _ = _source_answering(
        [httpx.Response(429), httpx.Response(503), httpx.Response(200, text="ok")], waits
    )

    assert source.fetch_html("LeBron James", 1) == "ok"
    assert waits == [5, 10, 0.5]


def test_a_retry_after_given_as_a_date_falls_back_to_backing_off():
    waits: list[float] = []
    source, _ = _source_answering(
        [
            httpx.Response(429, headers={"Retry-After": "Sun, 01 Nov 2026 06:00:00 GMT"}),
            httpx.Response(200, text="ok"),
        ],
        waits,
    )

    source.fetch_html("LeBron James", 1)

    assert waits == [5, 0.5]


def test_gives_up_and_raises_when_the_throttling_never_stops():
    waits: list[float] = []
    source, seen = _source_answering([httpx.Response(429) for _ in range(5)], waits)

    with pytest.raises(httpx.HTTPStatusError):
        source.fetch_html("LeBron James", 1)

    assert len(seen) == 5


def test_a_permanent_failure_is_not_retried():
    waits: list[float] = []
    source, seen = _source_answering([httpx.Response(404)], waits)

    with pytest.raises(httpx.HTTPStatusError):
        source.fetch_html("LeBron James", 1)

    assert len(seen) == 1
    assert waits == []


def test_a_dropped_connection_backs_off_and_retries():
    waits: list[float] = []
    source, seen = _source_answering(
        [httpx.RemoteProtocolError("peer closed connection"), httpx.Response(200, text="ok")],
        waits,
    )

    assert source.fetch_html("LeBron James", 1) == "ok"
    assert len(seen) == 2
    assert waits == [5, 0.5]


def test_gives_up_and_raises_when_connections_keep_dropping():
    waits: list[float] = []
    source, seen = _source_answering([httpx.ReadTimeout("timed out") for _ in range(5)], waits)

    with pytest.raises(httpx.ReadTimeout):
        source.fetch_html("LeBron James", 1)

    assert len(seen) == 5
