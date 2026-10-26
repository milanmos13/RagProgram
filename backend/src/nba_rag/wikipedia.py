"""Adapters over Wikipedia: Article HTML for pinned revisions, and the lists behind the Corpus."""

import time
from collections.abc import Callable
from urllib.parse import quote, unquote

import httpx
from bs4 import BeautifulSoup, Tag

from nba_rag import __version__
from nba_rag.corpus import ManifestEntry

REST_HTML_URL = "https://en.wikipedia.org/api/rest_v1/page/html/{title}/{revision_id}"
CURRENT_HTML_URL = "https://en.wikipedia.org/api/rest_v1/page/html/{title}"
ACTION_API_URL = "https://en.wikipedia.org/w/api.php"
PROJECT_URL = "https://github.com/milanmos13/RagProgram"
REVISION_BATCH_SIZE = 50
"""The most titles the action API answers for in one request."""


class _PoliteClient:
    """One request at a time with a fixed delay after each.

    Wikimedia rejects requests without a descriptive user agent, so ``contact``
    (an email address or URL) is required.
    """

    def __init__(
        self,
        contact: str,
        client: httpx.Client | None,
        delay_seconds: float,
        sleep: Callable[[float], None],
    ) -> None:
        self._client = client or httpx.Client(timeout=30, follow_redirects=True)
        self._user_agent = f"nba-rag/{__version__} ({PROJECT_URL}; {contact})"
        self._delay_seconds = delay_seconds
        self._sleep = sleep

    def get(self, url: str, params: dict | None = None) -> httpx.Response:
        response = self._client.get(url, params=params, headers={"User-Agent": self._user_agent})
        response.raise_for_status()
        self._sleep(self._delay_seconds)
        return response


class WikipediaArticleSource:
    """Fetches the rendered HTML of an Article's pinned revision."""

    def __init__(
        self,
        contact: str,
        *,
        client: httpx.Client | None = None,
        delay_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = _PoliteClient(contact, client, delay_seconds, sleep)

    def fetch_html(self, title: str, revision_id: int) -> str:
        url = REST_HTML_URL.format(
            title=quote(title.replace(" ", "_"), safe=""), revision_id=revision_id
        )
        return self._http.get(url).text


class WikipediaCatalogue:
    """Reads the franchise and 75th Anniversary Team lists, and looks up current revisions."""

    def __init__(
        self,
        contact: str,
        *,
        client: httpx.Client | None = None,
        delay_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = _PoliteClient(contact, client, delay_seconds, sleep)

    def franchise_titles(self) -> list[str]:
        """The 30 current franchises: the row-header links of the league's team table."""
        soup = self._current_page("National Basketball Association")
        table = soup.select_one("table.wikitable.sortable.plainrowheaders")
        if table is None:
            raise ValueError("The league article has no team table; its layout may have changed")
        return _linked_titles(table.select("th[scope=row]"))

    def anniversary_team_titles(self) -> list[str]:
        """The 76 players: the row-header links of the roster table."""
        soup = self._current_page("NBA 75th Anniversary Team")
        table = soup.select_one("table.wikitable.sortable.plainrowheaders")
        if table is None:
            raise ValueError("The 75th Anniversary Team article has no roster table")
        return _linked_titles(table.select("th[scope=row]"))

    def current_revisions(self, titles: list[str]) -> list[ManifestEntry]:
        resolved: dict[str, ManifestEntry] = {}
        for start in range(0, len(titles), REVISION_BATCH_SIZE):
            batch = titles[start : start + REVISION_BATCH_SIZE]
            resolved.update(self._revisions_of(batch))
        return [resolved[title] for title in titles]

    def _current_page(self, title: str) -> BeautifulSoup:
        url = CURRENT_HTML_URL.format(title=quote(title.replace(" ", "_"), safe=""))
        return BeautifulSoup(self._http.get(url).text, "html.parser")

    def _revisions_of(self, titles: list[str]) -> dict[str, ManifestEntry]:
        response = self._http.get(
            ACTION_API_URL,
            params={
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "redirects": 1,
                "prop": "revisions",
                "rvprop": "ids",
                "titles": "|".join(titles),
            },
        )
        query = response.json()["query"]
        # Titles can be normalized, then redirected; follow both to reach the page.
        renamed = {
            change["from"]: change["to"]
            for change in [*query.get("normalized", []), *query.get("redirects", [])]
        }
        pages = {page["title"]: page for page in query["pages"]}
        entries: dict[str, ManifestEntry] = {}
        for title in titles:
            target = title
            while target in renamed:
                target = renamed[target]
            page = pages.get(target)
            if page is None or "revisions" not in page:
                raise LookupError(f"No Wikipedia article titled {title!r}")
            entries[title] = ManifestEntry(page["title"], page["revisions"][0]["revid"])
        return entries


def _linked_titles(cells: list[Tag]) -> list[str]:
    titles = []
    for cell in cells:
        link = cell.find("a", href=True)
        if link is not None:
            titles.append(unquote(link["href"].removeprefix("./")).replace("_", " "))
    return titles
