"""Article source backed by Wikipedia's rendered HTML for a pinned revision."""

import time
from urllib.parse import quote

import httpx

from nba_rag import __version__

REST_HTML_URL = "https://en.wikipedia.org/api/rest_v1/page/html/{title}/{revision_id}"
PROJECT_URL = "https://github.com/milanmos13/RagProgram"


class WikipediaArticleSource:
    """Fetches Article HTML politely: one request at a time with a fixed delay.

    Wikimedia rejects requests without a descriptive user agent, so ``contact``
    (an email address or URL) is required.
    """

    def __init__(
        self,
        contact: str,
        *,
        client: httpx.Client | None = None,
        delay_seconds: float = 1.0,
    ) -> None:
        self._client = client or httpx.Client(timeout=30, follow_redirects=True)
        self._user_agent = f"nba-rag/{__version__} ({PROJECT_URL}; {contact})"
        self._delay_seconds = delay_seconds

    def fetch_html(self, title: str, revision_id: int) -> str:
        url = REST_HTML_URL.format(
            title=quote(title.replace(" ", "_"), safe=""), revision_id=revision_id
        )
        response = self._client.get(url, headers={"User-Agent": self._user_agent})
        response.raise_for_status()
        time.sleep(self._delay_seconds)
        return response.text
