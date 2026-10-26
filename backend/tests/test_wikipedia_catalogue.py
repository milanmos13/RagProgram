import httpx
import pytest

from nba_rag.corpus import ManifestEntry
from nba_rag.wikipedia import WikipediaCatalogue

TEAMS_PAGE = """
<table class="wikitable sortable plainrowheaders">
<tr><th>Conference</th><th>Division</th><th>Team</th></tr>
<tr><th rowspan="2"><a href="./Eastern_Conference_(NBA)">Eastern Conference</a></th>
    <th rowspan="2"><a href="./Atlantic_Division_(NBA)">Atlantic</a></th>
    <th scope="row"><a href="./Boston_Celtics">Boston Celtics</a></th><td>Boston</td></tr>
<tr><th scope="row"><a href="./Philadelphia_76ers">Philadelphia 76ers</a></th>
    <td>Philadelphia</td></tr>
</table>
<table class="wikitable sortable"><tr><th>Teams</th></tr>
<tr><th scope="row"><a href="./Not_A_Franchise">Elsewhere</a></th></tr></table>
"""

TEAM_75_PAGE = """
<table class="wikitable"><tr><td>Italics</td><td>Denotes active player</td></tr>
<tr><td><a href="./Not_A_Player">Key</a></td></tr></table>
<table class="wikitable sortable plainrowheaders">
<tr><th>Name</th><th>Team(s)</th></tr>
<tr><th scope="row"><a href="./Kareem_Abdul-Jabbar">Kareem Abdul-Jabbar</a></th><td>Lakers</td></tr>
<tr><th scope="row"><a href="./Stephen_Curry">Stephen Curry</a></th><td>Warriors</td></tr>
<tr><th scope="row"><a href="./Dikembe_Mutombo">Dikembe Mutombo</a></th><td>Nuggets</td></tr>
</table>
"""


def catalogue(handler, requests=None) -> WikipediaCatalogue:
    def record(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        return handler(request)

    return WikipediaCatalogue(
        contact="dev@example.com",
        client=httpx.Client(transport=httpx.MockTransport(record)),
        delay_seconds=0,
    )


def test_franchises_are_the_team_links_in_the_leagues_team_table():
    requests: list[httpx.Request] = []
    pages = catalogue(lambda r: httpx.Response(200, text=TEAMS_PAGE), requests)

    assert pages.franchise_titles() == ["Boston Celtics", "Philadelphia 76ers"]
    [request] = requests
    assert request.url.path == "/api/rest_v1/page/html/National_Basketball_Association"
    assert "dev@example.com" in request.headers["user-agent"]


def test_anniversary_team_is_the_player_links_in_the_roster_table():
    requests: list[httpx.Request] = []
    pages = catalogue(lambda r: httpx.Response(200, text=TEAM_75_PAGE), requests)

    assert pages.anniversary_team_titles() == [
        "Kareem Abdul-Jabbar",
        "Stephen Curry",
        "Dikembe Mutombo",
    ]
    assert requests[0].url.path == "/api/rest_v1/page/html/NBA_75th_Anniversary_Team"


def test_a_page_without_the_expected_table_fails_loudly():
    pages = catalogue(lambda r: httpx.Response(200, text="<p>redesigned</p>"))

    with pytest.raises(ValueError, match="team table"):
        pages.franchise_titles()
    with pytest.raises(ValueError, match="roster table"):
        pages.anniversary_team_titles()


def revisions_response(request: httpx.Request) -> httpx.Response:
    titles = request.url.params["titles"].split("|")
    pages = []
    for title in titles:
        if title == "Missing page":
            pages.append({"title": title, "missing": True})
        else:
            canonical = "Los Angeles Clippers" if title == "LA Clippers" else title
            pages.append({"title": canonical, "revisions": [{"revid": len(canonical)}]})
    redirects = (
        [{"from": "LA Clippers", "to": "Los Angeles Clippers"}] if "LA Clippers" in titles else []
    )
    return httpx.Response(200, json={"query": {"redirects": redirects, "pages": pages}})


def test_current_revisions_resolve_redirects_and_keep_the_requested_order():
    requests: list[httpx.Request] = []
    pages = catalogue(revisions_response, requests)

    entries = pages.current_revisions(["Utah Jazz", "LA Clippers"])

    assert entries == [
        ManifestEntry("Utah Jazz", len("Utah Jazz")),
        ManifestEntry("Los Angeles Clippers", len("Los Angeles Clippers")),
    ]
    [request] = requests
    assert request.url.path == "/w/api.php"
    assert request.url.params["redirects"] == "1"
    assert request.url.params["prop"] == "revisions"


def test_current_revisions_are_looked_up_in_batches_of_fifty():
    requests: list[httpx.Request] = []
    pages = catalogue(revisions_response, requests)
    titles = [f"Article {i}" for i in range(120)]

    entries = pages.current_revisions(titles)

    assert [e.title for e in entries] == titles
    assert [len(r.url.params["titles"].split("|")) for r in requests] == [50, 50, 20]


def test_a_title_with_no_page_fails_loudly():
    pages = catalogue(revisions_response)

    with pytest.raises(LookupError, match="Missing page"):
        pages.current_revisions(["Utah Jazz", "Missing page"])
