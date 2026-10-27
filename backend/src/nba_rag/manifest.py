"""Generate the manifest from the Corpus inclusion rule.

The Corpus is every BAA/NBA season, every Finals, the current franchises and the NBA 75th
Anniversary Team. Each Article is pinned to its current revision so the Corpus stays
reproducible; an Article already pinned in a previous manifest keeps its pin.
"""

from typing import Protocol

from nba_rag.corpus import Manifest, ManifestEntry

FIRST_SEASON = 1946
FIRST_NBA_SEASON = 1949
"""The BAA became the NBA in 1949, which changes how season and Finals titles read."""


class Catalogue(Protocol):
    def franchise_titles(self) -> list[str]: ...

    def anniversary_team_titles(self) -> list[str]: ...

    def current_revisions(self, titles: list[str]) -> list[ManifestEntry]:
        """The current revision of each title, in order, under its canonical title."""
        ...


def generate_manifest(
    catalogue: Catalogue, *, version: str, last_season: int, previous: Manifest | None = None
) -> Manifest:
    """``last_season`` is the start year of the latest completed season (2025 for 2025–26).

    Articles in ``previous`` keep their pinned revision, so Evidence written against them
    stays valid; only new Articles are looked up.
    """
    titles = [
        *(season_title(year) for year in range(FIRST_SEASON, last_season + 1)),
        *(finals_title(year) for year in range(FIRST_SEASON, last_season + 1)),
        *catalogue.franchise_titles(),
        *catalogue.anniversary_team_titles(),
    ]
    pinned = {entry.title: entry for entry in previous.articles} if previous else {}
    new_titles = [title for title in titles if title not in pinned]
    looked_up = iter(catalogue.current_revisions(new_titles))
    articles = [pinned[title] if title in pinned else next(looked_up) for title in titles]
    return Manifest(version=version, articles=articles)


def _league(start_year: int) -> str:
    return "BAA" if start_year < FIRST_NBA_SEASON else "NBA"


def season_title(start_year: int) -> str:
    end_year = start_year + 1
    end = str(end_year) if end_year % 100 == 0 else f"{end_year % 100:02d}"
    return f"{start_year}–{end} {_league(start_year)} season"


def finals_title(season_start_year: int) -> str:
    return f"{season_start_year + 1} {_league(season_start_year)} Finals"
