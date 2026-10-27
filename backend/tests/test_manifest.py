from fakes import FakeCatalogue
from nba_rag.corpus import Manifest, ManifestEntry
from nba_rag.manifest import generate_manifest

FRANCHISES = ["Boston Celtics", "Golden State Warriors"]
ANNIVERSARY_TEAM = ["LeBron James", "Stephen Curry"]


def generate(**overrides):
    catalogue = FakeCatalogue(FRANCHISES, ANNIVERSARY_TEAM)
    manifest = generate_manifest(
        catalogue, version="full-test", last_season=overrides.pop("last_season", 2025), **overrides
    )
    return manifest, catalogue


def test_includes_every_season_every_finals_every_franchise_and_the_anniversary_team():
    manifest, _ = generate()

    titles = [entry.title for entry in manifest.articles]
    assert manifest.version == "full-test"
    assert len(titles) == 80 + 80 + len(FRANCHISES) + len(ANNIVERSARY_TEAM)
    assert len(set(titles)) == len(titles)
    assert set(FRANCHISES + ANNIVERSARY_TEAM) <= set(titles)


def test_names_seasons_and_finals_as_wikipedia_does_across_the_baa_nba_and_century_changes():
    manifest, _ = generate()

    titles = {entry.title for entry in manifest.articles}
    assert {
        "1946–47 BAA season",
        "1948–49 BAA season",
        "1949–50 NBA season",
        "1999–2000 NBA season",
        "2025–26 NBA season",
        "1947 BAA Finals",
        "1949 BAA Finals",
        "1950 NBA Finals",
        "2026 NBA Finals",
    } <= titles
    assert "2026–27 NBA season" not in titles
    assert "2027 NBA Finals" not in titles


def test_pins_the_current_revision_of_every_article():
    manifest, catalogue = generate()

    assert sorted(catalogue.revisions_requested) == sorted(e.title for e in manifest.articles)
    pinned = {entry.title: entry.revision_id for entry in manifest.articles}
    assert pinned["1946–47 BAA season"] == 1000 + catalogue.revisions_requested.index(
        "1946–47 BAA season"
    )


def test_articles_already_pinned_keep_their_revision_and_are_not_looked_up_again():
    previous = Manifest(version="older", articles=[ManifestEntry("Stephen Curry", 111)])

    manifest, catalogue = generate(previous=previous)

    pinned = {entry.title: entry.revision_id for entry in manifest.articles}
    assert pinned["Stephen Curry"] == 111
    assert "Stephen Curry" not in catalogue.revisions_requested
    assert "LeBron James" in catalogue.revisions_requested
