from fakes import FixtureArticleSource, count_words
from nba_rag.article_cache import CachedArticleSource
from nba_rag.corpus import Manifest, ManifestEntry, build_corpus

CURRY = ManifestEntry("Stephen Curry", 111)


def test_a_fetched_article_is_read_from_disk_the_next_time(tmp_path):
    network = FixtureArticleSource()

    first = CachedArticleSource(network, tmp_path).fetch_html("Stephen Curry", 111)
    second = CachedArticleSource(network, tmp_path).fetch_html("Stephen Curry", 111)

    assert second == first
    assert network.requests == [("Stephen Curry", 111)]


def test_each_pinned_revision_is_cached_separately(tmp_path):
    network = FixtureArticleSource()
    cached = CachedArticleSource(network, tmp_path)

    cached.fetch_html("Stephen Curry", 111)
    cached.fetch_html("Stephen Curry", 112)

    assert network.requests == [("Stephen Curry", 111), ("Stephen Curry", 112)]


def test_titles_with_slashes_and_punctuation_are_cached(tmp_path):
    network = FixtureArticleSource()
    cached = CachedArticleSource(network, tmp_path / "nested")

    network.fetch_html = lambda title, revision_id: f"<p>{title}</p>"  # type: ignore[method-assign]
    assert cached.fetch_html("AC/DC: 2015–16", 1) == "<p>AC/DC: 2015–16</p>"
    assert cached.fetch_html("AC/DC: 2015–16", 1) == "<p>AC/DC: 2015–16</p>"


def test_rebuilding_the_corpus_makes_no_network_calls(tmp_path):
    network = FixtureArticleSource()
    manifest = Manifest(version="test", articles=[CURRY])
    build_corpus(manifest, source=CachedArticleSource(network, tmp_path), count_tokens=count_words)
    requests_after_first_build = list(network.requests)

    rebuilt = build_corpus(
        manifest,
        source=CachedArticleSource(network, tmp_path),
        count_tokens=count_words,
        chunk_size=6,
    )

    assert network.requests == requests_after_first_build == [("Stephen Curry", 111)]
    assert len(rebuilt.chunks) > 4
