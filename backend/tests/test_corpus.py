import pytest

from fakes import FixtureArticleSource, count_words
from nba_rag.corpus import EmptyArticleError, Manifest, ManifestEntry, build_corpus

CURRY = ManifestEntry(title="Stephen Curry", revision_id=111)
FINALS = ManifestEntry(title="2016 NBA Finals", revision_id=222)


def build(*entries: ManifestEntry, chunk_size: int = 200):
    manifest = Manifest(version="test", articles=list(entries))
    return build_corpus(
        manifest,
        source=FixtureArticleSource(),
        count_tokens=count_words,
        chunk_size=chunk_size,
    )


def test_builds_prose_chunks_with_ids_sections_and_revisions():
    corpus = build(CURRY)

    assert [(c.id, c.section_path, c.text) for c in corpus.chunks] == [
        (
            "Stephen Curry#0",
            (),
            "Wardell Stephen Curry II (born March 14, 1988) is an American professional "
            "basketball player for the Golden State Warriors of the National Basketball "
            "Association (NBA).",
        ),
        (
            "Stephen Curry#1",
            (),
            "He is widely regarded as the greatest shooter in NBA history.",
        ),
        (
            "Stephen Curry#2",
            ("Early life",),
            "Curry was born in Akron, Ohio, while his father Dell Curry was playing for the "
            "Cleveland Cavaliers.",
        ),
        (
            "Stephen Curry#3",
            (
                "Professional career",
                "Golden State Warriors (2009–present)",
                "2015–16 season: Unanimous MVP",
            ),
            "Curry became the first unanimous MVP in league history.",
        ),
    ]
    for position, chunk in enumerate(corpus.chunks):
        assert chunk.article_title == "Stephen Curry"
        assert chunk.revision_id == 111
        assert chunk.origin == "prose"
        assert chunk.position == position


def test_splits_paragraphs_into_chunk_size_pieces_without_crossing_paragraphs():
    corpus = build(FINALS, CURRY, chunk_size=6)

    finals = [c for c in corpus.chunks if c.article_title == "2016 NBA Finals"]
    assert [(c.id, c.text) for c in finals] == [
        ("2016 NBA Finals#0", "The 2016 NBA Finals was the"),
        ("2016 NBA Finals#1", "championship series of the 2015–16 season."),
        ("2016 NBA Finals#2", "Kyrie Irving hit a go-ahead three-pointer"),
        ("2016 NBA Finals#3", "with 53 seconds left in Game"),
        ("2016 NBA Finals#4", "7."),
        ("2016 NBA Finals#5", "LeBron James blocked a layup by"),
        ("2016 NBA Finals#6", "Andre Iguodala late in the fourth"),
        ("2016 NBA Finals#7", "quarter."),
    ]
    assert finals[2].section_path == ("Game summaries", "Game 7")
    assert corpus.chunks[len(finals)].id == "Stephen Curry#0"
    assert all(c.revision_id == 222 for c in finals)


def test_article_with_no_prose_fails_loudly():
    with pytest.raises(EmptyArticleError, match="Empty Article"):
        build(ManifestEntry(title="Empty Article", revision_id=333))
