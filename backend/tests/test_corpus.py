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
    prose = prose_chunks(corpus)

    assert [(c.id, c.section_path, c.text) for c in prose] == [
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
    for position, chunk in enumerate(prose):
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


def test_boilerplate_sections_and_navigation_boxes_contribute_no_chunks():
    corpus = build(CURRY, FINALS)

    sections = {title for chunk in corpus.chunks for title in chunk.section_path}
    assert not sections & {"See also", "References", "External links", "Notes", "Citations"}
    text = " ".join(chunk.text for chunk in corpus.chunks)
    assert "[1]" not in text
    assert "Navigation" not in text and "v t e" not in text


def test_navigation_boxes_inside_a_prose_section_are_stripped():
    corpus = build(ManifestEntry(title="Navbox Article", revision_id=444))

    assert [(c.section_path, c.text) for c in corpus.chunks] == [
        ((), "Prose in the lead."),
        (("History",), "Prose before the box."),
        (("History",), "Prose after the box."),
    ]


def test_a_chunk_never_crosses_a_section_boundary_even_when_the_budget_would_allow_it():
    corpus = build(CURRY, chunk_size=10_000)

    # Every prose paragraph is its own Chunk: nothing is merged into a neighbour in another section.
    assert [c.section_path for c in prose_chunks(corpus)] == [
        (),
        (),
        ("Early life",),
        (
            "Professional career",
            "Golden State Warriors (2009–present)",
            "2015–16 season: Unanimous MVP",
        ),
    ]


def prose_chunks(corpus):
    return [c for c in corpus.chunks if c.origin == "prose"]


def infobox_chunks(corpus):
    return [c for c in corpus.chunks if c.origin == "infobox"]


def test_infobox_becomes_field_value_lines_in_its_own_chunk():
    corpus = build(CURRY)

    (chunk,) = infobox_chunks(corpus)
    assert chunk.text.splitlines() == [
        "Stephen Curry",
        "Position: Point guard",
        "Draft: 2009 / Round: 1 / Pick: 7th overall",
        "Career highlights: 4× NBA champion; 10× NBA All-Star",
        "Teams: Golden State Warriors; Davidson Wildcats",
    ]
    assert chunk.article_title == "Stephen Curry"
    assert chunk.revision_id == 111
    assert chunk.section_path == ()
    assert chunk.id == f"Stephen Curry#{chunk.position}"


def test_every_infobox_chunk_starts_with_its_article_title_within_the_budget():
    corpus = build(CURRY, chunk_size=12)

    chunks = infobox_chunks(corpus)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.splitlines()[0] == "Stephen Curry"
        assert count_words(chunk.text) <= 12


def test_chunks_never_mix_origins_and_ids_stay_unique():
    corpus = build(CURRY, FINALS, chunk_size=12)

    prose = [c for c in corpus.chunks if c.origin == "prose"]
    assert all("Draft:" not in c.text and "Position:" not in c.text for c in prose)
    assert len(infobox_chunks(corpus)) > 1  # budget forces the infobox to split
    assert all(
        line.count(": ") >= 1 for c in infobox_chunks(corpus) for line in c.text.splitlines()[1:]
    )
    assert [c.position for c in corpus.chunks if c.article_title == "Stephen Curry"] == list(
        range(len([c for c in corpus.chunks if c.article_title == "Stephen Curry"]))
    )
    assert len({c.id for c in corpus.chunks}) == len(corpus.chunks)


def test_infobox_lines_are_never_split_across_chunks_when_they_fit():
    corpus = build(CURRY, chunk_size=12)

    lines = [line for c in infobox_chunks(corpus) for line in c.text.splitlines()[1:]]
    assert "Draft: 2009 / Round: 1 / Pick: 7th overall" in lines


def test_article_without_infobox_has_only_prose_chunks():
    corpus = build(FINALS)

    assert infobox_chunks(corpus) == []
