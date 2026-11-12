from nba_rag.textsearch import search_text


def test_finds_chunks_containing_every_word_ignoring_case_and_whitespace(corpus):
    hits = search_text(corpus, "BORN   akron")

    assert hits
    assert all("born" in c.text.lower() and "akron" in c.text.lower() for c in hits)
    assert {c.article_title for c in hits} == {"Stephen Curry"}


def test_returns_nothing_when_a_word_is_missing(corpus):
    assert search_text(corpus, "akron zzzzz") == []


def test_limits_hits_and_keeps_corpus_order(corpus):
    hits = search_text(corpus, "the", limit=3)

    assert len(hits) == 3
    assert hits == [c for c in corpus.chunks if c in hits]
