from nba_rag.eval import OutOfCorpusCandidate, load_out_of_corpus, log_out_of_corpus


def test_logged_candidates_round_trip_in_order(tmp_path):
    path = tmp_path / "out_of_corpus.jsonl"
    first = OutOfCorpusCandidate(
        question="Who won the 2004 Slam Dunk Contest?",
        found_at="https://example.com/thread",
        suggested_article="2004 NBA All-Star Game",
        note="no Slam Dunk Contest articles in the Corpus",
    )
    second = OutOfCorpusCandidate(
        question="Who is Larry Bird's coach?", found_at="", suggested_article="", note=""
    )

    log_out_of_corpus(path, first)
    log_out_of_corpus(path, second)

    assert load_out_of_corpus(path) == [first, second]


def test_missing_log_is_empty(tmp_path):
    assert load_out_of_corpus(tmp_path / "none.jsonl") == []
