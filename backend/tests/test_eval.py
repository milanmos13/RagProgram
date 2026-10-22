from datetime import UTC, datetime

import pytest

from fakes import FixtureArticleSource, KeywordEmbedder, count_words
from nba_rag.corpus import Manifest, ManifestEntry, build_corpus
from nba_rag.eval import (
    EvalQuestion,
    Evidence,
    InvalidEvidenceError,
    evaluate,
    load_eval_set,
    load_latest_results,
    save_results,
    validate_evidence,
)
from nba_rag.retrieval import build_baseline

FINALS = "2016 NBA Finals"
CURRY = "Stephen Curry"


def question(qid, text, *evidence, question_type="lookup", split="dev") -> EvalQuestion:
    return EvalQuestion(
        id=qid,
        text=text,
        question_type=question_type,
        source="hand-written",
        split=split,
        evidence=[Evidence(article_title=title, text=span) for title, span in evidence],
    )


def six_word_chunks_of_the_finals():
    manifest = Manifest(version="test", articles=[ManifestEntry(FINALS, 222)])
    return build_corpus(
        manifest, source=FixtureArticleSource(), count_tokens=count_words, chunk_size=6
    )


# With the KeywordEmbedder (vocabulary: irving, james, curry, mvp, born), the fixture
# Corpus is ranked as follows. Chunk order: Finals#0-2, then Curry#0-3.
#   q1 "Was Curry the unanimous MVP?"  -> Curry#3 first                       (rank 1)
#   q2 "Where was Curry born?"         -> Curry#0 (1.0), Curry#2 (0.949)     (rank 2)
#   q3 no keywords: every score is 0, so Chunks keep Corpus order; Curry#3 is rank 7
#   q4 "irving" + "james": Finals#1, Finals#2 tie at 0.707, the rest score 0 in
#      Corpus order, so Finals#1 is rank 1 and Curry#1 is rank 5
EVAL_SET = [
    question(
        "q1",
        "Was Curry the unanimous MVP?",
        # Extra whitespace is ignored when matching Evidence.
        (CURRY, "Curry became  the first\nunanimous MVP in league history."),
    ),
    question("q2", "Where was Curry born?", (CURRY, "Curry was born in Akron, Ohio")),
    question(
        "q3",
        "Why is that season remembered?",
        (CURRY, "Curry became the first unanimous MVP in league history."),
        question_type="explanation",
    ),
    question(
        "q4",
        "Did Irving and James outshine the greatest shooter?",
        (FINALS, "Kyrie Irving hit a go-ahead three-pointer"),
        (CURRY, "He is widely regarded as the greatest shooter in NBA history."),
        question_type="multi-hop",
    ),
]


def test_scores_recall_and_mrr_against_hand_worked_rankings(corpus):
    results = evaluate(build_baseline(corpus, KeywordEmbedder()), EVAL_SET, split="dev")

    # First hit ranks: q1 1, q2 2, q3 7, q4 needs both spans (ranks 1 and 5).
    # recall@1  = q1                 = 1/4
    # recall@5  = q1, q2, q4         = 3/4
    # recall@10 = all four           = 4/4
    # MRR@10 uses the rank at which all Evidence has been found: q4 -> 1/5
    #           = (1 + 1/2 + 1/7 + 1/5) / 4
    assert results.question_count == 4
    assert results.metrics.recall_at_1 == pytest.approx(0.25)
    assert results.metrics.recall_at_5 == pytest.approx(0.75)
    assert results.metrics.recall_at_10 == pytest.approx(1.0)
    assert results.metrics.mrr_at_10 == pytest.approx((1 + 1 / 2 + 1 / 7 + 1 / 5) / 4)
    assert [q.evidence_ranks for q in results.questions] == [[1], [2], [7], [1, 5]]


def test_multi_hop_questions_report_all_evidence_and_any_evidence(corpus):
    results = evaluate(build_baseline(corpus, KeywordEmbedder()), EVAL_SET, split="dev")

    # Only q4 is Multi-hop; its two spans are found at ranks 1 and 5.
    multi_hop = results.multi_hop
    assert multi_hop.question_count == 1
    assert multi_hop.all_evidence.recall_at_1 == 0.0
    assert multi_hop.all_evidence.recall_at_5 == 1.0
    assert multi_hop.any_evidence.recall_at_1 == 1.0
    assert multi_hop.any_evidence.recall_at_5 == 1.0
    # All Evidence is in by rank 5; the first span is at rank 1.
    assert multi_hop.all_evidence.mrr_at_10 == pytest.approx(1 / 5)
    assert multi_hop.any_evidence.mrr_at_10 == 1.0


def test_only_questions_in_the_chosen_split_are_scored(corpus):
    held_out = question(
        "t1", "Where was Curry born?", (CURRY, "Curry was born in Akron, Ohio"), split="test"
    )

    results = evaluate(
        build_baseline(corpus, KeywordEmbedder()), [*EVAL_SET, held_out], split="test"
    )

    assert [q.id for q in results.questions] == ["t1"]
    assert results.split == "test"


def test_evidence_split_across_a_chunk_boundary_is_a_miss():
    # ADR 0001: a span that straddles two Chunks counts as a miss; no fuzzy matching.
    # At 6 words per Chunk this sentence ends "...left in Game" | "7." across two Chunks.
    small = six_word_chunks_of_the_finals()
    straddling = question("q", "Game 7", (FINALS, "with 53 seconds left in Game 7."))

    results = evaluate(build_baseline(small, KeywordEmbedder()), [straddling], split="dev")

    assert results.questions[0].evidence_ranks == [None]
    assert results.metrics.recall_at_10 == 0.0


def test_validator_accepts_evidence_found_verbatim_in_its_article(corpus):
    validate_evidence(EVAL_SET, corpus)


def test_validator_accepts_a_span_that_straddles_chunks():
    small = six_word_chunks_of_the_finals()

    validate_evidence([question("q", "Game 7", (FINALS, "left in Game 7."))], small)


def test_validator_rejects_evidence_not_found_verbatim_in_its_article(corpus):
    invented = question("bad1", "?", (CURRY, "Curry was born in Charlotte."))
    wrong_article = question("bad2", "?", (FINALS, "Curry was born in Akron, Ohio"))
    paraphrased = question("bad3", "?", (CURRY, "Curry was the first unanimous MVP."))

    with pytest.raises(InvalidEvidenceError) as error:
        validate_evidence([*EVAL_SET, invented, wrong_article, paraphrased], corpus)

    message = str(error.value)
    for qid in ("bad1", "bad2", "bad3"):
        assert qid in message
    assert "q1" not in message


VALID_LINE = (
    '{"id": "q0", "text": "?", "question_type": "lookup", "source": "web", "split": "dev", '
    '"evidence": [{"article_title": "A", "text": "x"}]}'
)


def test_eval_set_loads_from_jsonl_one_question_per_line(tmp_path):
    path = tmp_path / "eval_set.jsonl"
    path.write_text(
        '{"id": "q1", "text": "Where was Curry born?", "question_type": "lookup", '
        '"source": "hand-written", "split": "dev", '
        '"evidence": [{"article_title": "Stephen Curry", "text": "Curry was born in Akron"}]}\n'
        "\n"
        '{"id": "q2", "text": "Who hit the go-ahead three?", "question_type": "multi-hop", '
        '"source": "web", "split": "test", "evidence": ['
        '{"article_title": "2016 NBA Finals", "text": "Kyrie Irving hit"}, '
        '{"article_title": "Stephen Curry", "text": "greatest shooter"}]}\n',
        encoding="utf-8",
    )

    assert load_eval_set(path) == [
        question("q1", "Where was Curry born?", (CURRY, "Curry was born in Akron")),
        EvalQuestion(
            id="q2",
            text="Who hit the go-ahead three?",
            question_type="multi-hop",
            source="web",
            split="test",
            evidence=[
                Evidence(article_title=FINALS, text="Kyrie Irving hit"),
                Evidence(article_title=CURRY, text="greatest shooter"),
            ],
        ),
    ]


@pytest.mark.parametrize(
    "line",
    [
        '{"id": "q1", "text": "?", "question_type": "trivia", "source": "web", "split": "dev", '
        '"evidence": [{"article_title": "A", "text": "x"}]}',
        '{"id": "q1", "text": "?", "question_type": "lookup", "source": "web", "split": "train", '
        '"evidence": [{"article_title": "A", "text": "x"}]}',
        '{"id": "q1", "text": "?", "question_type": "lookup", "source": "web", "split": "dev", '
        '"evidence": []}',
        '{"id": "q1", "text": "?", "question_type": "lookup", "source": "web", "split": "dev"}',
    ],
)
def test_malformed_eval_questions_are_rejected_with_their_line(tmp_path, line):
    path = tmp_path / "eval_set.jsonl"
    path.write_text(VALID_LINE + "\n" + line + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_eval_set(path)


def test_duplicate_eval_question_ids_are_rejected(tmp_path):
    path = tmp_path / "eval_set.jsonl"
    path.write_text(VALID_LINE + "\n" + VALID_LINE + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate id q0"):
        load_eval_set(path)


def test_split_all_scores_every_question(corpus):
    held_out = question("t1", "Where was Curry born?", (CURRY, "Curry was born"), split="test")

    results = evaluate(
        build_baseline(corpus, KeywordEmbedder()), [*EVAL_SET, held_out], split="all"
    )

    assert [q.id for q in results.questions] == ["q1", "q2", "q3", "q4", "t1"]


def test_results_file_records_the_run_and_the_latest_run_is_served(corpus, tmp_path):
    baseline = build_baseline(corpus, KeywordEmbedder())
    dev = evaluate(baseline, EVAL_SET, split="dev")
    everything = evaluate(baseline, EVAL_SET, split="all")

    first = save_results(dev, tmp_path, created_at=datetime(2026, 10, 1, tzinfo=UTC))
    save_results(everything, tmp_path, created_at=datetime(2026, 10, 2, tzinfo=UTC))
    latest = load_latest_results(tmp_path)

    assert first.name == "baseline_dev_test.json"
    assert latest["configuration"] == "baseline"
    assert latest["split"] == "all"
    assert latest["manifest_version"] == "test"
    assert latest["created_at"] == "2026-10-02T00:00:00+00:00"
    assert latest["metrics"]["recall_at_5"] == pytest.approx(0.75)
    assert latest["multi_hop"]["any_evidence"]["recall_at_1"] == 1.0
    assert latest["multi_hop"]["all_evidence"]["mrr_at_10"] == pytest.approx(1 / 5)
    assert latest["questions"][3]["id"] == "q4"
    assert latest["questions"][3]["evidence_ranks"] == [1, 5]
    assert latest["questions"][3]["retrieved"][0] == "2016 NBA Finals#1"


def test_latest_results_is_none_before_any_eval_run(tmp_path):
    assert load_latest_results(tmp_path) is None
    assert load_latest_results(tmp_path / "missing") is None


def test_all_evidence_mrr_is_zero_when_any_span_is_never_found(corpus):
    # The second span names the wrong Article, so no Chunk can match it.
    half_found = question(
        "h1",
        "Was Curry the unanimous MVP?",
        (CURRY, "Curry became the first unanimous MVP in league history."),
        (FINALS, "Curry became the first unanimous MVP in league history."),
        question_type="multi-hop",
    )

    results = evaluate(build_baseline(corpus, KeywordEmbedder()), [half_found], split="dev")

    assert results.questions[0].evidence_ranks == [1, None]
    assert results.metrics.mrr_at_10 == 0.0
    assert results.multi_hop.all_evidence.mrr_at_10 == 0.0
    assert results.multi_hop.any_evidence.mrr_at_10 == 1.0
