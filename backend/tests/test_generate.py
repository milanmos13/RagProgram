import json

import pytest

from fakes import FakeLLM
from nba_rag.generate import article_type, generate_candidates, sample_chunks

BORN = "Curry was born in Akron, Ohio"


def reply(question, evidence, question_type="lookup"):
    return json.dumps({"question": question, "evidence": evidence, "question_type": question_type})


def test_article_types_come_from_the_title():
    assert article_type("2015–16 NBA season") == "season"
    assert article_type("1949–50 BAA season") == "season"
    assert article_type("2016 NBA Finals") == "finals"
    assert article_type("Stephen Curry") == "other"
    assert article_type("Boston Celtics") == "other"


def test_sampling_covers_every_article_type_and_origin(corpus):
    strata = {(article_type(c.article_title), c.origin) for c in corpus.chunks}

    sample = sample_chunks(corpus, len(strata), seed=3)

    assert {(article_type(c.article_title), c.origin) for c in sample} == strata


def test_sampling_is_reproducible_and_never_repeats_a_chunk(corpus):
    sample = sample_chunks(corpus, 5, seed=3)

    assert sample == sample_chunks(corpus, 5, seed=3)
    assert len({c.id for c in sample}) == 5


def test_sampling_everything_returns_every_chunk_once(corpus):
    assert sorted(c.id for c in sample_chunks(corpus, 10_000, seed=1)) == sorted(
        c.id for c in corpus.chunks
    )


def test_verbatim_evidence_becomes_a_candidate_from_the_llm_generated_source(corpus):
    chunk = next(c for c in corpus.chunks if BORN in c.text)
    llm = FakeLLM(reply("Which Ohio city is the Warriors star's hometown?", BORN))

    result = generate_candidates(corpus, llm, [chunk])

    [candidate] = result.candidates
    assert candidate.text == "Which Ohio city is the Warriors star's hometown?"
    assert candidate.question_type == "lookup"
    assert candidate.source == "llm-generated"
    assert candidate.evidence[0].article_title == "Stephen Curry"
    assert candidate.evidence[0].text == BORN
    assert candidate.chunk_id == chunk.id
    assert result.rejected == []


def test_prompt_shows_the_chunk_and_asks_for_paraphrase_and_nicknames(corpus):
    chunk = corpus.chunks[0]
    llm = FakeLLM(reply("q", "nope"))

    generate_candidates(corpus, llm, [chunk])

    [prompt] = llm.prompts
    assert chunk.text in prompt
    assert chunk.article_title in prompt
    assert "nickname" in prompt.lower()
    assert "paraphrase" in prompt.lower()


def test_non_verbatim_evidence_is_rejected_automatically(corpus):
    chunk = next(c for c in corpus.chunks if BORN in c.text)
    llm = FakeLLM(reply("Where was he born?", "Curry was born in Cleveland, Ohio."))

    result = generate_candidates(corpus, llm, [chunk])

    assert result.candidates == []
    [rejection] = result.rejected
    assert rejection.chunk_id == chunk.id
    assert "Cleveland" in rejection.reason


@pytest.mark.parametrize(
    "bad_reply",
    ["not json at all", json.dumps({"question": "q"}), reply("q", BORN, "multi-hop")],
)
def test_malformed_replies_are_rejected_not_crashed_on(corpus, bad_reply):
    chunk = next(c for c in corpus.chunks if BORN in c.text)

    result = generate_candidates(corpus, FakeLLM(bad_reply), [chunk])

    assert result.candidates == []
    assert len(result.rejected) == 1


def test_candidate_ids_continue_from_the_start_number(corpus):
    chunks = [c for c in corpus.chunks if BORN in c.text]
    llm = FakeLLM(reply("q?", BORN))

    [candidate] = generate_candidates(corpus, llm, chunks[:1], first_number=41).candidates

    assert candidate.id == "g041"
