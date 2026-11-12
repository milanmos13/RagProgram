import json

import pytest

from fakes import FakeLLM, KeywordEmbedder
from nba_rag import eval_tools
from nba_rag.eval import load_eval_set, load_out_of_corpus
from nba_rag.retrieval import build_baseline
from nba_rag.review import load_review_file
from nba_rag.store import save_built

BORN = "Curry was born in Akron, Ohio"


def reply(evidence: str) -> str:
    return json.dumps({"question": "Where from?", "evidence": evidence, "question_type": "lookup"})


@pytest.fixture
def workspace(corpus, tmp_path, monkeypatch):
    save_built(
        tmp_path / "data", corpus, {"tutorial": build_baseline(corpus, KeywordEmbedder()).matrix}
    )
    monkeypatch.setenv("NBA_RAG_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def test_search_prints_matching_chunks_with_their_article(workspace, capsys):
    eval_tools.main(["search", "born", "akron"])

    out = capsys.readouterr().out
    assert "Stephen Curry" in out
    assert BORN in out


def test_search_says_when_nothing_matches(workspace, capsys):
    eval_tools.main(["search", "zzzzz"])

    assert "No Chunks contain" in capsys.readouterr().out


def test_generate_writes_surviving_candidates_to_the_review_file(workspace, monkeypatch, capsys):
    good = reply(BORN)
    bad = reply("Invented sentence.")
    llm = FakeLLM(*([good] + [bad] * 50))
    monkeypatch.setattr(eval_tools, "OpenAIClient", lambda: llm)
    review = workspace / "review.jsonl"

    eval_tools.main(["generate", "--count", "6", "--out", str(review)])

    candidates = load_review_file(review)
    assert len(candidates) <= 1
    assert "rejected" in capsys.readouterr().out
    assert len(llm.prompts) == 6


def test_review_then_finalize_appends_kept_questions_to_the_eval_set(
    workspace, monkeypatch, capsys
):
    good = reply(BORN)
    llm = FakeLLM(*([good] * 50))
    monkeypatch.setattr(eval_tools, "OpenAIClient", lambda: llm)
    review = workspace / "review.jsonl"
    eval_set = workspace / "eval_set.jsonl"
    eval_tools.main(["generate", "--count", "50", "--out", str(review)])
    pending = load_review_file(review)
    assert pending
    answers = iter(["k"] + ["d"] * (len(pending) - 1))
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    eval_tools.main(["review", str(review)])
    eval_tools.main(["finalize", str(review), "--eval-set", str(eval_set)])

    [question] = load_eval_set(eval_set)
    assert question.source == "llm-generated"
    assert question.evidence[0].text == BORN


def test_split_rewrites_the_splits_of_the_eval_set(workspace, tmp_path):
    eval_set = tmp_path / "eval_set.jsonl"
    lines = [
        json.dumps(
            {
                "id": f"q{i}",
                "text": "?",
                "question_type": "lookup",
                "source": "web",
                "split": "test",
                "evidence": [{"article_title": "A", "text": "x"}],
            }
        )
        for i in range(10)
    ]
    eval_set.write_text("\n".join(lines) + "\n", encoding="utf-8")

    eval_tools.main(["split", "--eval-set", str(eval_set), "--dev-size", "3"])

    splits = [q.split for q in load_eval_set(eval_set)]
    assert splits.count("dev") == 3
    assert splits.count("test") == 7


def test_log_out_of_corpus_appends_to_the_candidate_log(tmp_path):
    log = tmp_path / "out_of_corpus.jsonl"

    eval_tools.main(
        [
            "log-out-of-corpus",
            "Who won the 2004 Slam Dunk Contest?",
            "--found-at",
            "https://example.com/t",
            "--log",
            str(log),
        ]
    )

    [candidate] = load_out_of_corpus(log)
    assert candidate.question == "Who won the 2004 Slam Dunk Contest?"
    assert candidate.found_at == "https://example.com/t"


def test_generate_refuses_to_overwrite_an_existing_review_file(workspace, monkeypatch):
    monkeypatch.setattr(eval_tools, "OpenAIClient", lambda: FakeLLM())
    review = workspace / "review.jsonl"
    review.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        eval_tools.main(["generate", "--count", "1", "--out", str(review)])

    assert review.read_text() == "keep me\n"


def test_finalize_says_how_many_kept_questions_it_skipped_for_clashing_ids(
    workspace, monkeypatch, capsys
):
    from nba_rag.eval import EvalQuestion, Evidence, save_eval_set
    from nba_rag.generate import Candidate
    from nba_rag.review import write_review_file

    review, eval_set = workspace / "review.jsonl", workspace / "eval_set.jsonl"
    evidence = [Evidence("Stephen Curry", BORN)]
    save_eval_set(eval_set, [EvalQuestion("g001", "?", "lookup", "hand-written", "dev", evidence)])
    write_review_file(
        review, [Candidate("g001", "Where?", "lookup", "llm-generated", evidence, "c", "kept")]
    )

    eval_tools.main(["finalize", str(review), "--eval-set", str(eval_set)])

    assert "skipped 1" in capsys.readouterr().out
