import json

import pytest

from fakes import KeywordEmbedder
from nba_rag import configurations
from nba_rag import evaluate as evaluate_command
from nba_rag.eval import InvalidEvidenceError
from nba_rag.retrieval import build_baseline
from nba_rag.store import save_built

CURRY_BORN = {"article_title": "Stephen Curry", "text": "Curry was born in Akron, Ohio"}


def eval_line(qid, split, evidence=CURRY_BORN):
    return json.dumps(
        {
            "id": qid,
            "text": "Where was Curry born?",
            "question_type": "lookup",
            "source": "hand-written",
            "split": split,
            "evidence": [evidence],
        }
    )


@pytest.fixture
def workspace(corpus, tmp_path, monkeypatch):
    """A built Corpus in the data dir, a fake Embedder, and an empty results dir."""
    save_built(tmp_path / "data", corpus, build_baseline(corpus, KeywordEmbedder()).matrix)
    monkeypatch.setenv("NBA_RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NBA_RAG_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(configurations, "TutorialEmbedder", lambda model_name: KeywordEmbedder())
    return tmp_path


def write_eval_set(workspace, *lines):
    path = workspace / "eval_set.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def result_files(workspace):
    return sorted(p.name for p in (workspace / "results").glob("*.json"))


def test_scores_the_dev_split_unless_told_otherwise(workspace, capsys):
    eval_set = write_eval_set(workspace, eval_line("q1", "dev"), eval_line("q2", "test"))

    evaluate_command.main(["--eval-set", eval_set])

    assert result_files(workspace) == ["baseline_dev_test.json"]
    results = json.loads((workspace / "results" / "baseline_dev_test.json").read_text())
    assert [q["id"] for q in results["questions"]] == ["q1"]
    assert "Wrote" in capsys.readouterr().out


def test_scores_the_split_it_is_asked_for(workspace):
    eval_set = write_eval_set(workspace, eval_line("q1", "dev"), eval_line("q2", "test"))

    evaluate_command.main(["--eval-set", eval_set, "--split", "test"])
    evaluate_command.main(["--eval-set", eval_set, "--split", "all"])

    assert result_files(workspace) == [
        "baseline_all_test.json",
        "baseline_test_test.json",
    ]


def test_validate_only_checks_the_evidence_without_scoring_or_loading_the_model(
    workspace, monkeypatch, capsys
):
    def no_model(model_name):
        raise AssertionError("validation must not load the embedding model")

    monkeypatch.setattr(configurations, "TutorialEmbedder", no_model)
    eval_set = write_eval_set(workspace, eval_line("q1", "dev"), eval_line("q2", "test"))

    evaluate_command.main(["--eval-set", eval_set, "--validate-only"])

    assert not (workspace / "results").exists()
    assert "2 Eval questions" in capsys.readouterr().out


def test_validate_only_still_fails_on_broken_evidence(workspace):
    broken = {"article_title": "Stephen Curry", "text": "Curry was born in Charlotte."}
    eval_set = write_eval_set(workspace, eval_line("q1", "dev", broken))

    with pytest.raises(InvalidEvidenceError, match="q1"):
        evaluate_command.main(["--eval-set", eval_set, "--validate-only"])


def test_broken_evidence_fails_loudly_instead_of_scoring_misses(workspace):
    broken = {"article_title": "Stephen Curry", "text": "Curry was born in Charlotte."}
    eval_set = write_eval_set(workspace, eval_line("q1", "dev", broken))

    with pytest.raises(InvalidEvidenceError, match="q1"):
        evaluate_command.main(["--eval-set", eval_set])

    assert not (workspace / "results").exists()
