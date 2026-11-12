import json

import pytest

from nba_rag.eval import EvalQuestion, Evidence, InvalidEvidenceError
from nba_rag.generate import Candidate
from nba_rag.review import Decision, load_review_file, review_candidates, write_review_file

BORN = "Curry was born in Akron, Ohio"


def candidate(cid, text="Where?", evidence=BORN) -> Candidate:
    return Candidate(
        id=cid,
        text=text,
        question_type="lookup",
        source="llm-generated",
        evidence=[Evidence("Stephen Curry", evidence)],
        chunk_id="Stephen Curry#0",
        status="pending",
    )


def test_review_file_round_trips(tmp_path):
    path = tmp_path / "review.jsonl"
    candidates = [candidate("g001"), candidate("g002")]

    write_review_file(path, candidates)

    assert load_review_file(path) == candidates
    assert len(path.read_text().splitlines()) == 2


def test_keep_edit_and_drop_are_recorded_in_the_file(corpus, tmp_path):
    path = tmp_path / "review.jsonl"
    write_review_file(path, [candidate("g001"), candidate("g002"), candidate("g003")])
    decisions = {
        "g001": Decision.keep(),
        "g002": Decision.edit(text="Where is his hometown?"),
        "g003": Decision.drop(),
    }

    review_candidates(path, corpus, lambda c: decisions[c.id])

    reviewed = {c.id: c for c in load_review_file(path)}
    assert reviewed["g001"].status == "kept"
    assert reviewed["g002"].status == "kept"
    assert reviewed["g002"].text == "Where is his hometown?"
    assert reviewed["g003"].status == "dropped"


def test_review_resumes_and_only_asks_about_pending_candidates(corpus, tmp_path):
    path = tmp_path / "review.jsonl"
    write_review_file(path, [candidate("g001"), candidate("g002")])
    asked: list[str] = []

    def decide(c):
        asked.append(c.id)
        if c.id == "g002":
            raise KeyboardInterrupt
        return Decision.keep()

    with pytest.raises(KeyboardInterrupt):
        review_candidates(path, corpus, decide)
    assert [c.status for c in load_review_file(path)] == ["kept", "pending"]

    asked.clear()
    review_candidates(path, corpus, lambda c: asked.append(c.id) or Decision.drop())

    assert asked == ["g002"]


def test_an_edit_that_breaks_the_evidence_is_refused(corpus, tmp_path):
    path = tmp_path / "review.jsonl"
    write_review_file(path, [candidate("g001")])

    with pytest.raises(InvalidEvidenceError):
        review_candidates(
            path, corpus, lambda c: Decision.edit(evidence=[Evidence("Stephen Curry", "Made up.")])
        )

    assert load_review_file(path)[0].status == "pending"


def test_kept_candidates_become_eval_questions_without_a_split_decision(corpus, tmp_path):
    path = tmp_path / "review.jsonl"
    write_review_file(path, [candidate("g001"), candidate("g002")])
    review_candidates(
        path, corpus, lambda c: Decision.keep() if c.id == "g001" else Decision.drop()
    )

    from nba_rag.review import kept_questions

    [question] = kept_questions(path)
    assert isinstance(question, EvalQuestion)
    assert question.id == "g001"
    assert question.source == "llm-generated"
    assert question.split == "test"
    assert json.loads(json.dumps(question.text))


def test_blank_lines_in_a_hand_edited_review_file_are_skipped(tmp_path):
    path = tmp_path / "review.jsonl"
    write_review_file(path, [candidate("g001")])
    path.write_text("\n" + path.read_text() + "\n\n", encoding="utf-8")

    assert [c.id for c in load_review_file(path)] == ["g001"]
