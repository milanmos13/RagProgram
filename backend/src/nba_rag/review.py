"""Review generated Eval questions: keep, edit or drop each one, saved in the JSONL file.

The review file holds one candidate per line with a status. Every decision is written
back immediately, so an interrupted review resumes at the first pending candidate.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from nba_rag.corpus import Corpus
from nba_rag.eval import (
    EvalQuestion,
    Evidence,
    InvalidEvidenceError,
    article_texts,
    evidence_problems,
)
from nba_rag.generate import Candidate


@dataclass(frozen=True)
class Decision:
    action: Literal["keep", "edit", "drop"]
    text: str | None = None
    evidence: list[Evidence] | None = None

    @classmethod
    def keep(cls) -> "Decision":
        return cls("keep")

    @classmethod
    def drop(cls) -> "Decision":
        return cls("drop")

    @classmethod
    def edit(cls, text: str | None = None, evidence: list[Evidence] | None = None) -> "Decision":
        return cls("edit", text, evidence)


def write_review_file(path: Path, candidates: list[Candidate]) -> None:
    lines = [json.dumps(asdict(c), ensure_ascii=False) for c in candidates]
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def load_review_file(path: Path) -> list[Candidate]:
    return [
        Candidate(**{**data, "evidence": [Evidence(**e) for e in data["evidence"]]})
        for data in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    ]


def review_candidates(path: Path, corpus: Corpus, decide: Callable[[Candidate], Decision]) -> None:
    """Ask ``decide`` about each pending candidate and record the answer in the file.

    An edit that leaves Evidence not verbatim in the Corpus raises InvalidEvidenceError
    and leaves the candidate pending.
    """
    candidates = load_review_file(path)
    texts = article_texts(corpus)
    for index, candidate in enumerate(candidates):
        if candidate.status != "pending":
            continue
        decision = decide(candidate)
        if decision.action == "drop":
            candidates[index] = replace(candidate, status="dropped")
        else:
            updated = replace(
                candidate,
                text=decision.text or candidate.text,
                evidence=decision.evidence or candidate.evidence,
                status="kept",
            )
            problems = evidence_problems(updated.to_question(), texts)
            if problems:
                raise InvalidEvidenceError("\n".join(problems))
            candidates[index] = updated
        write_review_file(path, candidates)


def kept_questions(path: Path) -> list[EvalQuestion]:
    return [c.to_question() for c in load_review_file(path) if c.status == "kept"]
