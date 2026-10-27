"""The Eval set and scoring a configuration against it (see ADR 0001)."""

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args

from nba_rag.corpus import Chunk, Corpus
from nba_rag.retrieval import Configuration, retrieve

QuestionType = Literal["lookup", "multi-hop", "explanation"]
Source = Literal["llm-generated", "hand-written", "web"]
Split = Literal["dev", "test"]
SplitChoice = Split | Literal["all"]

DEPTH = 10
"""How many Chunks are retrieved per Eval question: the largest k we report."""


@dataclass(frozen=True)
class Evidence:
    """A verbatim span from a named Article that supports an Eval question's answer."""

    article_title: str
    text: str


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    text: str
    question_type: QuestionType
    source: Source
    split: Split
    evidence: list[Evidence]


@dataclass(frozen=True)
class QuestionResult:
    id: str
    text: str
    question_type: QuestionType
    source: Source
    split: Split
    evidence_ranks: list[int | None]
    """Rank (1-based) of the first retrieved Chunk containing each Evidence span."""
    retrieved: list[str]
    """IDs of the retrieved Chunks, best first."""


@dataclass(frozen=True)
class Metrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float


@dataclass(frozen=True)
class MultiHopMetrics:
    """Multi-hop scored two ways, to tell "found half" from "found nothing"."""

    question_count: int
    all_evidence: Metrics
    any_evidence: Metrics


@dataclass(frozen=True)
class EvalResults:
    configuration: str
    split: SplitChoice
    manifest_version: str
    question_count: int
    metrics: Metrics
    """A question counts as found at the rank by which all its Evidence has been
    retrieved; both recall@k and MRR@10 use that rank."""
    multi_hop: MultiHopMetrics
    questions: list[QuestionResult]


def load_eval_set(path: Path) -> list[EvalQuestion]:
    """Read the Eval set: JSONL, one Eval question per line. Blank lines are skipped."""
    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            question = _parse_question(json.loads(line))
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path} line {number}: {error!r}") from error
        if question.id in seen:
            raise ValueError(f"{path} line {number}: duplicate id {question.id}")
        seen.add(question.id)
        questions.append(question)
    return questions


def _parse_question(data: dict) -> EvalQuestion:
    for field, allowed in (
        ("question_type", get_args(QuestionType)),
        ("source", get_args(Source)),
        ("split", get_args(Split)),
    ):
        if data[field] not in allowed:
            raise ValueError(f"{field} must be one of {allowed}, not {data[field]!r}")
    evidence = [
        Evidence(article_title=e["article_title"], text=e["text"]) for e in data["evidence"]
    ]
    if not evidence:
        raise ValueError("an Eval question needs at least one Evidence span")
    return EvalQuestion(
        id=data["id"],
        text=data["text"],
        question_type=data["question_type"],
        source=data["source"],
        split=data["split"],
        evidence=evidence,
    )


_SENTENCE_BREAK = re.compile(r"[a-z0-9)][.!?] [A-Z]")
"""A sentence end followed by a new sentence. Initials like "U.S. Army" don't match."""


class InvalidEvidenceError(ValueError):
    """Evidence that isn't verbatim in its named Article, e.g. after a revision change."""


def validate_evidence(eval_set: Sequence[EvalQuestion], corpus: Corpus) -> None:
    """Check every Evidence span appears verbatim (up to whitespace) in its Article.

    Article text is rebuilt from its Chunks, so a span straddling two Chunks is still
    valid Evidence; it just can't be retrieved as a hit (ADR 0001). A span running over
    more than one sentence is rejected: that is how it would cross a paragraph, where it
    would pass the substring check on the joined text yet never fit in any Chunk.
    """
    articles: dict[str, list[str]] = {}
    for chunk in sorted(corpus.chunks, key=lambda c: c.position):
        articles.setdefault(chunk.article_title, []).append(normalize_whitespace(chunk.text))
    article_text = {title: " ".join(texts) for title, texts in articles.items()}

    problems = [
        f"{q.id}: {e.article_title!r} does not contain {e.text!r}"
        for q in eval_set
        for e in q.evidence
        if normalize_whitespace(e.text) not in article_text.get(e.article_title, "")
    ] + [
        f"{q.id}: Evidence spans more than one sentence: {e.text!r}"
        for q in eval_set
        for e in q.evidence
        if _SENTENCE_BREAK.search(normalize_whitespace(e.text))
    ]
    if problems:
        raise InvalidEvidenceError("Evidence not found in the Corpus:\n" + "\n".join(problems))


def evaluate(
    configuration: Configuration, eval_set: Sequence[EvalQuestion], split: SplitChoice
) -> EvalResults:
    questions = [_score(configuration, q) for q in eval_set if split in ("all", q.split)]
    return EvalResults(
        configuration=configuration.name,
        split=split,
        manifest_version=configuration.manifest_version,
        question_count=len(questions),
        metrics=_metrics(questions, _rank_all_found),
        multi_hop=_multi_hop_metrics(questions),
        questions=questions,
    )


def _score(configuration: Configuration, question: EvalQuestion) -> QuestionResult:
    retrieved = [r.chunk for r in retrieve(configuration, question.text, DEPTH)]
    return QuestionResult(
        id=question.id,
        text=question.text,
        question_type=question.question_type,
        source=question.source,
        split=question.split,
        evidence_ranks=[_first_rank(retrieved, e) for e in question.evidence],
        retrieved=[c.id for c in retrieved],
    )


def _first_rank(retrieved: list[Chunk], evidence: Evidence) -> int | None:
    span = normalize_whitespace(evidence.text)
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk.article_title == evidence.article_title and span in normalize_whitespace(
            chunk.text
        ):
            return rank
    return None


def _multi_hop_metrics(questions: list[QuestionResult]) -> MultiHopMetrics:
    multi_hop = [q for q in questions if q.question_type == "multi-hop"]
    return MultiHopMetrics(
        question_count=len(multi_hop),
        all_evidence=_metrics(multi_hop, _rank_all_found),
        any_evidence=_metrics(multi_hop, _rank_any_found),
    )


def _metrics(
    questions: list[QuestionResult], rank: Callable[[QuestionResult], int | None]
) -> Metrics:
    """recall@k and MRR@10 from the rank at which each question counts as found."""
    ranks = [rank(q) for q in questions]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def recall(k: int) -> float:
        return mean([float(r is not None and r <= k) for r in ranks])

    return Metrics(
        recall_at_1=recall(1),
        recall_at_5=recall(5),
        recall_at_10=recall(10),
        mrr_at_10=mean([1 / r if r is not None and r <= DEPTH else 0.0 for r in ranks]),
    )


def _rank_all_found(question: QuestionResult) -> int | None:
    """The rank by which every Evidence span has been found, or None if one never is."""
    ranks = question.evidence_ranks
    return None if None in ranks else max(ranks)


def _rank_any_found(question: QuestionResult) -> int | None:
    """The rank of the first Chunk containing any Evidence span."""
    found = [r for r in question.evidence_ranks if r is not None]
    return min(found, default=None)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def save_results(
    results: EvalResults, directory: Path, *, created_at: datetime | None = None
) -> Path:
    """Write one results file per configuration, split and manifest version."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{results.configuration}_{results.split}_{results.manifest_version}.json"
    record = {"created_at": (created_at or datetime.now(UTC)).isoformat(), **asdict(results)}
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_latest_results(directory: Path) -> dict | None:
    """The most recently written results file, as stored, or None if there are none."""
    records = [json.loads(p.read_text(encoding="utf-8")) for p in directory.glob("*.json")]
    return max(records, key=lambda r: r["created_at"], default=None)
