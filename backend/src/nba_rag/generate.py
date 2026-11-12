"""Generate candidate Eval questions from sampled Chunks with an LLM.

Sampling is stratified by Article type and Origin so the candidates aren't all prose about
players. The LLM is asked to paraphrase rather than copy the Chunk, and to use nicknames
and other names where they exist. A candidate whose Evidence isn't verbatim in the Corpus
is rejected automatically; survivors go to a review file for a human (see ``review``).
"""

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import httpx

from nba_rag.corpus import Chunk, Corpus
from nba_rag.eval import (
    EvalQuestion,
    Evidence,
    QuestionType,
    Source,
    article_texts,
    evidence_problems,
)
from nba_rag.llm import LLMClient

ArticleType = Literal["season", "finals", "other"]
Status = Literal["pending", "kept", "dropped"]

GENERATED_TYPES = ("lookup", "explanation")
"""Multi-hop questions are hand-written, so the generator never produces them."""

_SEASON = re.compile(r"^\d{4}–\d{2,4} (NBA|BAA) season$")
_FINALS = re.compile(r"^\d{4} (NBA|BAA) Finals$")


def article_type(title: str) -> ArticleType:
    """Season and Finals Articles are told apart by title; franchises and players are "other"."""
    if _SEASON.match(title):
        return "season"
    if _FINALS.match(title):
        return "finals"
    return "other"


@dataclass(frozen=True)
class Candidate:
    """A generated Eval question awaiting review."""

    id: str
    text: str
    question_type: QuestionType
    source: Source
    evidence: list[Evidence]
    chunk_id: str
    status: Status = "pending"

    def to_question(self) -> EvalQuestion:
        # The split is assigned later, over the whole Eval set (see ``split_eval_set``).
        return EvalQuestion(
            id=self.id,
            text=self.text,
            question_type=self.question_type,
            source=self.source,
            split="test",
            evidence=self.evidence,
        )


@dataclass(frozen=True)
class Rejection:
    chunk_id: str
    reason: str


@dataclass(frozen=True)
class GenerationResult:
    candidates: list[Candidate]
    rejected: list[Rejection]


def sample_chunks(corpus: Corpus, count: int, seed: int) -> list[Chunk]:
    """Pick Chunks round-robin across (Article type, Origin) strata, without repeats."""
    rng = random.Random(seed)
    strata: dict[tuple[ArticleType, str], list[Chunk]] = defaultdict(list)
    for chunk in corpus.chunks:
        strata[(article_type(chunk.article_title), chunk.origin)].append(chunk)
    queues = [rng.sample(chunks, len(chunks)) for _, chunks in sorted(strata.items())]

    sample: list[Chunk] = []
    while queues and len(sample) < count:
        for queue in queues:
            if len(sample) == count:
                break
            sample.append(queue.pop())
        queues = [q for q in queues if q]
    return sample


def generate_candidates(
    corpus: Corpus, llm: LLMClient, chunks: list[Chunk], first_number: int = 1
) -> GenerationResult:
    texts = article_texts(corpus)
    candidates: list[Candidate] = []
    rejected: list[Rejection] = []
    for chunk in chunks:
        number = first_number + len(candidates)
        try:
            candidate = _parse_reply(llm.complete(_prompt(chunk)), chunk, f"g{number:03d}")
        except (ValueError, httpx.HTTPError) as error:
            reason = str(error) or type(error).__name__
            if isinstance(error, httpx.HTTPError):
                reason = f"LLM call failed: {type(error).__name__}: {reason}"
            rejected.append(Rejection(chunk.id, reason))
            continue
        problems = evidence_problems(candidate.to_question(), texts)
        if problems:
            rejected.append(Rejection(chunk.id, "; ".join(problems)))
        else:
            candidates.append(candidate)
    return GenerationResult(candidates, rejected)


def _prompt(chunk: Chunk) -> str:
    return (
        f"Below is a passage from the Wikipedia article {chunk.article_title!r}.\n\n"
        f"{chunk.text}\n\n"
        "Write one question that this passage answers. Paraphrase: do not reuse the "
        "passage's wording, and refer to people, teams and events by nicknames or other "
        "names where they have one. Also give the evidence: ONE sentence (or one row) "
        "copied verbatim from the passage that answers the question.\n"
        'Reply with only JSON: {"question": "...", "evidence": "...", '
        f'"question_type": "lookup" (a single fact) or "explanation" (why/how)}}'
    )


def _parse_reply(reply: str, chunk: Chunk, candidate_id: str) -> Candidate:
    try:
        data = json.loads(_strip_code_fence(reply))
        question, evidence, question_type = (
            data["question"],
            data["evidence"],
            data.get("question_type", "lookup"),
        )
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"malformed LLM reply: {reply[:80]!r}") from error
    if question_type not in GENERATED_TYPES:
        raise ValueError(f"question_type {question_type!r} is not one of {GENERATED_TYPES}")
    if not (
        isinstance(question, str)
        and isinstance(evidence, str)
        and question.strip()
        and evidence.strip()
    ):
        raise ValueError("question and evidence must be non-empty strings")
    return Candidate(
        id=candidate_id,
        text=question.strip(),
        question_type=question_type,
        source="llm-generated",
        evidence=[Evidence(chunk.article_title, evidence.strip())],
        chunk_id=chunk.id,
    )


def _strip_code_fence(reply: str) -> str:
    """Models often wrap JSON in a ```json fence; take what is inside it."""
    text = reply.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].removesuffix("```")
    return text
