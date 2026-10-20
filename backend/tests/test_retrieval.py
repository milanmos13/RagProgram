import pytest

from fakes import KeywordEmbedder
from nba_rag.corpus import Chunk, Corpus
from nba_rag.retrieval import build_baseline, retrieve


def test_baseline_ranks_chunks_by_cosine_similarity(corpus):
    baseline = build_baseline(corpus, KeywordEmbedder())

    results = retrieve(baseline, "Was Curry the unanimous MVP?", k=3)

    # Query vector has curry=1, mvp=1. Worked by hand:
    #   "...first unanimous MVP"            curry=1 mvp=1  -> 1.0
    #   "Curry was born ... Dell Curry"     curry=2 born=1 -> 2/sqrt(10) = 0.6325
    #   "Wardell Stephen Curry II (born..." curry=1 born=1 -> 1/2 = 0.5
    assert [r.chunk.id for r in results] == [
        "Stephen Curry#3",
        "Stephen Curry#2",
        "Stephen Curry#0",
    ]
    assert [r.score for r in results] == pytest.approx([1.0, 0.6325, 0.5], abs=1e-4)


def test_chunks_with_no_overlap_score_zero(corpus):
    baseline = build_baseline(corpus, KeywordEmbedder())

    results = retrieve(baseline, "Who blocked Iguodala?", k=10)

    assert len(results) == len(corpus.chunks)
    assert all(r.score == 0.0 for r in results)


def test_identical_chunk_texts_are_collapsed_to_the_best_one():
    def chunk(article: str, position: int, text: str) -> Chunk:
        return Chunk(
            id=f"{article}#{position}",
            article_title=article,
            revision_id=1,
            section_path=(),
            origin="prose",
            text=text,
            position=position,
        )

    corpus = Corpus(
        manifest_version="test",
        chunks=[
            chunk("A", 0, "Irving and James."),
            chunk("B", 0, "Irving and James."),
            chunk("B", 1, "Irving."),
        ],
    )
    baseline = build_baseline(corpus, KeywordEmbedder())

    results = retrieve(baseline, "Irving James", k=3)

    assert [r.chunk.id for r in results] == ["A#0", "B#1"]
