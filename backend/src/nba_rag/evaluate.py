"""One-off command: score a configuration on a split of the Eval set and write results.

    python -m nba_rag.evaluate --configuration baseline --split all

Every Evidence span is checked against the built Corpus first, so Evidence broken by a
revision change fails loudly instead of scoring as misses.
"""

import argparse
from pathlib import Path
from typing import get_args

from nba_rag.embedders import TutorialEmbedder
from nba_rag.eval import Split, evaluate, load_eval_set, save_results, validate_evidence
from nba_rag.retrieval import build_baseline
from nba_rag.settings import EMBEDDING_MODEL, data_dir, results_dir
from nba_rag.store import load_built


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=["baseline"], default="baseline")
    parser.add_argument("--split", choices=[*get_args(Split), "all"], default="all")
    parser.add_argument("--eval-set", type=Path, default=Path("eval_set.jsonl"))
    parser.add_argument("--out", type=Path, default=None, help="defaults to NBA_RAG_RESULTS_DIR")
    args = parser.parse_args()

    eval_set = load_eval_set(args.eval_set)
    corpus, matrix = load_built(data_dir())
    validate_evidence(eval_set, corpus)
    configuration = build_baseline(corpus, TutorialEmbedder(EMBEDDING_MODEL), matrix)

    results = evaluate(configuration, eval_set, args.split)
    path = save_results(results, args.out or results_dir())

    m = results.metrics
    print(
        f"{results.configuration} on {results.split} ({results.question_count} questions): "
        f"recall@1 {m.recall_at_1:.3f}  recall@5 {m.recall_at_5:.3f}  "
        f"recall@10 {m.recall_at_10:.3f}  MRR@10 {m.mrr_at_10:.3f}"
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
