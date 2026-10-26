"""One-off command: score a configuration on a split of the Eval set and write results.

    python -m nba_rag.evaluate --configuration baseline --split dev

The split defaults to dev, so tuning a configuration doesn't spend the test questions;
ask for ``--split test`` deliberately, and ``--split all`` mixes both.

Every Evidence span is checked against the built Corpus first, so Evidence broken by a
revision change fails loudly instead of scoring as misses. ``--validate-only`` stops
after that check; the production image build runs it, so a manifest change that breaks
Evidence fails the build.
"""

import argparse
from pathlib import Path
from typing import get_args

from nba_rag.configurations import CONFIGURATION_NAMES, load_configurations
from nba_rag.eval import Split, evaluate, load_eval_set, save_results, validate_evidence
from nba_rag.settings import data_dir, results_dir
from nba_rag.store import load_built


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=CONFIGURATION_NAMES, default="baseline")
    parser.add_argument("--split", choices=[*get_args(Split), "all"], default="dev")
    parser.add_argument("--eval-set", type=Path, default=Path("eval_set.jsonl"))
    parser.add_argument("--out", type=Path, default=None, help="defaults to NBA_RAG_RESULTS_DIR")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check the Evidence against the built Corpus and stop, without scoring",
    )
    args = parser.parse_args(argv)

    eval_set = load_eval_set(args.eval_set)
    corpus, matrices = load_built(data_dir())
    validate_evidence(eval_set, corpus)
    if args.validate_only:
        print(f"All Evidence in {len(eval_set)} Eval questions is in the Corpus.")
        return

    configuration = load_configurations(corpus, matrices)[args.configuration]
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
