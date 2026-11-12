"""Tools for building the Eval set (run from the backend directory).

    python -m nba_rag.eval_tools search born akron        # find Evidence by plain text
    python -m nba_rag.eval_tools generate --count 80      # LLM candidates -> review.jsonl
    python -m nba_rag.eval_tools review review.jsonl      # keep / edit / drop each one
    python -m nba_rag.eval_tools finalize review.jsonl    # append kept ones to eval_set.jsonl
    python -m nba_rag.eval_tools split --dev-size 35      # stratified dev/test split
    python -m nba_rag.eval_tools log-out-of-corpus "question" --found-at URL

``generate`` needs OPENAI_API_KEY. ``search`` never touches a retrieval configuration, so
Evidence for hand-written and web questions is found independently of the system under test.
"""

import argparse
from pathlib import Path

from nba_rag.eval import (
    Evidence,
    OutOfCorpusCandidate,
    load_eval_set,
    log_out_of_corpus,
    save_eval_set,
    split_eval_set,
)
from nba_rag.generate import generate_candidates, sample_chunks
from nba_rag.llm import OpenAIClient
from nba_rag.review import Decision, kept_questions, review_candidates, write_review_file
from nba_rag.settings import data_dir
from nba_rag.store import load_built
from nba_rag.textsearch import search_text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    search = commands.add_parser("search", help="plain text search over the Corpus")
    search.add_argument("words", nargs="+")
    search.add_argument("--limit", type=int, default=10)

    generate = commands.add_parser("generate", help="ask the LLM for candidate questions")
    generate.add_argument("--count", type=int, default=80, help="Chunks to sample")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--first-number", type=int, default=1, help="first candidate id, g001")
    generate.add_argument("--out", type=Path, default=Path("review.jsonl"))

    review = commands.add_parser("review", help="keep, edit or drop each pending candidate")
    review.add_argument("file", type=Path)

    finalize = commands.add_parser("finalize", help="append kept candidates to the Eval set")
    finalize.add_argument("file", type=Path)
    finalize.add_argument("--eval-set", type=Path, default=Path("eval_set.jsonl"))

    split = commands.add_parser("split", help="re-split the Eval set into dev and test")
    split.add_argument("--eval-set", type=Path, default=Path("eval_set.jsonl"))
    split.add_argument("--dev-size", type=int, default=35)
    split.add_argument("--seed", type=int, default=0)

    log = commands.add_parser(
        "log-out-of-corpus", help="log a web question the Corpus can't answer"
    )
    log.add_argument("question")
    log.add_argument("--found-at", default="")
    log.add_argument("--suggested-article", default="")
    log.add_argument("--note", default="")
    log.add_argument("--log", type=Path, default=Path("out_of_corpus.jsonl"))

    args = parser.parse_args(argv)
    match args.command:
        case "search":
            _search(args)
        case "generate":
            _generate(args)
        case "review":
            _review(args)
        case "finalize":
            _finalize(args)
        case "split":
            _split(args)
        case "log-out-of-corpus":
            log_out_of_corpus(
                args.log,
                OutOfCorpusCandidate(
                    args.question, args.found_at, args.suggested_article, args.note
                ),
            )
            print(f"Logged to {args.log}")


def _search(args: argparse.Namespace) -> None:
    corpus, _ = load_built(data_dir())
    hits = search_text(corpus, " ".join(args.words), args.limit)
    if not hits:
        print("No Chunks contain all of those words.")
    for chunk in hits:
        print(f"[{chunk.article_title} | {' > '.join(chunk.section_path)} | {chunk.origin}]")
        print(chunk.text, end="\n\n")


def _generate(args: argparse.Namespace) -> None:
    corpus, _ = load_built(data_dir())
    chunks = sample_chunks(corpus, args.count, args.seed)
    result = generate_candidates(corpus, OpenAIClient(), chunks, args.first_number)
    write_review_file(args.out, result.candidates)
    for rejection in result.rejected:
        print(f"rejected {rejection.chunk_id}: {rejection.reason}")
    print(
        f"{len(result.candidates)} candidates for review in {args.out}; "
        f"{len(result.rejected)} rejected"
    )


def _review(args: argparse.Namespace) -> None:
    corpus, _ = load_built(data_dir())

    def decide(candidate) -> Decision:
        print(f"\n{candidate.id} [{candidate.question_type}] {candidate.text}")
        for e in candidate.evidence:
            print(f"  Evidence ({e.article_title}): {e.text}")
        while True:
            answer = input("(k)eep, (e)dit, (d)rop? ").strip().lower()
            if answer == "k":
                return Decision.keep()
            if answer == "d":
                return Decision.drop()
            if answer == "e":
                text = input("question (blank to keep): ").strip() or None
                span = input("evidence (blank to keep): ").strip()
                evidence = [Evidence(candidate.evidence[0].article_title, span)] if span else None
                return Decision.edit(text, evidence)

    review_candidates(args.file, corpus, decide)
    print("Review finished.")


def _finalize(args: argparse.Namespace) -> None:
    existing = load_eval_set(args.eval_set) if args.eval_set.exists() else []
    taken = {q.id for q in existing}
    kept = [q for q in kept_questions(args.file) if q.id not in taken]
    save_eval_set(args.eval_set, [*existing, *kept])
    print(f"Added {len(kept)} questions to {args.eval_set}; run `split` to assign dev/test.")


def _split(args: argparse.Namespace) -> None:
    questions = split_eval_set(load_eval_set(args.eval_set), args.dev_size, args.seed)
    save_eval_set(args.eval_set, questions)
    print(f"{args.dev_size} dev / {len(questions) - args.dev_size} test in {args.eval_set}")


if __name__ == "__main__":
    main()
