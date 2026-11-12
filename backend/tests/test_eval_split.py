from collections import Counter

from nba_rag.eval import EvalQuestion, Evidence, split_eval_set


def questions(spec: dict[tuple[str, str], int]) -> list[EvalQuestion]:
    out = []
    for (question_type, source), count in spec.items():
        for i in range(count):
            out.append(
                EvalQuestion(
                    id=f"{question_type}-{source}-{i}",
                    text="?",
                    question_type=question_type,
                    source=source,
                    split="test",
                    evidence=[Evidence("A", "x")],
                )
            )
    return out


SPEC = {
    ("lookup", "llm-generated"): 50,
    ("lookup", "web"): 19,
    ("multi-hop", "hand-written"): 20,
    ("explanation", "llm-generated"): 16,
    ("explanation", "web"): 20,
}


def test_split_has_exactly_the_requested_dev_size():
    result = split_eval_set(questions(SPEC), dev_size=35, seed=1)

    assert Counter(q.split for q in result) == {"dev": 35, "test": 90}


def test_each_type_and_source_keeps_its_proportion_in_dev():
    result = split_eval_set(questions(SPEC), dev_size=35, seed=1)

    for key, count in SPEC.items():
        dev = sum(1 for q in result if (q.question_type, q.source) == key and q.split == "dev")
        assert abs(dev - count * 35 / 125) < 1, key


def test_split_is_reproducible_and_keeps_order_and_content():
    original = questions(SPEC)

    first = split_eval_set(original, dev_size=35, seed=7)

    assert first == split_eval_set(original, dev_size=35, seed=7)
    assert [q.id for q in first] == [q.id for q in original]
    assert first != split_eval_set(original, dev_size=35, seed=8)
