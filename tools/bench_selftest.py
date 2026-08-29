"""Reverse verification: are the evaluation benches fake green lights?

A bench that always reports a good score is more dangerous than no bench at all —
it accelerates work in the wrong direction. So before trusting either bench, prove
it can fail: feed it an agent that cannot possibly work, and require the score to
collapse.

Wherever the answer to "if this broke, who would find out?" is "nobody", this step
is mandatory.

Acceptance: the dummy agent scores below 0.05 on both the verbatim and the
paraphrased bench.
"""

from __future__ import annotations

import sys

import evaluator.local_evaluator as official

from tools.run_eval import install_paraphrase

THRESHOLD = 0.05


class BrokenAgent:
    """An agent that does nothing. Its score must collapse."""

    def __init__(self, catalog_path: str = "data/catalog.jsonl") -> None:
        self.catalog_path = catalog_path

    def reset(self, session_id: str, user_profile: dict) -> None:
        pass

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {"message": "", "ask_attribute": None, "recommendations": []}


def score_with(agent, catalog: str, dataset: str) -> float:
    ids, cats, products = official.catalog_index(catalog)
    samples = official.load_jsonl(dataset)
    return official.evaluate(agent, samples, ids, cats, products)["recommended_technical_score"]


def main() -> int:
    catalog, dataset = "data/catalog.jsonl", "data/public_set.jsonl"

    plain = score_with(BrokenAgent(), catalog, dataset)
    install_paraphrase()
    perturbed = score_with(BrokenAgent(), catalog, dataset)

    checks = [("verbatim bench can fail", plain), ("paraphrased bench can fail", perturbed)]
    print("Bench self-test: can the benches fail?")
    ok = True
    for name, value in checks:
        passed = value < THRESHOLD
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
              f"   dummy agent scored {value} (must be < {THRESHOLD})")
    if not ok:
        print("\nThe bench did not fail. It is a fake green light, and every score it\n"
              "has reported so far is untrustworthy.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
