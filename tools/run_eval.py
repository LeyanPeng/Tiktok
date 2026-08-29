"""Run our Agent through the official evaluator.

The official `evaluator/local_evaluator.py` hardcodes `import starter.agent`, and
the evaluator is judging apparatus that we treat as frozen. So this reuses its
`evaluate()` and substitutes only the Agent.

Usage:
    python -m tools.run_eval                    # score
    python -m tools.run_eval --gate 0.891       # score with an acceptance gate
    python -m tools.run_eval --paraphrase       # against paraphrased customer phrasing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import evaluator.local_evaluator as official

from agent import Agent

PROGRESS_KEYS = ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")


def install_paraphrase() -> None:
    """Wrap the simulator so it rephrases the customer's messages.

    The official specification notes that natural-language paraphrasing may be
    added to the private split, and that it cannot decide correctness. This layer
    quantifies what such a rewrite would cost us.

    Note that the rewrite below is our own invention: the private split may
    paraphrase differently, or not at all.
    """
    base_initial, base_reply = official.initial_message, official.customer_reply

    def initial(sample, category, disclosed):
        text = base_initial(sample, category, disclosed)
        return (
            text.replace("I'm looking for ", "Hi! I want to find some ")
                .replace(". A key requirement is: ", " and it really has to have ")
                .replace(", but I'm still exploring.", " -- just browsing around for now, nothing fixed yet.")
        )

    def reply(sample, ask_attribute, disclosed, boundary_used):
        text, used = base_reply(sample, ask_attribute, disclosed, boundary_used)
        return text.replace("For that, what matters is: ", "Honestly what I care about here would be "), used

    official.initial_message, official.customer_reply = initial, reply


def main() -> int:
    parser = argparse.ArgumentParser(description="Run our Agent through the official evaluator")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="docs/our_results.json")
    parser.add_argument("--gate", type=float, default=None, help="acceptance floor for TechnicalScore")
    parser.add_argument("--paraphrase", action="store_true", help="paraphrase the customer's phrasing")
    parser.add_argument("--label", default="", help="tag written into the results file")
    args = parser.parse_args()

    if args.paraphrase:
        install_paraphrase()

    started = time.perf_counter()
    catalog_ids, categories, products = official.catalog_index(args.catalog)
    samples = official.load_jsonl(args.dataset)
    build_done = time.perf_counter()

    result = official.evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)
    finished = time.perf_counter()

    result["run_meta"] = {
        "label": args.label or ("paraphrase" if args.paraphrase else "offline"),
        "paraphrase": args.paraphrase,
        "index_build_seconds": round(build_done - started, 2),
        "eval_seconds": round(finished - build_done, 2),
        "seconds_per_session": round((finished - build_done) / max(1, len(samples)), 4),
        "network_required": False,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    score = result["recommended_technical_score"]
    mode = "paraphrased" if args.paraphrase else "offline, verbatim"
    print(f"Mode      {mode}")
    print(f"Timing    index {result['run_meta']['index_build_seconds']}s"
          f" / eval {result['run_meta']['eval_seconds']}s"
          f" / {result['run_meta']['seconds_per_session']}s per session")
    print(f"Tokens    {result['reported_token_usage']['total_tokens']}"
          f"  (zero calls = runs with the network disabled)")
    print()
    for key in PROGRESS_KEYS:
        print(f"  {key:<32} {result[key]}")
    print("\nBy scenario")
    for name, info in result["scenario_metrics"].items():
        print(f"  {name:<16} n={info['sample_count']:<4}"
              f" hit={info['hit_rate_at_10']:<8} mrr={info['mrr']:<10} mttc={info['mttc']}")

    if args.gate is not None:
        passed = score >= args.gate
        print(f"\nAcceptance\n  [{'PASS' if passed else 'FAIL'}] TechnicalScore >= {args.gate}"
              f"   actual={score}")
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
