"""Data health check: quantify the structure of this task.

One script, two jobs:
  1. A precondition check before any development. If these numbers do not come out
     as expected, our understanding of the evaluator is wrong and work should stop
     rather than continue on a false premise.
  2. The single source for the structural figures quoted in the technical report.

Usage:
    python -m tools.data_health                    # full report -> docs/data_health.json
    python -m tools.data_health --check            # acceptance verdict only

Acceptance:
    1115 category buckets, and the target product falls inside its own bucket in
    all 200 sessions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# The simulator's own rules are reused here rather than reimplemented, so this
# audit cannot drift away from the thing it is auditing.
from evaluator.local_evaluator import (
    classify_constraint,
    coarse_category,
    intent_card,
)

EXPECTED_BUCKETS = 1115
EXPECTED_SESSIONS = 200


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_catalog(path: Path) -> dict[str, dict]:
    products: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            products[str(product["parent_asin"])] = product
    return products


def bucket_of(product: dict) -> str:
    return coarse_category([str(v) for v in (product.get("categories") or [])])


def percentiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    n = len(ordered)
    def at(fraction: float) -> int:
        return ordered[min(n - 1, int(fraction * n))]
    return {
        "min": ordered[0],
        "p25": at(0.25),
        "median": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "max": ordered[-1],
    }


def run(catalog_path: Path, dataset_path: Path) -> dict:
    products = build_catalog(catalog_path)
    sessions = load_jsonl(dataset_path)

    # ── Catalog field coverage ──────────────────────────────────────
    fields = ["title", "features", "details", "description", "categories",
              "store", "price", "average_rating", "rating_number"]
    coverage = {
        field: round(
            sum(1 for p in products.values() if p.get(field) not in (None, "", [], {}))
            / len(products), 4
        )
        for field in fields
    }

    # ── Category buckets: the only place a hard filter is allowed, which is
    #    licensed entirely by a 100% containment rate ─────────────────
    buckets: dict[str, list[str]] = defaultdict(list)
    for asin, product in products.items():
        buckets[bucket_of(product)].append(asin)

    bucket_sizes: list[int] = []
    in_bucket = 0
    for session in sessions:
        target = str(session["ground_truth"]["parent_asin"])
        name = bucket_of(products[target])
        bucket_sizes.append(len(buckets[name]))
        if target in buckets[name]:
            in_bucket += 1

    # ── Constraints: this distribution drives the clarification ordering ──
    constraint_types: Counter[str] = Counter()
    constraints_per_session: Counter[int] = Counter()
    constraint_lengths: list[int] = []
    for session in sessions:
        target = str(session["ground_truth"]["parent_asin"])
        card = intent_card(products[target])
        distinct = list(dict.fromkeys(card["hard_constraints"] + card["soft_preferences"]))
        constraints_per_session[len(distinct)] += 1
        for value in distinct:
            constraint_types[classify_constraint(value)] += 1
            constraint_lengths.append(len(value.split()))

    total_constraints = sum(constraint_types.values())

    return {
        "catalog": {
            "product_count": len(products),
            "field_coverage": coverage,
        },
        "buckets": {
            "bucket_count": len(buckets),
            "target_in_bucket": f"{in_bucket}/{len(sessions)}",
            "target_in_bucket_rate": round(in_bucket / len(sessions), 4),
            "size_percentiles": percentiles(bucket_sizes),
            "sessions_with_bucket_le_50": sum(1 for s in bucket_sizes if s <= 50),
            "sessions_with_bucket_le_200": sum(1 for s in bucket_sizes if s <= 200),
            "mean_size": round(statistics.fmean(bucket_sizes), 1),
        },
        "constraints": {
            "total": total_constraints,
            "per_session": dict(sorted(constraints_per_session.items())),
            "by_type": {
                name: {"count": count, "share": round(count / total_constraints, 4)}
                for name, count in constraint_types.most_common()
            },
            "median_word_length": statistics.median(constraint_lengths),
        },
        "scenarios": dict(Counter(s["scenario_type"] for s in sessions).most_common()),
        "difficulty": dict(Counter(s.get("difficulty_bucket", "?") for s in sessions).most_common()),
    }


def acceptance(report: dict) -> tuple[bool, list[str]]:
    """Binary acceptance. No "close enough"."""
    checks = [
        ("category buckets == 1115",
         report["buckets"]["bucket_count"] == EXPECTED_BUCKETS,
         report["buckets"]["bucket_count"]),
        ("target falls in its own bucket == 200/200",
         report["buckets"]["target_in_bucket"] == f"{EXPECTED_SESSIONS}/{EXPECTED_SESSIONS}",
         report["buckets"]["target_in_bucket"]),
    ]
    lines = [f"  [{'PASS' if ok else 'FAIL'}] {name}   actual={actual}"
             for name, ok, actual in checks]
    return all(ok for _, ok, _ in checks), lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Data health check")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="docs/data_health.json")
    parser.add_argument("--check", action="store_true", help="print the verdict only")
    args = parser.parse_args()

    report = run(Path(args.catalog), Path(args.dataset))
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    passed, lines = acceptance(report)
    print("Data health")
    print("\n".join(lines))

    if not args.check:
        b, c = report["buckets"], report["constraints"]
        print(f"\ncatalog          {report['catalog']['product_count']} products")
        print(f"buckets          {b['bucket_count']} | size median {b['size_percentiles']['median']}"
              f" (p25 {b['size_percentiles']['p25']} / p75 {b['size_percentiles']['p75']}"
              f" / p90 {b['size_percentiles']['p90']})")
        print(f"pruning          50000 -> {b['size_percentiles']['median']} at the median, about "
              f"{report['catalog']['product_count'] // max(1, b['size_percentiles']['median'])}x")
        print(f"small buckets    <=50 items: {b['sessions_with_bucket_le_50']}/{EXPECTED_SESSIONS}"
              f" | <=200 items: {b['sessions_with_bucket_le_200']}/{EXPECTED_SESSIONS}")
        print(f"\nconstraints      {c['total']} total | {list(c['per_session'])} per session")
        print("by type          (this ordering drives the clarification policy)")
        for name, info in c["by_type"].items():
            bar = "#" * round(info["share"] * 40)
            print(f"  {name:<10} {info['count']:>4}  {info['share']:>6.1%}  {bar}")
        print(f"\nscenarios        {report['scenarios']}")
        print(f"difficulty       {report['difficulty']}")
        print(f"\nfull report -> {args.output}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
