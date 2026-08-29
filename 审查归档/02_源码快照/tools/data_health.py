"""T1 · 数据体检

把「这道题的结构长什么样」量化成可复跑的数字。一份代码两个用途：
  1. 开工前的前提核验——数字对不上就说明我们对评测器的理解有误，必须停下重读；
  2. 最终技术报告里三张数据表的唯一数据来源。

用法:
    python -m tools.data_health                    # 跑体检，写 docs/data_health.json
    python -m tools.data_health --check            # 只做验收判定，过则退出码 0

验收线 (T1):
    类目桶数 == 1115  且  目标商品落桶率 == 200/200
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 直接复用官方评测器的规则，避免我们自己复刻一份走样的实现。
# 一旦官方改了规则，这里会立刻跟着变，而不是悄悄地和评测器产生分歧。
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

    # ── 目录：字段覆盖率 ────────────────────────────────────────────
    fields = ["title", "features", "details", "description", "categories",
              "store", "price", "average_rating", "rating_number"]
    coverage = {
        field: round(
            sum(1 for p in products.values() if p.get(field) not in (None, "", [], {}))
            / len(products), 4
        )
        for field in fields
    }

    # ── 类目桶：这是我们唯一敢用硬过滤的地方，落桶率必须 100% ──────────
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

    # ── 约束：决定 ask_attribute 的排序，信息量最大的先问 ──────────────
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
    """T1 验收：二元判定，没有'差不多'。"""
    checks = [
        ("类目桶数 == 1115",
         report["buckets"]["bucket_count"] == EXPECTED_BUCKETS,
         report["buckets"]["bucket_count"]),
        ("目标落桶率 == 200/200",
         report["buckets"]["target_in_bucket"] == f"{EXPECTED_SESSIONS}/{EXPECTED_SESSIONS}",
         report["buckets"]["target_in_bucket"]),
    ]
    lines = [f"  [{'PASS' if ok else 'FAIL'}] {name}  实际={actual}" for name, ok, actual in checks]
    return all(ok for _, ok, _ in checks), lines


def main() -> int:
    parser = argparse.ArgumentParser(description="T1 数据体检")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="docs/data_health.json")
    parser.add_argument("--check", action="store_true", help="只打印验收判定")
    args = parser.parse_args()

    report = run(Path(args.catalog), Path(args.dataset))
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    passed, lines = acceptance(report)
    print("T1 验收")
    print("\n".join(lines))

    if not args.check:
        b, c = report["buckets"], report["constraints"]
        print(f"\n目录          {report['catalog']['product_count']} 件商品")
        print(f"类目桶        {b['bucket_count']} 个 | 桶大小 中位数 {b['size_percentiles']['median']}"
              f" (P25 {b['size_percentiles']['p25']} / P75 {b['size_percentiles']['p75']}"
              f" / P90 {b['size_percentiles']['p90']})")
        print(f"剪枝效果      50000 -> {b['size_percentiles']['median']} (中位数, 约 "
              f"{report['catalog']['product_count'] // max(1, b['size_percentiles']['median'])}x)")
        print(f"小桶占比      <=50 件: {b['sessions_with_bucket_le_50']}/{EXPECTED_SESSIONS}"
              f" | <=200 件: {b['sessions_with_bucket_le_200']}/{EXPECTED_SESSIONS}")
        print(f"\n约束总数      {c['total']} 条 | 每场 {list(c['per_session'])} 条")
        print("约束类型分布  (决定 ask_attribute 的追问顺序)")
        for name, info in c["by_type"].items():
            bar = "#" * round(info["share"] * 40)
            print(f"  {name:<10} {info['count']:>4}  {info['share']:>6.1%}  {bar}")
        print(f"\n场景分布      {report['scenarios']}")
        print(f"难度分布      {report['difficulty']}")
        print(f"\n完整报告 -> {args.output}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
