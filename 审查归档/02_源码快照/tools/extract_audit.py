"""T2 验收 · 约束抽取审计

问题：我们的抽取层，到底听懂了顾客说的百分之多少？

做法：用官方模拟器逐轮生成真实对话，同时记录每句话里**实际塞进去**的约束原句，
然后把同一句话喂给 src/extract.py，比对两边。这是一次严格的对照，不是自评。

验收线 (T2)：>= 95% 的 session，抽出条数 == 实际吐露条数
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# 模拟器一侧：直接用官方实现，保证对照基准没有走样
from evaluator.local_evaluator import (
    ALLOWED_ATTRIBUTES,
    MAX_TURNS,
    behavior_for,
    classify_constraint,
    coarse_category,
    intent_card,
)
from src.catalog import Catalog
from src.session_state import SessionState

# 固定追问顺序：按 T1 实测的信息量排序（feature 50.5% > material 37.8% > color 7.5%）
ASK_SEQUENCE = ["feature", "material", "color", "style", "size", "use_case", "budget", "brand"]


def spoken_initial(sample: dict, category: str, disclosed: set[str]) -> tuple[str, list[str]]:
    """复刻 evaluator.initial_message，额外返回这句话里真实吐露的约束。"""
    scenario = sample["scenario_type"]
    card = sample["intent_card"]
    if scenario == "buying" and card.get("hard_constraints"):
        value = str(card["hard_constraints"][0])
        disclosed.add(value)
        return f"I'm looking for {category}. A key requirement is: {value}.", [value]
    if scenario == "intent_override":
        old = str(sample["behavior"]["override"]["old_value"])
        # 注意：官方没有把 old_value 记进 disclosed，但顾客确实说出口了
        return f"I'm looking for {category}. {old}", [old]
    return f"I'm looking for {category}, but I'm still exploring.", []


def spoken_reply(
    sample: dict, ask_attribute: str | None, disclosed: set[str], boundary_used: bool
) -> tuple[str, list[str], bool]:
    """复刻 evaluator.customer_reply，额外返回这句话里真实吐露的约束。"""
    attribute = ask_attribute if isinstance(ask_attribute, str) else None
    if sample["scenario_type"] == "boundary" and not boundary_used and attribute:
        return f"I don't have a preference for {attribute}; please use your judgment.", [], True
    if not attribute:
        return "Those options are not quite right yet. Ask me about one specific attribute.", [], boundary_used
    if attribute not in ALLOWED_ATTRIBUTES:
        attribute = "other"
    pool = [
        *[str(v) for v in sample["intent_card"].get("hard_constraints", [])],
        *[str(v) for v in sample["intent_card"].get("soft_preferences", [])],
    ]
    matches = [
        v for v in pool
        if v not in disclosed and (attribute == "other" or classify_constraint(v) == attribute)
    ][:2]
    if not matches:
        return f"I don't have an additional preference for {attribute}.", [], boundary_used
    disclosed.update(matches)
    return "For that, what matters is: " + "; ".join(matches) + ".", matches, boundary_used


def audit_session(sample: dict, products: dict, catalog: Catalog) -> dict:
    target = str(sample["ground_truth"]["parent_asin"])
    card = intent_card(products[target])
    rng = random.Random(f"{sample.get('sample_id', '')}\0{sample.get('scenario_type', '')}")
    behavior = behavior_for(str(sample["scenario_type"]), card, rng)
    full = {**sample, "intent_card": card, "behavior": behavior}

    category = coarse_category([str(v) for v in (products[target].get("categories") or [])])
    state = SessionState(session_id=sample["sample_id"])
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"

    all_spoken: list[str] = []
    transcript: list[dict] = []

    message, spoken = spoken_initial(full, category, disclosed)
    for turn in range(1, MAX_TURNS + 1):
        all_spoken.extend(spoken)
        before = len(state.slots)
        state.observe(
            message, turn, catalog.categories_by_length,
            verify=lambda t: catalog.contains_verbatim(t, state.category),
        )
        transcript.append({
            "turn": turn,
            "message": message,
            "spoken": spoken,
            "extracted": [s.text for s in state.slots[before:]],
        })

        if turn == MAX_TURNS:
            break
        ask = ASK_SEQUENCE[(turn - 1) % len(ASK_SEQUENCE)]
        state.asked.append(ask)

        override = full.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            disclosed.add(new_value)
            message = str(override.get("message", ""))
            spoken = [new_value] if new_value else []
        else:
            message, spoken, boundary_used = spoken_reply(full, ask, disclosed, boundary_used)

    spoken_set = {s.strip().lower() for s in all_spoken}
    extracted_set = {s.text.strip().lower() for s in state.slots}
    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "spoken_count": len(spoken_set),
        "extracted_count": len(extracted_set),
        "count_match": len(spoken_set) == len(extracted_set),
        "exact_match": spoken_set == extracted_set,
        "recall": round(len(spoken_set & extracted_set) / len(spoken_set), 4) if spoken_set else 1.0,
        "missed": sorted(spoken_set - extracted_set),
        "spurious": sorted(extracted_set - spoken_set),
        "category_found": state.category == category,
        "intent": state.intent,
        "override_detected": state.override_turn is not None,
        "transcript": transcript,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="T2 约束抽取审计")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="docs/extract_audit.json")
    parser.add_argument("--show", type=int, default=3, help="打印几条失败样本")
    args = parser.parse_args()

    catalog = Catalog(args.catalog)
    products = {}
    with Path(args.catalog).open(encoding="utf-8") as handle:
        for line in handle:
            p = json.loads(line)
            products[str(p["parent_asin"])] = p
    samples = [json.loads(l) for l in Path(args.dataset).open(encoding="utf-8") if l.strip()]

    results = [audit_session(s, products, catalog) for s in samples]
    n = len(results)

    count_match = sum(r["count_match"] for r in results) / n
    exact_match = sum(r["exact_match"] for r in results) / n
    recall = sum(r["recall"] for r in results) / n
    category_hit = sum(r["category_found"] for r in results) / n
    override_hit = (
        sum(r["override_detected"] for r in results if r["scenario_type"] == "intent_override")
        / max(1, sum(1 for r in results if r["scenario_type"] == "intent_override"))
    )

    summary = {
        "sample_count": n,
        "count_match_rate": round(count_match, 4),
        "exact_match_rate": round(exact_match, 4),
        "mean_recall": round(recall, 4),
        "category_hit_rate": round(category_hit, 4),
        "override_detect_rate": round(override_hit, 4),
        "by_scenario": {},
    }
    for scenario in sorted({r["scenario_type"] for r in results}):
        group = [r for r in results if r["scenario_type"] == scenario]
        summary["by_scenario"][scenario] = {
            "n": len(group),
            "count_match": round(sum(g["count_match"] for g in group) / len(group), 4),
            "recall": round(sum(g["recall"] for g in group) / len(group), 4),
        }

    Path(args.output).write_text(
        json.dumps({"summary": summary, "sessions": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    passed = count_match >= 0.95
    print("T2 验收")
    print(f"  [{'PASS' if passed else 'FAIL'}] 抽出条数 == 吐露条数 的场次比例 >= 95%"
          f"   实际={count_match:.1%}")
    print()
    print(f"内容完全一致    {exact_match:.1%}")
    print(f"平均召回        {recall:.1%}")
    print(f"类目识别正确    {category_hit:.1%}")
    print(f"改主意识别      {override_hit:.1%}")
    print("\n分场景")
    for scenario, info in summary["by_scenario"].items():
        print(f"  {scenario:<16} n={info['n']:<4} 条数一致={info['count_match']:>6.1%}  召回={info['recall']:>6.1%}")

    failures = [r for r in results if not r["count_match"]]
    if failures and args.show:
        print(f"\n失败样本 {len(failures)} 例，前 {min(args.show, len(failures))} 例：")
        for r in failures[:args.show]:
            print(f"  -- {r['sample_id']} [{r['scenario_type']}] "
                  f"吐露 {r['spoken_count']} / 抽出 {r['extracted_count']}")
            for text in r["missed"][:2]:
                print(f"     漏抽: {text[:88]!r}")
            for text in r["spurious"][:2]:
                print(f"     多抽: {text[:88]!r}")

    print(f"\n完整报告 -> {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
