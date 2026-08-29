"""跑我们的 Agent 过官方评测器。

官方 evaluator/local_evaluator.py 硬编码 import starter.agent，而 evaluator/ 属于
判卷标准、冻结不许改。所以这里复用它的 evaluate()，只把 Agent 换成我们的。

用法:
    python -m tools.run_eval                    # 跑分
    python -m tools.run_eval --gate 0.75        # 加验收线，未达标退出码 1
    python -m tools.run_eval --paraphrase       # 话术改写扰动版（T5 起使用）
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
    """给模拟器套一层话术改写。

    官方规格说私有集「可能加入自然语言改写」，但改写不决定正确性。
    这一层用来量化：如果话术变了，我们会掉多少分。
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
    parser = argparse.ArgumentParser(description="跑我们的 Agent 过官方评测器")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="docs/our_results.json")
    parser.add_argument("--gate", type=float, default=None, help="验收线：TechnicalScore 下限")
    parser.add_argument("--paraphrase", action="store_true", help="启用话术改写扰动")
    parser.add_argument("--label", default="", help="写进结果文件的标记")
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
    mode = "话术改写版" if args.paraphrase else "离线原版"
    print(f"评测模式  {mode}")
    print(f"耗时      建索引 {result['run_meta']['index_build_seconds']}s"
          f" / 评测 {result['run_meta']['eval_seconds']}s"
          f" / 每场 {result['run_meta']['seconds_per_session']}s")
    print(f"Token     {result['reported_token_usage']['total_tokens']}  (零调用 = 可断网运行)")
    print()
    for key in PROGRESS_KEYS:
        print(f"  {key:<32} {result[key]}")
    print("\n分场景")
    for name, info in result["scenario_metrics"].items():
        print(f"  {name:<16} n={info['sample_count']:<4}"
              f" hit={info['hit_rate_at_10']:<8} mrr={info['mrr']:<10} mttc={info['mttc']}")

    if args.gate is not None:
        passed = score >= args.gate
        print(f"\n验收\n  [{'PASS' if passed else 'FAIL'}] TechnicalScore >= {args.gate}"
              f"   实际={score}")
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
