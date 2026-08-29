"""逐轮播放一场完整会话，供 Demo 视频录制与人工检查使用。

官方 Deliverables 要求 "One demonstrated multi-turn session"。
这个脚本把评测器内部发生的事全部摊开：顾客说了什么、我们抽到了什么、
候选池怎么收敛、为什么问这个属性、目标商品排到第几名。

用法:
    python -m tools.demo_session                       # 默认放一场 intent_override
    python -m tools.demo_session --scenario buying
    python -m tools.demo_session --sample public_0002
    python -m tools.demo_session --scenario browsing --failure   # 挑一场没拿第1名的
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import evaluator.local_evaluator as official

from agent import Agent
from tools.extract_audit import spoken_initial, spoken_reply

BAR = "=" * 78
SEP = "-" * 78


def load_products(path: str) -> dict[str, dict]:
    products = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            products[str(item["parent_asin"])] = item
    return products


def short(text: str, width: int = 68) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def play(sample: dict, products: dict, agent: Agent) -> None:
    target = str(sample["ground_truth"]["parent_asin"])
    card = official.intent_card(products[target])
    rng = random.Random(f"{sample['sample_id']}\0{sample['scenario_type']}")
    behavior = official.behavior_for(str(sample["scenario_type"]), card, rng)
    full = {**sample, "intent_card": card, "behavior": behavior}
    category = official.coarse_category(
        [str(v) for v in (products[target].get("categories") or [])]
    )

    print(BAR)
    print(f" 会话 {sample['sample_id']}   场景 {sample['scenario_type']}"
          f"   难度 {sample.get('difficulty_bucket', '?')}")
    print(f" 隐藏目标  {target}  {short(products[target].get('title'), 50)}")
    print(f" 类目桶    {category!r}   桶内 {len(agent.catalog.candidates(category))} 件"
          f"   (全目录 {len(agent.catalog)} 件)")
    print(f" 顾客档案  {short(sample['user_profile'].get('summary'), 62)}")
    print(BAR)

    session_id = f"demo_{sample['sample_id']}"
    agent.reset(session_id, sample["user_profile"])
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"

    message, _ = spoken_initial(full, category, disclosed)
    for turn in range(1, official.MAX_TURNS + 1):
        print(f"\n 第 {turn} 轮")
        print(SEP)
        print(f"  顾客   {short(message, 66)}")

        before = len(agent._sessions[session_id].slots)
        response = agent.respond(session_id, message, turn, official.TOP_K)
        state = agent._sessions[session_id]

        picked_up = [s.text for s in state.slots[before:]]
        if picked_up:
            for text in picked_up:
                print(f"  抽取   + {short(text, 62)}")
        else:
            print("  抽取   （本轮没有新信息）")

        if state.override_turn == turn:
            print("  状态   检测到改主意 → 旧槽位降权 ×0.35（不删除，仍指向同一商品）")

        ranked = [r["parent_asin"] for r in response["recommendations"]]
        position = ranked.index(target) + 1 if target in ranked else None
        print(f"  排序   已知 {len(state.slots)} 条约束"
              f" · 意图轨 {state.intent}"
              f" · 前十名{'含目标，第 ' + str(position) + ' 名' if position else '不含目标'}")
        for rank, asin in enumerate(ranked[:3], 1):
            mark = " ★" if asin == target else "  "
            print(f"         {rank}.{mark} {asin}  {short(agent.catalog.title[asin], 46)}")

        ask = response["ask_attribute"]
        if ask:
            reason = "策略换轨(纯分歧度)" if state.strategy_stalled() else "先验×候选池分歧度"
            print(f"  追问   {ask}  ← {reason}")
        else:
            print("  追问   （已问满 4 条约束，不再打扰顾客）")
        print(f"  Agent  {short(response['message'], 66)}")

        if position and override_applied:
            print(f"\n{BAR}")
            print(f" 命中：第 {turn} 轮，第 {position} 名"
                  f"   → 本场 MTTC={turn}  RR={1 / position:.4f}")
            print(f" Token 消耗 {response['usage']['prompt_tokens']}"
                  f" + {response['usage']['completion_tokens']}   全程离线")
            print(BAR)
            return
        if position and not override_applied:
            print("         （改主意尚未发生，按规则本轮不计命中）")

        if turn == official.MAX_TURNS:
            break
        override = full.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            disclosed.add(str(override.get("new_value", "")))
            message = str(override.get("message", ""))
        else:
            message, _, boundary_used = spoken_reply(full, ask, disclosed, boundary_used)

    print(f"\n{BAR}\n 10 轮用尽，未命中\n{BAR}")


def main() -> int:
    parser = argparse.ArgumentParser(description="逐轮播放一场会话")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--scenario", default="intent_override",
                        choices=["buying", "browsing", "intent_override", "boundary"])
    parser.add_argument("--sample", default=None, help="直接指定 sample_id")
    parser.add_argument("--failure", action="store_true",
                        help="挑一场没拿到第 1 名的，用于展示失败案例")
    args = parser.parse_args()

    products = load_products(args.catalog)
    samples = official.load_jsonl(args.dataset)
    agent = Agent(args.catalog)

    if args.sample:
        chosen = next(s for s in samples if s["sample_id"] == args.sample)
    else:
        pool = [s for s in samples if s["scenario_type"] == args.scenario]
        if args.failure:
            results = json.loads(Path("docs/our_results.json").read_text(encoding="utf-8"))
            bad = {r["sample_id"] for r in results["sessions"] if (r["best_rank"] or 99) > 1}
            pool = [s for s in pool if s["sample_id"] in bad] or pool
        chosen = pool[0]

    play(chosen, products, agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
