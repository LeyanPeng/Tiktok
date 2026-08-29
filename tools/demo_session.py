"""Replay one complete session turn by turn.

The official deliverables ask for "one demonstrated multi-turn session". This
script lays out everything that happens inside the evaluator: what the customer
said, what we extracted, how the candidate pool converges, why a particular
attribute is asked next, and where the hidden target ranks.

Usage:
    python -m tools.demo_session                       # an intent_override session
    python -m tools.demo_session --scenario buying
    python -m tools.demo_session --sample public_0002
    python -m tools.demo_session --scenario browsing --failure   # a session we get wrong
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
    return text if len(text) <= width else text[: width - 1] + "..."


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
    print(f" session {sample['sample_id']}   scenario {sample['scenario_type']}"
          f"   difficulty {sample.get('difficulty_bucket', '?')}")
    print(f" hidden target   {target}  {short(products[target].get('title'), 46)}")
    print(f" category bucket {category!r}   {len(agent.catalog.candidates(category))} items"
          f"   (catalog holds {len(agent.catalog)})")
    print(f" customer profile {short(sample['user_profile'].get('summary'), 58)}")
    print(BAR)

    session_id = f"demo_{sample['sample_id']}"
    agent.reset(session_id, sample["user_profile"])
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"

    message, _ = spoken_initial(full, category, disclosed)
    for turn in range(1, official.MAX_TURNS + 1):
        print()
        print(f" Turn {turn}")
        print(SEP)
        print(f"  customer   {short(message, 64)}")

        before = len(agent._sessions[session_id].slots)
        response = agent.respond(session_id, message, turn, official.TOP_K)
        state = agent._sessions[session_id]

        picked_up = [s.text for s in state.slots[before:]]
        if picked_up:
            for text in picked_up:
                print(f"  extracted  + {short(text, 60)}")
        else:
            print("  extracted  (nothing new this turn)")

        if state.override_turn == turn:
            print("  state      intent override detected -> earlier slots decayed x0.35")
            print("             (decayed, not erased: they still point at the same product)")

        ranked = [r["parent_asin"] for r in response["recommendations"]]
        position = ranked.index(target) + 1 if target in ranked else None
        where = f"target at rank {position}" if position else "target not in top 10"
        print(f"  ranking    {len(state.slots)} constraints known"
              f" | track {state.intent} | {where}")
        for rank, asin in enumerate(ranked[:3], 1):
            mark = " *" if asin == target else "  "
            print(f"             {rank}.{mark} {asin}  {short(agent.catalog.title[asin], 44)}")

        ask = response["ask_attribute"]
        if ask:
            reason = ("strategy switch: pure pool divergence" if state.strategy_stalled()
                      else "prior x live pool divergence")
            print(f"  asks       {ask}  <- {reason}")
        else:
            print("  asks       (nothing further; the customer has stopped contributing)")
        print(f"  agent      {short(response['message'], 64)}")

        if position and override_applied:
            print()
            print(BAR)
            print(f" Converted on turn {turn} at rank {position}"
                  f"   ->  MTTC={turn}  RR={1 / position:.4f}")
            print(f" Tokens used {response['usage']['prompt_tokens']}"
                  f" + {response['usage']['completion_tokens']}   fully offline")
            print(BAR)
            return
        if position and not override_applied:
            print("             (the override has not happened yet, so this does not count)")

        if turn == official.MAX_TURNS:
            break
        override = full.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            disclosed.add(str(override.get("new_value", "")))
            message = str(override.get("message", ""))
        else:
            message, _, boundary_used = spoken_reply(full, ask, disclosed, boundary_used)

    print()
    print(BAR)
    print(" Ten turns exhausted without converting")
    print(BAR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay one session turn by turn")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--scenario", default="intent_override",
                        choices=["buying", "browsing", "intent_override", "boundary"])
    parser.add_argument("--sample", default=None, help="pick a specific sample_id")
    parser.add_argument("--failure", action="store_true",
                        help="pick a session where the target did not reach rank 1")
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
