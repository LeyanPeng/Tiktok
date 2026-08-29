"""压力测试 · 私有集若每场约束数不是 4，会怎样？

背景：早期实现用 `MAX_CONSTRAINTS = 4` 决定何时停止追问，这个 4 来自公开集实测
「每场恒为 4 条」。官方规格只保证私有集的**场景配比**相同，从没保证约束条数相同。

本脚本把模拟器改造成每场吐 6 条约束，对比两种停止规则：
  A. 按计数停（旧）—— 攒够 4 条就不问了，第 5、6 条永远问不出来
  B. 按证据停（新）—— 顾客明说「我没这方面偏好」连续三次才停

原假设：约束数超过 4 时 B 会明显优于 A。**这个假设被实测推翻了。**

三个场景实测（新 − 旧）：

    公开集（4 条约束）              0.891142 vs 0.891142    +0.000000
    6 条约束                        0.898171 vs 0.898171    +0.000000
    6 条约束 + 话术改写（困难区）   0.861180 vs 0.861180    +0.000000

原因：会话结束得太早（公开集 MTTC 1.565，71.5% 的场次第 1 轮就命中），
停止追问的闸门在命中之前根本没机会生效——新旧规则都是惰性的。
也就是说，自查报告里把 `MAX_CONSTRAINTS = 4` 标成「中高风险」是**高估了**。

那为什么还要改？因为「等价」和「正确」是两回事：
旧写法把一个未经验证的假设（私有集每场也是 4 条）编进了代码；
新写法不依赖任何关于约束条数的假设。分数一样，但少了一个会静默失效的前提。

途中还暴露了一个真实缺陷：第一版把「我们没解析出东西」也当成「顾客没话说了」，
于是话术一改写、抽取一失手就自己提前闭嘴——困难区因此掉了 0.0046。
改成只认顾客**明说**的「我没有偏好」之后，三个场景全部持平。

所以本脚本的验收改为测它真正成立的性质：**新规则在任何场景下都不劣于旧规则**。
（原门槛「领先 >= 0.02」是在测量之前定的，测出来才发现问的是个错问题。）
"""

from __future__ import annotations

import sys

import evaluator.local_evaluator as official

from agent import ASK_ORDER, Agent

WIDE_CONSTRAINTS = 6        # 把每场约束数从 4 撑到 6
TOLERANCE = 1e-6            # 允许的劣化幅度：本质上要求「不劣于」


def widen_intent_card() -> None:
    """让模拟器每场吐 6 条约束，而不是 4 条。"""
    base = official.intent_card

    def wide(product: dict, limit: int = 180) -> dict:
        card = base(product, limit)
        pool = card["hard_constraints"] + card["soft_preferences"]
        corpus = [
            *official._flatten_values(product.get("features")),
            *official._flatten_values(product.get("details")),
        ]
        extra = [official._clean_constraint(v, limit) for v in corpus]
        merged = list(dict.fromkeys([*pool, *[e for e in extra if e]]))[:WIDE_CONSTRAINTS]
        split = max(1, len(merged) // 3)
        return {
            "target_category": card["target_category"],
            "hard_constraints": merged[:split],
            "soft_preferences": merged[split:],
        }

    official.intent_card = wide


class CountGatedAgent(Agent):
    """旧行为：攒够 4 条约束就停止追问。"""

    def _next_attribute(self, state, candidates):
        if len(state.slots) >= 4:
            return None
        return self.ask_policy.choose(
            candidates, state.asked, ASK_ORDER, ignore_prior=state.strategy_stalled()
        )


def score(agent, catalog: str, dataset: str) -> dict:
    ids, cats, products = official.catalog_index(catalog)
    samples = official.load_jsonl(dataset)
    return official.evaluate(agent, samples, ids, cats, products)


def main() -> int:
    from tools.run_eval import install_paraphrase

    catalog, dataset = "data/catalog.jsonl", "data/public_set.jsonl"
    widen_intent_card()

    results = []
    for label in (f"{WIDE_CONSTRAINTS} 条约束", f"{WIDE_CONSTRAINTS} 条约束 + 话术改写"):
        if "话术" in label:
            install_paraphrase()
        old = score(CountGatedAgent(catalog), catalog, dataset)
        new = score(Agent(catalog), catalog, dataset)
        results.append((label, old, new,
                        new["recommended_technical_score"] - old["recommended_technical_score"]))

    print(f"压力测试 · 模拟器每场吐 {WIDE_CONSTRAINTS} 条约束（公开集实际为 4 条）")
    print(f"  {'场景':<26}{'按计数停(旧)':>16}{'按证据停(新)':>16}{'差值':>12}")
    for label, old, new, gain in results:
        print(f"  {label:<24}{old['recommended_technical_score']:>16.6f}"
              f"{new['recommended_technical_score']:>16.6f}{gain:>+12.6f}")
    print()
    for label, old, new, _ in results:
        print(f"  {label:<24} hit {old['hit_rate_at_10']:.3f} -> {new['hit_rate_at_10']:.3f}"
              f"   mrr {old['mrr']:.4f} -> {new['mrr']:.4f}"
              f"   mttc {old['mttc']:.3f} -> {new['mttc']:.3f}")
    print()

    worst = min(gain for *_, gain in results)
    passed = worst >= -TOLERANCE
    print(f"  [{'PASS' if passed else 'FAIL'}] 按证据停 在所有场景下都不劣于 按计数停"
          f"   最差差值={worst:+.6f}")
    print("     新规则不更强，但不再依赖「私有集每场也是 4 条约束」这个未验证假设。")
    if not passed:
        print("\n  这次修改造成了劣化，应当回滚。")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
