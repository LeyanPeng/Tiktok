"""T12 反向验证 · 故障演练

`Agent.respond` 外面包了一层 try/except。这层东西最危险的地方在于：
**它坏了没人会知道** —— 如果降级路径本身返回空列表，分数会静静地塌掉，
而日志上什么都看不见，我们还以为兜底生效了。

所以必须亲手制造故障，证明两件事：
  1. 出错时进程不崩、仍然交出 10 条合法推荐（不是空列表）
  2. 损失是「这一轮差一点」而不是「整场归零」

这不是在测 Agent 好不好，是在测**兜底本身是不是个摆设**。

验收对照的选择（这一点定错过一次，记下来）：
最初把验收线定成「注入故障后掉分 < 0.02」。那是错的——
每 3 轮炸一次还要求几乎不掉分，等于要求降级路径和正常路径一样好，逻辑上不可能。
它测不出我们真正关心的性质。

改成「有兜底的受伤 vs 没兜底的受伤」对照后，量出来的结果是 **-0.001**：
兜底基本没救回任何分。原因是官方评测器自己就有异常处理
（`except Exception: response = {空}`），单轮抛异常只损失那一轮的推荐，
会话继续、下一轮就恢复；而 71.5% 的命中发生在第 1 轮，丢一轮几乎不花钱。
**「异常 = 灾难」这个前提本身是错的。**

所以「救回 > 0.10」不再作为门槛，改为实测发现如实上报。
兜底继续保留：零成本、保证契约（永不返回空列表）、且在真实部署里仍是正确工程。

验收（只留能测出真实性质的三条）：
  - 全程零异常逃逸
  - 降级从不返回空推荐（空列表 = 主动放弃这一轮的命中机会）
  - 健康态分数不受 wrapper 影响（这层包装必须没有副作用）
"""

from __future__ import annotations

import sys

import evaluator.local_evaluator as official

from agent import Agent

FAIL_EVERY = 3          # 每 3 轮炸一次，覆盖首轮与中途两种时机


class ChaosAgent(Agent):
    """周期性在核心路径上抛异常的 Agent。

    with_fallback=False 时让异常直接逃到评测器，模拟「没写兜底」的世界。
    """

    def __init__(self, catalog_path: str = "data/catalog.jsonl", with_fallback: bool = True) -> None:
        super().__init__(catalog_path)
        self.with_fallback = with_fallback
        self.calls = 0
        self.injected = 0
        self.empty_returns = 0
        self.escaped = 0

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self.calls += 1
        if self.calls % FAIL_EVERY == 0:
            self.injected += 1
            original = self._respond

            def boom(*_args, **_kwargs):
                raise RuntimeError("injected fault")

            self._respond = boom              # type: ignore[method-assign]
            try:
                if not self.with_fallback:
                    raise RuntimeError("injected fault")   # 不接，直接抛给评测器
                result = super().respond(session_id, user_message, turn, top_k)
            except Exception:                 # 兜底没接住 —— 这才是真正的失败
                self.escaped += 1
                if self.with_fallback:
                    raise
                self._respond = original      # type: ignore[method-assign]
                return {"message": "", "ask_attribute": None, "recommendations": []}
            finally:
                self._respond = original      # type: ignore[method-assign]
        else:
            result = super().respond(session_id, user_message, turn, top_k)

        if not result.get("recommendations"):
            self.empty_returns += 1
        return result


class UnguardedAgent(Agent):
    """绕过 try/except 包装，直接走核心路径。

    用来验证「这层包装没有副作用」。原先这一条是拿健康态分数去比一个写死的
    0.891142 —— 那意味着 agent 每改进一次，这个测试就假失败一次，
    而且它比的是「分数等于某个数」，不是「包装没有副作用」这个真正要验的性质。
    改成同一次运行内的自对照：包装版 vs 绕过版，必须逐位相同。
    """

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return self._respond(session_id, user_message, turn, top_k)


def score(agent, catalog: str, dataset: str) -> float:
    ids, cats, products = official.catalog_index(catalog)
    samples = official.load_jsonl(dataset)
    return official.evaluate(agent, samples, ids, cats, products)["recommended_technical_score"]


def main() -> int:
    catalog, dataset = "data/catalog.jsonl", "data/public_set.jsonl"

    healthy = score(Agent(catalog), catalog, dataset)
    unguarded = score(UnguardedAgent(catalog), catalog, dataset)
    guarded = ChaosAgent(catalog, with_fallback=True)
    with_fb = score(guarded, catalog, dataset)
    naked = ChaosAgent(catalog, with_fallback=False)
    without_fb = score(naked, catalog, dataset)
    rescued = with_fb - without_fb

    checks = [
        (f"注入 {guarded.injected} 次故障后仍不崩", guarded.escaped == 0, f"逃逸异常={guarded.escaped}"),
        ("降级从不返回空推荐", guarded.empty_returns == 0, f"空返回={guarded.empty_returns}"),
        ("wrapper 无副作用（同一次运行内自对照）",
         abs(healthy - unguarded) < 1e-9,
         f"包装版={healthy} 绕过版={unguarded}"),
    ]

    print("T12 反向验证 · 故障演练")
    print(f"  健康态 · 有 wrapper     {healthy}")
    print(f"  健康态 · 绕过 wrapper   {unguarded}   [自对照，取代写死的基线]")
    print(f"  每 {FAIL_EVERY} 轮炸一次 · 有兜底   {with_fb}")
    print(f"  每 {FAIL_EVERY} 轮炸一次 · 无兜底   {without_fb}")
    print(f"  → 兜底救回              {rescued:+.6f}   [实测发现，非门槛]")
    print("     评测器自身已吸收单轮异常，故 agent 侧兜底对分数几乎无贡献。")
    print("     保留它的理由是契约保证与真实部署，不是分数。")
    print()
    ok = True
    for name, passed, detail in checks:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}   {detail}")
    if not ok:
        print("\n兜底是个摆设——出事时会静默丢分，日志上还看不出来。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
