"""T5 反向验证：测试台本身是不是个假绿灯？

一个永远给高分的测试台，比没有测试台更危险——它会让我们在错误的方向上加速。
所以先证明它会报警：塞一个必然失败的 Agent 进去，分数必须塌到接近 0。

问「这里坏了谁会知道」，答案是「没人」的地方，都要有这一步。

验收：哑 Agent 在原版和扰动版两条测试台上，TechnicalScore 都 < 0.05
"""

from __future__ import annotations

import sys

import evaluator.local_evaluator as official

from tools.run_eval import install_paraphrase

THRESHOLD = 0.05


class BrokenAgent:
    """什么都不干的 Agent。分数必须塌。"""

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

    checks = [("原版测试台会报警", plain), ("扰动版测试台会报警", perturbed)]
    print("T5 反向验证 · 测试台是否会响")
    ok = True
    for name, value in checks:
        passed = value < THRESHOLD
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
              f"   哑 Agent 得分={value} (要求 < {THRESHOLD})")
    if not ok:
        print("\n测试台没有报警——它是个假绿灯，后续所有分数都不可信。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
