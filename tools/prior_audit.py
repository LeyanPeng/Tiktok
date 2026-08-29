"""核验 · 从目录推导的追问先验，与公开集实测的约束分布是否一致？

背景：`TYPE_PRIOR` 曾是一组写死的数（feature .505 / material .378 / …），
来自公开集 200 场 800 条约束的实测。敏感度扫描显示它**真的有杠杆**：

    实测先验  0.891142      颠倒先验  0.791952  (-0.099)
    全均匀    0.882679                        (-0.008)

先验错了的代价远大于先验对了的收益。而写死的数只在「私有集分布与公开集相同」
时才成立 —— 官方从未保证这一点。

现在改为在**整个目录**上按模拟器自己的选择逻辑推导（见 catalog.constraint_candidates），
算的是总体分布而非 200 场的样本。本脚本验证这个推导没有跑偏：
和公开集实测比，**相对次序必须完全一致**（次序决定 argmax，也就决定问哪个属性）。

这是 rule_parity 的同类：一处复刻了官方逻辑的地方，一旦悄悄走样，没人会发现。
"""

from __future__ import annotations

import sys

from src.askpolicy import derive_type_prior
from src.catalog import Catalog

# 公开集 200 场 800 条约束的实测分布，仅作对照基准，不参与运行时决策
MEASURED = {
    "feature": 0.505, "material": 0.378, "color": 0.075,
    "style": 0.024, "size": 0.014, "use_case": 0.005,
}
MAX_ABS_DEVIATION = 0.15        # 单项幅度容差；次序才是硬要求


def main() -> int:
    catalog = Catalog("data/catalog.jsonl")
    derived = derive_type_prior(catalog)

    names = list(MEASURED)
    order_derived = sorted(names, key=lambda k: -derived[k])
    order_measured = sorted(names, key=lambda k: -MEASURED[k])
    worst = max(abs(derived[k] - MEASURED[k]) for k in names)

    print("追问先验一致性核验")
    print(f"  {'类型':<12}{'目录推导':>12}{'公开集实测':>14}{'差':>10}")
    for k in order_measured:
        print(f"  {k:<12}{derived[k]:>12.4f}{MEASURED[k]:>14.4f}{derived[k]-MEASURED[k]:>+10.4f}")
    print()
    print(f"  推导次序  {' > '.join(order_derived)}")
    print(f"  实测次序  {' > '.join(order_measured)}")
    print()

    checks = [
        ("相对次序完全一致（决定问哪个属性）", order_derived == order_measured,
         "一致" if order_derived == order_measured else "已偏离"),
        (f"单项幅度偏差 < {MAX_ABS_DEVIATION}", worst < MAX_ABS_DEVIATION, f"最大偏差={worst:.4f}"),
    ]
    ok = True
    for name, passed, detail in checks:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}   {detail}")
    if not ok:
        print("\n  推导已偏离实测分布，追问顺序会跟着错，需要检查 catalog.constraint_candidates。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
