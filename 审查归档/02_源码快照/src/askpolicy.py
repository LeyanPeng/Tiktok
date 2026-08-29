"""追问决策：问哪个属性，能把候选池切得最开。

这是整套系统里最值钱的一块，来自 ProductAgent 的实证：
追问选项凭空生成 vs 从当前候选池的统计里长出来，HIT@10 差 3 倍（15.60 → 47.00）。
那篇论文里没有任何别的选择带来过这么大的落差。

原理是二十问游戏的最优策略：问一个所有候选都一样的属性，信息增益为零；
只有问「候选池里分歧最大」的属性，才真的在缩小搜索空间。

期望收益 = P(顾客真有这类约束) × H(该属性在当前候选池上的取值熵)

  - 左项来自 T1 在 200 场 800 条约束上的实测分布，不是拍脑袋
  - 右项来自当前**还活着的**候选池，每轮重算 —— 这就是「接地到统计」
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .catalog import Catalog

# 顾客的约束落在各类型上的先验概率 —— **从只读目录现算，不写死**。
#
# 早先这里是一组写死的数（feature .505 / material .378 / …），来自公开集 200 场
# 800 条约束的实测。敏感度扫描显示这个先验**真的有杠杆**，而且风险形状很不对称：
#
#     实测先验    0.891142      颠倒先验    0.791952   (-0.099)
#     全均匀      0.882679                            (-0.008)
#
# 先验**错了**的代价（-0.099）远大于先验**对了**的收益（+0.008）。
# 而写死的数只在「私有集分布与公开集相同」时才对——官方从未保证这一点。
#
# 所以改为两步：
#   1. 从目录自己的商品文案里统计各类型片段的占比（私有集用同一份目录，推导必然适用）
#   2. 向均匀分布平滑，给最坏情况封顶 —— 既拿到次序信息，又不赌分布精确
#
# 目录推导 vs 公开集实测（见 tools/prior_audit.py）：相对次序一致，
# 只有 material 被低估（.140 vs .378），因为模拟器刻意把材质插在第 0 位。
# 平滑正是为这种「次序对、幅度不准」准备的。

# 无法从商品文案统计出来的属性（brand / budget / category），给一个极低的底数，
# 避免它们的先验为 0 而永远不被选中。
FALLBACK_PRIOR = 0.001

# 这里曾经有一个 PRIOR_SMOOTHING（向均匀分布平滑）的旋钮，已删除：
# 在 0.00 / 0.25 / 0.50 / 0.75 四档上实测，公开集与扰动集分数**完全不变**。
# 原因是平滑保序，而追问选的是 argmax —— 一个可证明什么都不做的参数就是噪音。

# 各属性的取值词表。与官方 classify_constraint 使用同一批词，
# 保证我们判断的「属性」和模拟器判断的是同一个东西。
VOCAB: dict[str, tuple[str, ...]] = {
    "material": ("cotton", "polyester", "nylon", "leather", "wool",
                 "spandex", "silk", "rayon", "fabric"),
    "color": ("black", "white", "blue", "red", "pink", "green",
              "brown", "gray", "grey", "purple", "yellow", "orange"),
    "size": ("size", "sizing", "width", "wide", "narrow"),
    "style": ("department", "style", "fit", "sleeve", "neck"),
    "use_case": ("hiking", "running", "gym", "winter", "outdoor", "work"),
}
PATTERNS = {
    name: re.compile(r"\b(" + "|".join(words) + r")\b")
    for name, words in VOCAB.items()
}

# 候选池可能上千件，全量统计不值得。抽样足够估出分布形状。
SAMPLE_CAP = 400

# feature 是自由文本，枚举不出取值，无法算熵。
# 但它占了全部约束的一半，且原句最长最独特——给一个保守的固定分歧度。
FEATURE_DIVERSITY = 0.70


def _normalised_entropy(values: list[str | None]) -> float:
    """取值分布的归一化熵，落在 [0, 1]。

    全都一样 → 0（问了也白问）；均匀分散 → 1（问了最能切开）。
    """
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return entropy / math.log2(len(counts))


def derive_type_prior(catalog: Catalog) -> dict[str, float]:
    """从目录现算「顾客会提哪类约束」的分布。

    立论：模拟器的顾客话术是从商品 features/details 里切出来的，
    所以「顾客会提哪类约束」的分布，应当跟着「商品文案里有哪类片段」走。
    目录是只读的、公开集和私有集共用同一份 —— 从它推导，就不存在「私有集不一样」的风险。
    """
    counts = catalog.span_types          # 目录加载时按模拟器的选择逻辑统计好的
    total = sum(counts.values()) or 1
    names = [*VOCAB, "feature", "brand", "budget", "category"]
    return {name: max(FALLBACK_PRIOR, counts.get(name, 0) / total) for name in names}


class AskPolicy:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        # 先验现算，不用写死的数。构建一次，全程复用。
        self.type_prior = derive_type_prior(catalog)

    def diversity(self, candidates: list[str], attribute: str) -> float:
        """这个属性，在当前候选池上有多分歧。"""
        if attribute == "feature":
            return FEATURE_DIVERSITY
        pattern = PATTERNS.get(attribute)
        if pattern is None:
            return 0.0
        pool = candidates[:SAMPLE_CAP]
        observed = [
            (match.group(1) if (match := pattern.search(self.catalog.blob[a])) else None)
            for a in pool
        ]
        return _normalised_entropy(observed)

    def choose(
        self,
        candidates: list[str],
        asked: list[str],
        fallback_order: tuple[str, ...],
        ignore_prior: bool = False,
    ) -> str | None:
        """选期望信息增益最大的未问属性。

        ignore_prior=True 时丢掉「顾客大概率有没有这类偏好」这一项，
        只看哪个属性最能把候选池切开。由停滞检测触发，见 SessionState.strategy_stalled。
        """
        best, best_score = None, 0.0
        for attribute, prior in self.type_prior.items():
            if attribute in asked:
                continue
            weight = 1.0 if ignore_prior else prior
            score = weight * self.diversity(candidates, attribute)
            if score > best_score:
                best, best_score = attribute, score

        if best is not None:
            return best
        # 所有可估属性都问过了，退回固定顺序兜底
        for attribute in fallback_order:
            if attribute not in asked:
                return attribute
        return None
