"""候选打分与排序。

核心立场：**纯加性打分，全程不做硬过滤**（类目剪枝除外，它的落桶率实测 200/200）。

理由来自 ProductAgent 的教训：把用户说的条件全部 AND 起来做结构化过滤，
听着严谨，实际后期 55% 的查询返回空集——正确答案被自己滤掉了。
加性打分让每个条件只是「加分项」，说错一个不至于全盘皆输。

两级信号：
  1. 原句子串命中 —— 约束是从商品 features/details 原样切下来的，等于拿到商品指纹，权重极高
  2. IDF 加权词覆盖 —— 原句没对上时的软兜底，罕见词命中比常见词更值钱
"""

from __future__ import annotations

from .catalog import Catalog, tokenize

# 原句在商品文本里被原样找到时的基础分。
# 取得远高于词覆盖的上限（3.0），确保「指纹命中」永远压过「词碰巧撞上」。
VERBATIM_BASE = 12.0

# 原句越长越独特，每多一个词额外加一点
VERBATIM_PER_WORD = 0.35

# 词覆盖率的满分。刻意压低——它只是兜底，不该和指纹信号抢话语权
COVERAGE_WEIGHT = 3.0


class Ranker:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    def score_one(self, asin: str, constraints: list[tuple[str, float]]) -> float:
        blob = self.catalog.blob[asin]
        total = 0.0
        for text, weight in constraints:
            needle = text.strip().lower()
            if not needle:
                continue
            if needle in blob:
                total += weight * (VERBATIM_BASE + VERBATIM_PER_WORD * len(needle.split()))
                continue
            # 兜底：这条约束的词，有多少比例出现在商品文本里（按 IDF 加权）
            terms = tokenize(needle)
            if not terms:
                continue
            token_set = self.catalog.tokens[asin]
            hit = sum(self.catalog.idf.get(t, 0.0) for t in terms if t in token_set)
            want = sum(self.catalog.idf.get(t, 0.0) for t in terms)
            if want > 0:
                total += weight * COVERAGE_WEIGHT * (hit / want)
        return total

    def rank(
        self,
        candidates: list[str],
        constraints: list[tuple[str, float]],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """返回 (asin, score) 前 top_k 条，按分数降序。

        没有任何约束时也必须返回满 top_k 条——每一轮都是一次免费的命中机会，
        空着不交等于白白浪费一轮（MTTC 会因此变差）。
        """
        prior = self.catalog.prior
        if not constraints:
            # Browsing 轨第一轮：顾客还没说任何条件，按目录顺序返回等于浪费一轮。
            # 零信息下最好的猜测是人气最高的商品（item popularity 基线）。
            ordered = sorted(candidates, key=lambda a: -prior.get(a, 0.0))
            return [(asin, 0.0) for asin in ordered[:top_k]]

        scored = [(asin, self.score_one(asin, constraints)) for asin in candidates]

        # 两级严格字典序：先按约束得分，完全打平才看人气先验。
        # 这样人气永远不可能盖过约束信号，只在没有信号可区分时才发言。
        #
        # 这里刻意**没有**更复杂的拆并列逻辑。试过三种，全部实测为负或无效：
        #   - 稀有度做乘数            0.891 -> 0.876（目标只匹配常见约束时信号被打死）
        #   - 稀有度做次级排序键      0.891 -> 0.891（第1轮只有1条约束时，并列各方稀有度相同，拆不开）
        #   - 字段加权做次级排序键    0.891 -> 0.881
        # 详细归因见 PROGRESS.md 与技术报告。留一个简单且已验证的版本，胜过留一个复杂且更差的。
        scored.sort(key=lambda pair: (-pair[1], -prior.get(pair[0], 0.0)))
        return scored[:top_k]
