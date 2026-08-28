"""会话状态机：增量槽位累积 + 意图覆盖重写 + 轮次预算。

对应赛题支柱 II「Dialog Strategy: Multi-Turn Scenario Evolution」。

关于覆盖重写的一个刻意设计：顾客说「不要之前那个」时，我们**降权**旧槽位而不是删除。
理由是本题的全部约束都源自同一件目标商品——旧信息依然指向正确答案，
直接抹掉等于自伤召回。降权既表达了「新意图优先」，又不丢证据。
这个取舍会写进技术报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .extract import (
    Verifier,
    detect_override,
    extract_constraints,
    is_browsing_opener,
    match_category,
)

# 覆盖发生后，改主意之前说过的约束保留多少权重
PRE_OVERRIDE_DECAY = 0.35

# 顾客一场最多吐露 4 条约束（实测 200/200 场次恒为 4），
# 据此判断「还值不值得再问一轮」
MAX_CONSTRAINTS = 4


@dataclass
class Slot:
    text: str
    turn: int
    weight: float = 1.0
    source: str = "stated"      # stated | override


@dataclass
class SessionState:
    session_id: str
    profile: dict = field(default_factory=dict)
    category: str | None = None
    slots: list[Slot] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    override_turn: int | None = None
    intent: str = "browsing"    # browsing | buying
    last_pool_size: int | None = None
    stale_rounds: int = 0       # 候选池连续多少轮没缩小
    last_recommendations: list = field(default_factory=list)  # 降级时交出上一轮结果

    # ── 观察 ────────────────────────────────────────────────────────
    def observe(
        self,
        message: str,
        turn: int,
        categories_by_length: list[str],
        verify: "Verifier | None" = None,
    ) -> None:
        if turn == 1:
            self.category, _ = match_category(message, categories_by_length)

        if detect_override(message) and self.override_turn is None:
            self.override_turn = turn
            for slot in self.slots:                     # 旧槽位降权，不删除
                slot.weight *= PRE_OVERRIDE_DECAY

        found = extract_constraints(message, self.category, verify)
        known = {s.text.lower() for s in self.slots}
        for text in found:
            if text.lower() not in known:
                self.slots.append(Slot(
                    text=text,
                    turn=turn,
                    source="override" if self.override_turn == turn else "stated",
                ))
                known.add(text.lower())

        if turn == 1:
            # 开场就带硬约束 = Buying 轨；开场泛化 = Browsing 轨
            self.intent = "browsing" if (is_browsing_opener(message) or not self.slots) else "buying"

    # ── 查询 ────────────────────────────────────────────────────────
    def constraints(self) -> list[tuple[str, float]]:
        return [(s.text, s.weight) for s in self.slots]

    def saturated(self) -> bool:
        """已经问干净了，再问也榨不出新信息。"""
        return len(self.slots) >= MAX_CONSTRAINTS

    def note_pool(self, size: int) -> None:
        """记录候选池规模，用于识别「追问失效」。"""
        if self.last_pool_size is not None and size >= self.last_pool_size:
            self.stale_rounds += 1
        else:
            self.stale_rounds = 0
        self.last_pool_size = size

    def strategy_stalled(self) -> bool:
        """连续两轮候选池没缩小 —— 当前追问策略失效，该换路了。"""
        return self.stale_rounds >= 2
