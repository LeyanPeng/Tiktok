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

# 判定「追问没在收敛」时，比较前几名；连续几轮不变算停滞
STALL_WINDOW = 5
STALL_ROUNDS = 2


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
    last_top: tuple | None = None
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

    def note_result(self, top_picks: list[str]) -> None:
        """记录本轮排在最前面的候选，用于识别「追问失效」。

        这里刻意**不**跟踪候选池的规模。第一版就是那么写的，是错的：
        候选池在第 1 轮类目剪枝之后就固定不变了，按规模判定会永远判成停滞。
        真正在动的是排序结果——所以看的是前几名有没有换人。
        """
        signature = tuple(top_picks[:STALL_WINDOW])
        if self.last_top is not None and signature == self.last_top:
            self.stale_rounds += 1
        else:
            self.stale_rounds = 0
        self.last_top = signature

    def strategy_stalled(self) -> bool:
        """连续两轮问完之后前几名纹丝不动 —— 当前这条追问路线没在收敛。

        触发后 Agent 会切换属性选择方式：从「先验加权」改为「纯分歧度优先」，
        即不再管顾客大概率有没有这类偏好，只挑最能把候选池切开的那个属性问。
        这是运行时的策略换轨，不是参数微调。

        必须已经拿到过信息才允许换轨。一无所有时前几名不动是正常的
        （顾客还没说任何条件），那不是「策略失效」而是「还没开始」——
        在那里换轨会让 boundary 场景去问一个先验极低的属性，白烧一轮。
        实测：不加这个前提时 MTTC 1.565 → 1.575。
        """
        return bool(self.slots) and self.stale_rounds >= STALL_ROUNDS
