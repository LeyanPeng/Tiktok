"""提交入口：TechJam Track 4 对话购物 Agent。

设计原则（详见 README）：
  - **完全离线**。无网络、无外部 API、无向量库，只用 Python 标准库。
    依据是官方提交规则：最终评分时可能关闭网络访问。
  - **纯加性打分**，除类目剪枝外不做任何硬过滤。
  - 每一轮都返回满 10 条推荐——每轮都是一次免费的命中机会。

接口遵循 docs/agent_api_contract.json。
"""

from __future__ import annotations

from pathlib import Path

from src.askpolicy import AskPolicy
from src.catalog import Catalog
from src.ranker import Ranker
from src.session_state import SessionState

# 追问顺序按 T1 实测的信息量排序：
# feature 50.5% / material 37.8% / color 7.5% / style 2.4% / size 1.4% / use_case 0.5%
# 先问信息量最大的，这就是赛题 Innovation Directions 里说的 question-value estimation。
ASK_ORDER = (
    "feature", "material", "color", "style",
    "size", "use_case", "budget", "brand", "category",
)


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog = Catalog(catalog_path)
        self.ranker = Ranker(self.catalog)
        self.ask_policy = AskPolicy(self.catalog)
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(
            session_id=session_id,
            profile=user_profile or {},
        )

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        """官方评测器规定：抛异常、输出非法、超时，都按 miss 计。

        所以这里包一层总兜底。它**不是**用来掩盖 bug 的——
        内部所有验收和自测都跑在未包裹的 _respond 上，出错会照常炸出来；
        这一层只负责保证：万一评测环境里出现我们没预料到的输入，
        损失是「这一轮排序差一点」，而不是「整场归零」。
        降级路径本身有反向验证，见 tools/failure_drill.py。
        """
        try:
            return self._respond(session_id, user_message, turn, top_k)
        except Exception:
            return self._fallback(session_id, top_k)

    def _fallback(self, session_id: str, top_k: int) -> dict:
        """降级：交出上一轮的推荐；连上一轮都没有就交该类目下最热门的。

        绝不返回空列表——空列表等于主动放弃这一轮的命中机会。
        """
        state = self._sessions.get(session_id)
        if state is not None and state.last_recommendations:
            picks = state.last_recommendations[:top_k]
        else:
            category = state.category if state is not None else None
            picks = self.catalog.top_popular(category, top_k)
        return {
            "message": "Let me show you some popular options while I narrow things down.",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": asin} for asin in picks],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:                      # 防御：reset 没被调用也不能崩
            self.reset(session_id, {})
            state = self._sessions[session_id]

        # 用当前候选池当裁判，消解约束切分歧义（见 src/extract._split_clause）
        state.observe(
            user_message,
            turn,
            self.catalog.categories_by_length,
            verify=lambda text: self.catalog.contains_verbatim(text, state.category),
        )

        candidates = self.catalog.candidates(state.category)
        state.note_pool(len(candidates))
        ranked = self.ranker.rank(candidates, state.constraints(), top_k)

        ask = self._next_attribute(state, candidates)
        if ask:
            state.asked.append(ask)

        picks = [asin for asin, _ in ranked]
        state.last_recommendations = picks      # 供异常降级使用

        return {
            "message": self._phrase(ask, state),
            "ask_attribute": ask,
            "recommendations": [{"parent_asin": asin} for asin in picks],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    # ── 追问策略 ────────────────────────────────────────────────────
    def _next_attribute(self, state: SessionState, candidates: list[str]) -> str | None:
        if state.saturated():        # 4 条约束已问全，再问榨不出新东西
            return None
        # 每轮对**当前还活着的**候选池重算各属性分歧度，选期望信息增益最大的那个。
        # 走死顺序等于假设所有场次的候选池长得一样，那显然不成立。
        return self.ask_policy.choose(candidates, state.asked, ASK_ORDER)

    @staticmethod
    def _phrase(attribute: str | None, state: SessionState) -> str:
        """给顾客看的自然语言。

        注意：官方模拟器只读 ask_attribute 字段，这段话不参与打分。
        写好它是为了 Demo 和评委，不是为了分数——所以不在这里花时间调优。
        """
        if attribute is None:
            return "I think I've got what I need — here are my top picks for you."
        phrasing = {
            "feature": "Is there a specific feature or detail that matters most to you?",
            "material": "Do you have a material preference?",
            "color": "Any colour you're leaning towards?",
            "style": "What style or fit are you after?",
            "size": "Any sizing requirements I should know about?",
            "use_case": "What will you mainly be using it for?",
            "budget": "Roughly what budget did you have in mind?",
            "brand": "Any brand you prefer or want to avoid?",
            "category": "Could you narrow down the kind of item you're after?",
        }
        return phrasing.get(attribute, "Could you tell me a bit more about what you need?")
