"""从顾客消息里抽出可用信息：类目、约束原句、改主意信号。

设计立场：抽取层只负责「听懂顾客说了什么」，不做任何过滤或判断。
凡是拿不准的，一律交给下游打分层用软权重消化——
硬过滤会把正确答案一起滤掉（ProductAgent 的 Text2SQL 后期 55% 查空就是这么来的）。
"""

from __future__ import annotations

import re
from typing import Callable

from .catalog import TOKEN_RE, tokenize

# 判定「这一串在只读目录里能否原文找到」的回调，由候选池提供
Verifier = Callable[[str], bool]

# 顾客表达约束时的引导语。按出现频率排序，命中即停。
CONSTRAINT_MARKERS = (
    "what matters is:",      # For that, what matters is: A; B.
    "what i need is:",       # Actually, ignore my earlier preference. What I need is: X.
    "key requirement is:",   # I'm looking for CAT. A key requirement is: X.
    "requirement is:",
    "matters is:",
    " is: ",                 # 兜底：任何 "…is: X" 结构
)

# 明确表示「没有偏好」的回复，抽不出任何约束
NEGATIVE_PATTERNS = (
    re.compile(r"don'?t have (?:an? )?(?:additional )?preference", re.I),
    re.compile(r"use your judg[e]?ment", re.I),
    re.compile(r"not quite right yet", re.I),
    re.compile(r"ask me about one specific attribute", re.I),
)

# 开场白里表示「还在逛」的信号
BROWSING_PATTERNS = (
    re.compile(r"still exploring", re.I),
    re.compile(r"just browsing", re.I),
    re.compile(r"nothing fixed yet", re.I),
    re.compile(r"not sure yet", re.I),
)

# 改主意信号。命中后旧槽位降权，不做删除——旧信息仍指向同一件商品。
OVERRIDE_PATTERNS = (
    re.compile(r"ignore my earlier", re.I),
    re.compile(r"actually,? (?:ignore|forget|scratch)", re.I),
    re.compile(r"forget what i said", re.I),
    re.compile(r"changed my mind", re.I),
    re.compile(r"instead,? what i (?:really )?(?:need|want)", re.I),
)

# 开场白里「我在找 X」的引导语，用于切出类目片段
LOOKING_FOR_RE = re.compile(
    r"(?:looking for|want to find|show me|searching for|need)\s+(?:some\s+|a\s+|an\s+)?(.+)",
    re.I,
)


def is_negative(message: str) -> bool:
    return any(p.search(message) for p in NEGATIVE_PATTERNS)


def is_browsing_opener(message: str) -> bool:
    return any(p.search(message) for p in BROWSING_PATTERNS)


def detect_override(message: str) -> bool:
    return any(p.search(message) for p in OVERRIDE_PATTERNS)


def match_category(message: str, categories_by_length: list[str]) -> tuple[str | None, str]:
    """识别顾客说的类目，返回 (类目名, 剩余文本)。

    两级策略：
      1. 最长前缀精确匹配 —— 未改写话术时命中率接近 100%
      2. token 重合度兜底 —— 为私有集可能出现的话术改写留后路（T6 会加强这一层）
    """
    match = LOOKING_FOR_RE.search(message)
    remainder = (match.group(1) if match else message).strip()

    lowered = remainder.lower()
    for name in categories_by_length:              # 已按长度降序，先长后短
        if lowered.startswith(name.lower()):
            return name, remainder[len(name):].strip()

    query_tokens = set(TOKEN_RE.findall(lowered))
    best, best_score = None, 0.0
    for name in categories_by_length:
        name_tokens = set(TOKEN_RE.findall(name.lower()))
        if not name_tokens:
            continue
        score = len(query_tokens & name_tokens) / len(name_tokens)
        if score > best_score:
            best, best_score = name, score
    if best_score >= 0.6:
        return best, remainder
    return None, remainder


def _split_clause(tail: str, verify: "Verifier | None" = None) -> list[str]:
    """把 'A; B.' 切成约束原句。

    难点：模拟器用 '; ' 拼接两条约束，但约束原句自身也可能含 '; '
    （例如 'solids: 100% cotton; heathers: 75% cotton, 25% polyester' 本来就是一条）。
    无脑切会把一条拆成两条。

    判据：约束原句是从商品文本里**原样**切下来的。
    所以「合起来那一串能在目录里原文找到」→ 分号是句内的，应合并；
    找不到 → 是模拟器拼接的，应切开。用只读目录当裁判，不猜规则。
    """
    tail = tail.strip().rstrip(".").strip()
    raw = [p.strip(" -;,.\t\n") for p in tail.split("; ")]
    raw = [p for p in raw if len(p) > 2]
    if verify is None or len(raw) <= 1:
        return raw

    merged: list[str] = []
    current = raw[0]
    for part in raw[1:]:
        joined = f"{current}; {part}"
        if verify(joined):          # 目录里能原文找到 → 本来就是一条
            current = joined
        else:                       # 找不到 → 模拟器拼的，切开
            merged.append(current)
            current = part
    merged.append(current)
    return merged


# 回捞时先剥掉的寒暄/句式外壳。这些词不可能出现在商品文案里，留着只会干扰。
CHATTER_RE = re.compile(
    r"(hi|hello|hey|honestly|actually|really|please|thanks|thank you|"
    r"i(?:'m| am)?|looking for|want to find|show me|searching for|"
    r"what i care about here would be|it really has to have|"
    r"a key requirement|what matters|my|the|some|a|an|is|are|be|to|for|of|and|but)",
    re.I,
)
MIN_SPAN_WORDS = 3
MAX_SPAN_WORDS = 24


def recover_spans(message: str, verify: Verifier, category_text: str | None = None) -> list[str]:
    """不靠句式，直接把顾客话里属于商品原文的片段捞出来。

    立论：约束原句是从目标商品的 features/details 里**原样**切下来的。
    所以不需要看懂句子结构——只要在候选池的原文里能找到的最长片段，就是约束本身。
    这让抽取对话术改写免疫：主办方怎么改写引导语都不影响，
    因为被改写的是外壳，而我们找的是内核。

    从长到短贪心扫，命中即认领并跳过已用词，避免同一段被重复计入。
    """
    text = message
    if category_text:
        # 类目名会同时出现在商品的 categories 字段里，不剥掉会捞出一堆噪声
        idx = text.lower().find(category_text.lower())
        if idx != -1:
            text = text[:idx] + " " + text[idx + len(category_text):]

    words = text.split()
    n = len(words)
    used = [False] * n
    found: list[str] = []

    for length in range(min(MAX_SPAN_WORDS, n), MIN_SPAN_WORDS - 1, -1):
        for start in range(0, n - length + 1):
            if any(used[start:start + length]):
                continue
            span = " ".join(words[start:start + length]).strip(" .,;:!?-")
            if len(span) < 8 or not tokenize(span):
                continue
            if CHATTER_RE.fullmatch(span.strip()):
                continue
            if verify(span):
                found.append(span)
                for i in range(start, start + length):
                    used[i] = True
    return found


def extract_constraints(
    message: str, category: str | None = None, verify: Verifier | None = None
) -> list[str]:
    """抽出顾客这一轮吐露的约束原句。

    这些原句是从目标商品的 features/details 里原样切下来的，
    因此拿去做子串匹配相当于拿到了商品指纹——这是本题最强的信号。
    """
    if is_negative(message):
        return []

    lowered = message.lower()
    for marker in CONSTRAINT_MARKERS:
        index = lowered.find(marker)
        if index != -1:
            hit = _split_clause(message[index + len(marker):], verify)
            if hit:
                return hit
            break

    # 无引导语的情况：intent_override 的开场白是 "I'm looking for CAT. OLD_VALUE"
    # 剥掉类目片段后，剩下的整句就是约束。
    if is_browsing_opener(message):
        return []

    match = LOOKING_FOR_RE.search(message)
    if not match:
        # 引导语被改写掉了 —— 不认输，直接去目录里把原文片段捞回来
        return recover_spans(message, verify, category) if verify else []

    remainder = match.group(1).strip()
    if category and remainder.lower().startswith(category.lower()):
        remainder = remainder[len(category):]
    remainder = remainder.strip(" .,;-")
    if len(remainder) > 2 and tokenize(remainder):
        return _split_clause(remainder, verify)
    return []
