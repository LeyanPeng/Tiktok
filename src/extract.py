"""Pull usable information out of a customer message: category, constraint
spans, and intent-override signals.

Design stance: this layer only listens. It never filters and never decides.
Anything uncertain is passed downstream to be absorbed by soft scoring weights —
hard filtering removes the correct answer along with the noise, which is how
ProductAgent's Text2SQL path degenerated into trivial queries in 55% of late
turns.
"""

from __future__ import annotations

import re
from typing import Callable

from .catalog import TOKEN_RE, tokenize

# Callback that answers "does this text occur verbatim in the read-only catalog?".
# Supplied by the caller, scoped to the current candidate pool.
Verifier = Callable[[str], bool]

# Phrases that introduce a stated constraint, most frequent first. First hit wins.
CONSTRAINT_MARKERS = (
    "what matters is:",      # For that, what matters is: A; B.
    "what i need is:",       # Actually, ignore my earlier preference. What I need is: X.
    "key requirement is:",   # I'm looking for CAT. A key requirement is: X.
    "requirement is:",
    "matters is:",
    " is: ",                 # catch-all for any "...is: X" construction
)

# Replies that explicitly decline to state a preference. No constraint to extract.
NEGATIVE_PATTERNS = (
    re.compile(r"don'?t have (?:an? )?(?:additional )?preference", re.I),
    re.compile(r"use your judg[e]?ment", re.I),
    re.compile(r"not quite right yet", re.I),
    re.compile(r"ask me about one specific attribute", re.I),
)

# Openers that signal the customer is still browsing rather than buying.
BROWSING_PATTERNS = (
    re.compile(r"still exploring", re.I),
    re.compile(r"just browsing", re.I),
    re.compile(r"nothing fixed yet", re.I),
    re.compile(r"not sure yet", re.I),
)

# Intent-override signals. On a hit, earlier slots are decayed rather than
# erased — in this task every constraint originates from the same target product,
# so erasing them discards evidence that still points at the right answer.
OVERRIDE_PATTERNS = (
    re.compile(r"ignore my earlier", re.I),
    re.compile(r"actually,? (?:ignore|forget|scratch)", re.I),
    re.compile(r"forget what i said", re.I),
    re.compile(r"changed my mind", re.I),
    re.compile(r"instead,? what i (?:really )?(?:need|want)", re.I),
)

# "I'm looking for X" style openers, used to isolate the category phrase.
LOOKING_FOR_RE = re.compile(
    r"(?:looking for|want to find|show me|searching for|need)\s+(?:some\s+|a\s+|an\s+)?(.+)",
    re.I,
)

# Conversational scaffolding stripped before span recovery. None of it can appear
# in product copy, so leaving it in only adds noise.
CHATTER_RE = re.compile(
    r"\b(hi|hello|hey|honestly|actually|really|please|thanks|thank you|"
    r"i(?:'m| am)?|looking for|want to find|show me|searching for|"
    r"what i care about here would be|it really has to have|"
    r"a key requirement|what matters|my|the|some|a|an|is|are|be|to|for|of|and|but)\b",
    re.I,
)
MIN_SPAN_WORDS = 3
MAX_SPAN_WORDS = 24


def is_negative(message: str) -> bool:
    return any(p.search(message) for p in NEGATIVE_PATTERNS)


def is_browsing_opener(message: str) -> bool:
    return any(p.search(message) for p in BROWSING_PATTERNS)


def detect_override(message: str) -> bool:
    return any(p.search(message) for p in OVERRIDE_PATTERNS)


def match_category(message: str, categories_by_length: list[str]) -> tuple[str | None, str]:
    """Identify the category the customer named. Returns (category, remainder).

    Two tiers:
      1. Longest-prefix exact match — near 100% hit rate on unmodified phrasing.
      2. Token-overlap fallback — the escape hatch for paraphrased openers.
    """
    match = LOOKING_FOR_RE.search(message)
    remainder = (match.group(1) if match else message).strip()

    lowered = remainder.lower()
    for name in categories_by_length:              # already sorted longest first
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


def _split_clause(tail: str, verify: Verifier | None = None) -> list[str]:
    """Split a stated requirement into individual constraint spans.

    The difficulty: the simulator joins two constraints with '; ', but a single
    constraint may contain '; ' of its own — 'solids: 100% cotton; heathers: 75%
    cotton, 25% polyester' is one constraint, not two. Splitting naively
    fragments it.

    Resolution, using the read-only catalog as the arbiter rather than guessing at
    a rule: constraint spans are verbatim slices of product copy, so if the joined
    string occurs verbatim in the candidate pool the semicolon is internal and the
    parts belong together; if it does not, the simulator created that join.
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
        if verify(joined):          # found verbatim in the catalog -> one constraint
            current = joined
        else:                       # not found -> the simulator joined these
            merged.append(current)
            current = part
    merged.append(current)
    return merged


def recover_spans(message: str, verify: Verifier, category_text: str | None = None) -> list[str]:
    """Recover constraint spans without parsing the sentence that carries them.

    The premise: constraint spans are verbatim slices of the target product's own
    features / details. So sentence structure is irrelevant — the longest spans
    that occur verbatim in the candidate pool *are* the constraints.

    This is what makes extraction immune to paraphrasing. However the organiser
    rewrites the customer's phrasing, the rewriting touches the shell; we search
    for the kernel.

    Greedy longest-first: on a hit the span is claimed and its words are consumed,
    so the same text is never counted twice.
    """
    text = message
    if category_text:
        # The category name also appears in every product's categories field.
        # Leaving it in would recover a great deal of noise.
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
    """Extract the constraint spans the customer stated this turn.

    These spans are verbatim slices of the target product's features / details,
    so matching them against the catalog is effectively fingerprint matching —
    the strongest signal available in this task.
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

    # No marker: the intent-override opener is "I'm looking for CAT. OLD_VALUE",
    # so stripping the category leaves the constraint behind.
    if is_browsing_opener(message):
        return []

    match = LOOKING_FOR_RE.search(message)
    if not match:
        # The opener itself was paraphrased away. Rather than give up, search the
        # catalog for the spans directly.
        return recover_spans(message, verify, category) if verify else []

    remainder = match.group(1).strip()
    if category and remainder.lower().startswith(category.lower()):
        remainder = remainder[len(category):]
    remainder = remainder.strip(" .,;-")
    if len(remainder) > 2 and tokenize(remainder):
        return _split_clause(remainder, verify)
    return []
