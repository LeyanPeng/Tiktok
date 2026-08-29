"""Clarification policy: which attribute splits the candidate pool best?

This is the most valuable idea in the system, and it is borrowed. ProductAgent's
ablation reports HIT@10 of 15.60 when clarification questions are generated
freely, against 47.00 when their options are grounded in statistics computed over
the live candidate pool — a 3x gap, larger than any other single choice in that
paper.

The principle is the optimal-split criterion from twenty questions: asking about
an attribute on which every surviving candidate agrees yields zero information
and burns a turn. Only the attribute the pool most disagrees on actually shrinks
the search space.

    expected value = P(customer holds a constraint of this type)
                   x H(that attribute's values across the live candidate pool)

The left term is derived from the catalog, not hardcoded (see `derive_type_prior`).
The right term is recomputed every turn from the candidates still alive — that is
what "grounded in statistics" means here.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .catalog import Catalog

# Attributes that cannot be counted from product copy (brand / budget / category)
# get a small floor so their prior is never exactly zero, which would make them
# unreachable forever.
FALLBACK_PRIOR = 0.001

# Attribute vocabularies. Deliberately the same word lists the official
# `classify_constraint` uses, so that the attribute we reason about and the
# attribute the simulator reasons about are the same thing.
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

# A pool can hold thousands of items; counting all of them is not worth it.
# A sample is enough to estimate the shape of the distribution.
SAMPLE_CAP = 400

# `feature` is free text with no enumerable values, so its entropy cannot be
# computed. It also accounts for roughly half of all constraints and carries the
# longest, most distinctive phrases, so it gets a conservative fixed divergence.
FEATURE_DIVERSITY = 0.70


def _normalised_entropy(values: list[str | None]) -> float:
    """Normalised entropy of a value distribution, in [0, 1].

    All identical -> 0 (asking is wasted). Evenly spread -> 1 (asking splits best).
    """
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return entropy / math.log2(len(counts))


def derive_type_prior(catalog: Catalog) -> dict[str, float]:
    """Derive "which kinds of constraints do customers state" from the catalog.

    This used to be a hardcoded table (feature .505 / material .378 / ...) measured
    across the 200 public sessions. A sensitivity sweep showed the term carries real
    leverage, with a badly asymmetric risk profile:

        measured prior   0.891142        reversed prior   0.791952   (-0.099)
        uniform          0.882679                                    (-0.008)

    Being *wrong* about the prior costs an order of magnitude more than having no
    prior at all. A hardcoded table is only correct if the private split matches the
    public one, and the specification fixes the scenario mix while saying nothing
    about constraint types.

    So the distribution is now computed over the whole catalog by replaying the
    simulator's own span-selection logic (see `catalog.constraint_candidates`). That
    yields a population distribution rather than a 200-session sample. The private
    split draws different products but shares this catalog and this selection logic,
    so the population estimate holds for both.

    `tools/prior_audit.py` verifies on every run that the derived ordering still
    matches the measured one.
    """
    counts = catalog.span_types          # tallied at load time, span by span
    total = sum(counts.values()) or 1
    names = [*VOCAB, "feature", "brand", "budget", "category"]
    return {name: max(FALLBACK_PRIOR, counts.get(name, 0) / total) for name in names}


class AskPolicy:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        # Derived once at construction, reused for every session.
        self.type_prior = derive_type_prior(catalog)

    def diversity(self, candidates: list[str], attribute: str) -> float:
        """How much the live candidate pool disagrees on this attribute."""
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
        """Pick the unasked attribute with the highest expected information gain.

        With `ignore_prior=True` the "how likely is the customer to care" term is
        dropped and selection is driven purely by how much the attribute splits the
        pool. This is triggered by stall detection — see `SessionState.strategy_stalled`.
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
        # Every estimable attribute has been asked; fall back to a fixed order.
        for attribute in fallback_order:
            if attribute not in asked:
                return attribute
        return None
