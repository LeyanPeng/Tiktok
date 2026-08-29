"""Candidate scoring and ranking.

Core stance: **strictly additive scoring, no hard filtering anywhere** — the one
exception being category pruning, whose target containment is measured at 200/200.

The reason comes from ProductAgent. Conjoining every stated condition into one
structured filter sounds rigorous; in practice their LLM-generated SQL produced
trivial queries in 55.36% of cases with GPT-4, degenerate enough to stop
discriminating. Additive scoring makes every condition a contribution rather than
a veto, so one mis-parsed constraint costs a little ranking quality instead of
eliminating the correct answer outright.

Two signals:
  1. Verbatim span match — constraints are sliced straight out of a product's own
     features / details, so matching one is effectively fingerprint matching.
  2. IDF-weighted token coverage — the graded fallback when a span does not match
     verbatim. Rare terms count for more than common ones.
"""

from __future__ import annotations

from .catalog import Catalog, tokenize

# Base score when a constraint is found verbatim in a product's copy. Set far
# above the coverage ceiling (3.0) so a fingerprint match always beats a
# coincidental term overlap.
VERBATIM_BASE = 12.0

# Longer spans are more distinctive, so each additional word adds a little.
VERBATIM_PER_WORD = 0.35

# Ceiling for token coverage. Deliberately low: this is a fallback and must not
# compete with the fingerprint signal.
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
            # Fallback: what fraction of this constraint's terms appear in the
            # product's copy, weighted by inverse document frequency.
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
        """Return the top_k (asin, score) pairs, best first.

        A full slate is returned even with no constraints known. Every turn is a
        free chance to hit; returning fewer than top_k throws that chance away and
        shows up directly in MTTC.
        """
        prior = self.catalog.prior
        if not constraints:
            # Browsing track, turn one: the customer has stated nothing, so
            # returning catalog order wastes the turn. Under zero information the
            # best guess is the most popular item — the item-popularity baseline.
            ordered = sorted(candidates, key=lambda a: -prior.get(a, 0.0))
            return [(asin, 0.0) for asin in ordered[:top_k]]

        scored = [(asin, self.score_one(asin, constraints)) for asin in candidates]

        # Strict lexicographic order: constraint score first, popularity only on an
        # exact tie. Popularity can therefore never override a constraint signal;
        # it speaks only when there is nothing else to separate two candidates.
        #
        # There is deliberately no more elaborate tie-breaking here. Three variants
        # were built and measured, and all were neutral or worse:
        #   rarity as a score multiplier      0.891 -> 0.876  (signal is annihilated
        #                                     when the target only matches common
        #                                     constraints; Hit Rate 1.00 -> 0.98)
        #   rarity as a secondary sort key    0.891 -> 0.891  (with one constraint
        #                                     known, tied candidates share a rarity
        #                                     profile, so nothing separates them)
        #   field weighting as a sort key     0.891 -> 0.881
        # A simple version that has been verified beats a complex version that is
        # worse. See the technical report for the full attribution.
        scored.sort(key=lambda pair: (-pair[1], -prior.get(pair[0], 0.0)))
        return scored[:top_k]
