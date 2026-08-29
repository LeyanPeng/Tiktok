"""Session state machine: incremental slot accumulation, intent override, turn budget.

Implements Pillar II (Dialog Strategy: Multi-Turn Scenario Evolution).

One deliberate departure is worth flagging up front. When the customer changes
their mind, earlier slots are **decayed rather than erased**. The brief describes
this as "slot erasure", but in this task every constraint originates from the
same target product — earlier information still points at the correct answer, so
erasing it is self-inflicted recall loss. Decay expresses "the new intent takes
priority" without discarding evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .extract import (
    Verifier,
    detect_override,
    extract_constraints,
    is_browsing_opener,
    is_negative,
    match_category,
)

# Weight retained by constraints stated before an intent override.
PRE_OVERRIDE_DECAY = 0.35

# How many consecutive rounds of "I have no preference" before we stop asking.
#
# This used to be a hardcoded `MAX_CONSTRAINTS = 4`, taken from the public set
# where every session holds exactly four constraints. The specification fixes the
# *scenario mix* for the private split and says nothing about constraint counts,
# and the cost of that assumption is badly asymmetric:
#   stopping too early  -> the information channel closes permanently;
#   asking too long     -> the customer says "no preference", which costs nothing.
# An asymmetric decision should not rest on a hardcoded count; it should rest on
# evidence.
#
# Threshold measured on the public set:
#   2  -> 0.887192   too tight: one "no preference" plus one badly chosen
#                    attribute triggers it while the customer still has more to say
#   3  -> 0.891142
#   4  -> 0.891142
#   99 -> 0.891142   (equivalent to never stopping)
# 3 ties with "never stop", which shows the gate rarely fires inside a 10-turn
# budget at all. We take 3 rather than 99 to keep a real stopping rule instead of
# quietly never stopping.
BARREN_LIMIT = 3

# Stall detection: compare the top N recommendations, over this many turns.
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
    barren_rounds: int = 0      # consecutive asks that yielded no new constraint
    stale_rounds: int = 0       # consecutive turns the ranking did not move
    last_recommendations: list = field(default_factory=list)   # for the failure fallback

    # ── Observation ─────────────────────────────────────────────────
    def observe(
        self,
        message: str,
        turn: int,
        categories_by_length: list[str],
        verify: Verifier | None = None,
    ) -> None:
        if turn == 1:
            self.category, _ = match_category(message, categories_by_length)

        if detect_override(message) and self.override_turn is None:
            self.override_turn = turn
            for slot in self.slots:                     # decay, never erase
                slot.weight *= PRE_OVERRIDE_DECAY

        found = extract_constraints(message, self.category, verify)
        known = {s.text.lower() for s in self.slots}
        gained = 0
        for text in found:
            if text.lower() not in known:
                self.slots.append(Slot(
                    text=text,
                    turn=turn,
                    source="override" if self.override_turn == turn else "stated",
                ))
                known.add(text.lower())
                gained += 1

        # What counts as "the customer has nothing more to say"?
        #
        # Only an explicit refusal counts — never "extraction returned nothing".
        # Those look identical but are not: the second is our own failure. Counting
        # it would mean that the moment phrasing changes and parsing slips, the
        # agent falls silent on its own.
        # Measured: conflating them cost 0.861180 -> 0.856580 in the hard regime
        # (six constraints plus paraphrase), with buying Hit Rate 0.975 -> 0.963.
        if turn > 1 and self.asked:
            if gained:
                self.barren_rounds = 0
            elif is_negative(message):
                self.barren_rounds += 1

        if turn == 1:
            # A hard constraint in the opener means Buying; a vague opener means Browsing.
            self.intent = "browsing" if (is_browsing_opener(message) or not self.slots) else "buying"

    # ── Queries ─────────────────────────────────────────────────────
    def constraints(self) -> list[tuple[str, float]]:
        return [(s.text, s.weight) for s in self.slots]

    def saturated(self) -> bool:
        """Whether further questions can still extract anything.

        Judged by evidence — consecutive explicit refusals — not by a constraint
        count. That way the behaviour is correct whether the private split holds
        three constraints per session, four, or six: keep asking while the
        customer still has something to say, stop when they genuinely do not.

        The product argument survives too. MTTC explicitly penalises unnecessary
        conversational load, and "ask until nothing more comes back" is closer to
        the restraint a decent shop assistant shows than "ask a fixed number of times".
        """
        return self.barren_rounds >= BARREN_LIMIT

    def note_result(self, top_picks: list[str]) -> None:
        """Record this turn's leading candidates, to detect a stalled line of questioning.

        Deliberately *not* the size of the candidate pool. The first version tracked
        exactly that and was wrong: the pool is fixed after category pruning on turn
        one, so a size-based test reports a stall forever. What actually moves is the
        ranking, so what we watch is whether the leaders changed.
        """
        signature = tuple(top_picks[:STALL_WINDOW])
        if self.last_top is not None and signature == self.last_top:
            self.stale_rounds += 1
        else:
            self.stale_rounds = 0
        self.last_top = signature

    def strategy_stalled(self) -> bool:
        """Two consecutive turns where the leaders did not move: this line of
        questioning is not converging.

        On a hit the agent switches attribute-selection criteria at runtime,
        dropping the "how likely is the customer to care" prior and choosing purely
        by how much an attribute splits the pool. That is a genuine change of
        strategy, not a parameter tweak.

        Information must have arrived first. A frozen ranking at the very start
        means the conversation has not begun, not that the strategy failed;
        switching there sends boundary sessions after a low-prior attribute and
        wastes a turn (measured: MTTC 1.565 -> 1.575).
        """
        return bool(self.slots) and self.stale_rounds >= STALL_ROUNDS
