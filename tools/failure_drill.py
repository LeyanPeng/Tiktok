"""Fault drill: is the failure fallback real, or is it decoration?

`Agent.respond` wraps the core path in try/except. The dangerous property of such
a wrapper is that **if it is broken, nobody finds out**: were the degraded path to
return an empty slate, the score would quietly sag and the logs would show nothing
at all, while we went on believing the fallback was doing its job.

So the fault is injected deliberately, to establish two things:
  1. the process survives and still returns ten valid recommendations, not an
     empty list;
  2. the wrapper itself has no side effect on the healthy path.

This does not measure how good the agent is. It measures whether the safety net
is actually a net.

On choosing the right comparison — a mistake worth recording:

The acceptance bar started out as "score drops by less than 0.02 under injection".
That was wrong. Failing one turn in three and still demanding almost no score loss
amounts to requiring the degraded path to be as good as the healthy one, which is
not achievable. It measured nothing we cared about.

Reframed as "injured with fallback vs injured without fallback", the measured
answer was **-0.001**: the fallback rescues essentially nothing. The official
evaluator already has its own `except Exception: response = {empty}`, so a single
raised exception costs only that turn's recommendations and the session recovers on
the next one — and since 71.5% of hits land on turn 1, losing one turn is nearly
free. The premise that "an exception is a catastrophe" was simply false.

So "rescues more than 0.10" is no longer a gate; the rescue figure is reported as
a measured finding. The fallback stays, for contract compliance (never an empty
slate) and because it remains correct engineering in a real deployment — not
because it earns points.

Acceptance keeps only the three properties that can actually be tested:
  - no exception escapes
  - the degraded path never returns an empty slate
  - the wrapper has no effect on the healthy score
"""

from __future__ import annotations

import sys

import evaluator.local_evaluator as official

from agent import Agent

FAIL_EVERY = 3          # inject on every third call, covering turn 1 and mid-session


class ChaosAgent(Agent):
    """Raises periodically inside the core path.

    With `with_fallback=False` the exception escapes to the evaluator, simulating
    a world in which no fallback was written.
    """

    def __init__(self, catalog_path: str = "data/catalog.jsonl", with_fallback: bool = True) -> None:
        super().__init__(catalog_path)
        self.with_fallback = with_fallback
        self.calls = 0
        self.injected = 0
        self.empty_returns = 0
        self.escaped = 0

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self.calls += 1
        if self.calls % FAIL_EVERY == 0:
            self.injected += 1
            original = self._respond

            def boom(*_args, **_kwargs):
                raise RuntimeError("injected fault")

            self._respond = boom              # type: ignore[method-assign]
            try:
                if not self.with_fallback:
                    raise RuntimeError("injected fault")   # let it reach the evaluator
                result = super().respond(session_id, user_message, turn, top_k)
            except Exception:                 # the fallback failed to catch — a real failure
                self.escaped += 1
                if self.with_fallback:
                    raise
                self._respond = original      # type: ignore[method-assign]
                return {"message": "", "ask_attribute": None, "recommendations": []}
            finally:
                self._respond = original      # type: ignore[method-assign]
        else:
            result = super().respond(session_id, user_message, turn, top_k)

        if not result.get("recommendations"):
            self.empty_returns += 1
        return result


class UnguardedAgent(Agent):
    """Bypasses the try/except wrapper and calls the core path directly.

    Used to verify the wrapper has no side effect. This check originally compared
    the healthy score against a hardcoded 0.891142, which was wrong twice over: it
    would report a false failure every time the agent legitimately improved, and it
    tested "the score equals a number" rather than "the wrapper changes nothing".
    Comparing guarded against unguarded within a single run tests the actual property.
    """

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return self._respond(session_id, user_message, turn, top_k)


def score(agent, catalog: str, dataset: str) -> float:
    ids, cats, products = official.catalog_index(catalog)
    samples = official.load_jsonl(dataset)
    return official.evaluate(agent, samples, ids, cats, products)["recommended_technical_score"]


def main() -> int:
    catalog, dataset = "data/catalog.jsonl", "data/public_set.jsonl"

    healthy = score(Agent(catalog), catalog, dataset)
    unguarded = score(UnguardedAgent(catalog), catalog, dataset)
    guarded = ChaosAgent(catalog, with_fallback=True)
    with_fb = score(guarded, catalog, dataset)
    naked = ChaosAgent(catalog, with_fallback=False)
    without_fb = score(naked, catalog, dataset)
    rescued = with_fb - without_fb

    checks = [
        (f"survives {guarded.injected} injected faults", guarded.escaped == 0,
         f"escaped exceptions={guarded.escaped}"),
        ("degraded path never returns an empty slate", guarded.empty_returns == 0,
         f"empty returns={guarded.empty_returns}"),
        ("wrapper has no side effect (self-comparison within one run)",
         abs(healthy - unguarded) < 1e-9,
         f"guarded={healthy} unguarded={unguarded}"),
    ]

    print("Fault drill")
    print(f"  healthy, wrapped          {healthy}")
    print(f"  healthy, wrapper bypassed {unguarded}   [self-comparison, replaces a hardcoded baseline]")
    print(f"  fault every {FAIL_EVERY} turns, with fallback     {with_fb}")
    print(f"  fault every {FAIL_EVERY} turns, without fallback  {without_fb}")
    print(f"  -> rescued by the fallback  {rescued:+.6f}   [measured finding, not a gate]")
    print("     The evaluator already absorbs a single-turn exception, so an")
    print("     agent-side fallback contributes almost nothing to the score.")
    print("     It is kept for contract compliance and real deployment, not points.")
    print()
    ok = True
    for name, passed, detail in checks:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}   {detail}")
    if not ok:
        print("\nThe fallback is decoration: it would lose score silently, with nothing in the logs.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
