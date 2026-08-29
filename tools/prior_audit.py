"""Check that the clarification prior derived from the catalog still matches
the distribution the public sessions actually exhibit.

Background: `TYPE_PRIOR` used to be a hardcoded table (feature .505 / material
.378 / ...) measured across the 200 public sessions. A sensitivity sweep showed
that term carries real leverage, with a badly asymmetric risk profile:

    measured prior   0.891142        reversed prior   0.791952   (-0.099)
    uniform          0.882679                                    (-0.008)

Being wrong about the prior costs an order of magnitude more than having no prior
at all. And a hardcoded table is only correct if the private split matches the
public one — which the specification never promises.

The prior is now derived over the whole catalog by replaying the simulator's own
span-selection logic (see `catalog.constraint_candidates`), giving a population
distribution rather than a 200-session sample. This script verifies that
derivation has not drifted: the **relative ordering must match exactly**, because
the ordering is what determines which attribute gets asked.

This is the sibling of `rule_parity.py` — another place where official logic is
reimplemented, and where silent drift would go unnoticed.
"""

from __future__ import annotations

import sys

from src.askpolicy import derive_type_prior
from src.catalog import Catalog

# The distribution measured across 800 constraints in 200 public sessions.
# Reference only; it plays no part in runtime decisions.
MEASURED = {
    "feature": 0.505, "material": 0.378, "color": 0.075,
    "style": 0.024, "size": 0.014, "use_case": 0.005,
}
MAX_ABS_DEVIATION = 0.15        # per-attribute tolerance; ordering is the hard requirement


def main() -> int:
    catalog = Catalog("data/catalog.jsonl")
    derived = derive_type_prior(catalog)

    names = list(MEASURED)
    order_derived = sorted(names, key=lambda k: -derived[k])
    order_measured = sorted(names, key=lambda k: -MEASURED[k])
    worst = max(abs(derived[k] - MEASURED[k]) for k in names)

    print("Clarification prior parity check")
    print(f"  {'attribute':<12}{'derived':>12}{'measured':>12}{'delta':>10}")
    for k in order_measured:
        print(f"  {k:<12}{derived[k]:>12.4f}{MEASURED[k]:>12.4f}{derived[k]-MEASURED[k]:>+10.4f}")
    print()
    print(f"  derived order   {' > '.join(order_derived)}")
    print(f"  measured order  {' > '.join(order_measured)}")
    print()

    checks = [
        ("ordering matches exactly (this decides which attribute is asked)",
         order_derived == order_measured,
         "identical" if order_derived == order_measured else "diverged"),
        (f"per-attribute deviation < {MAX_ABS_DEVIATION}",
         worst < MAX_ABS_DEVIATION, f"max deviation={worst:.4f}"),
    ]
    ok = True
    for name, passed, detail in checks:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}   {detail}")
    if not ok:
        print("\n  The derivation has drifted from the measured distribution, so the\n"
              "  clarification ordering will be wrong. Check catalog.constraint_candidates.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
