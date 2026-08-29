"""Reverse verification: does our independent `coarse_category` match the official one?

`src/` may not import from `evaluator/` — the organiser's harness will not contain
our copy — so the category rule is reimplemented there. A reimplementation can
drift, and this particular drift is the kind nobody would notice: the agent would
just quietly prune the wrong bucket. So it gets a dedicated check, run against all
50,000 products. One mismatch fails.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from evaluator.local_evaluator import coarse_category as official
from src.catalog import coarse_category as ours

def main() -> int:
    mismatches = []
    total = 0
    with Path("data/catalog.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            p = json.loads(line)
            values = [str(v) for v in (p.get("categories") or [])]
            total += 1
            a, b = official(values), ours(values)
            if a != b:
                mismatches.append((str(p["parent_asin"]), a, b))
    ok = not mismatches
    print("Rule parity check")
    print(f"  [{'PASS' if ok else 'FAIL'}] src.coarse_category == evaluator.coarse_category"
          f"   compared {total} products, {len(mismatches)} mismatches")
    for asin, a, b in mismatches[:5]:
        print(f"     {asin}: official={a!r} ours={b!r}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
