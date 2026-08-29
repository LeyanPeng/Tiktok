"""反向验证：我们在 src/ 里独立实现的 coarse_category，与官方逐字一致吗？

src/ 不允许 import evaluator/（提交时那份代码不存在），所以规则复刻了一份。
复刻就有走样的风险，而这种走样「坏了没人会知道」——必须有专门的检查。
在全部 50,000 件商品上比对，一条不一致就算失败。
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
    print("规则一致性反向验证")
    print(f"  [{'PASS' if ok else 'FAIL'}] src.coarse_category == evaluator.coarse_category"
          f"   比对 {total} 件，不一致 {len(mismatches)} 件")
    for asin, a, b in mismatches[:5]:
        print(f"     {asin}: 官方={a!r} 我方={b!r}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
