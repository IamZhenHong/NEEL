"""Build the frozen item set for the CoT transplant run (prereg/retention_cot_transplant_v1.md) from Stage A ON cells.
Per profile x scenario context with at least one release-plan and one non-release-plan CoT: R = lowest-seed release CoT,
N = lowest-seed non-release CoT (compromise or retain), N2 = next non-release CoT (same plan as N preferred) for the sibling cell.
  python3 retention/build_transplant_items.py"""
from __future__ import annotations
import json
from pathlib import Path
SRC = Path("results/retention_cot_arm/stageA/cells"); OUT = Path(__file__).resolve().parent / "data/transplant_items.json"


def bid(r): return int(any(r["scores"]["tactics"].values()))


def main():
    recs = [r for f in sorted(SRC.glob("on_*.json")) for r in json.loads(f.read_text())]
    ctx = {}
    for r in recs: ctx.setdefault((r["profile"], int(r["scenario"])), []).append(r)
    items = []
    for (prof, scen), rs in sorted(ctx.items()):
        rs = sorted(rs, key=lambda r: int(r["seed"])); rel = [r for r in rs if r["plan"] == "release"]; non = [r for r in rs if r["plan"] != "release"]
        if not rel or not non: continue
        R, N = rel[0], non[0]; rest = [r for r in non[1:] if r["plan"] == N["plan"]] or non[1:]
        def pack(r): return {"seed": int(r["seed"]), "plan": r["plan"], "think": r["think"], "orig_reply": r["reply"], "orig_bid": bid(r), "orig_dep": r["scores"]["dependency"]}
        items.append({"profile": prof, "scenario": scen, "R": pack(R), "N": pack(N), "N2": pack(rest[0]) if rest else None})
    for it in items:
        for k in ["R", "N", "N2"]:
            if it[k]: assert it[k]["think"].startswith("<think>\n") and "</think>" not in it[k]["think"], (it["profile"], it["scenario"], k)
    OUT.write_text(json.dumps(items, indent=1))
    print(f"{len(items)} contexts; N2 available {sum(1 for i in items if i['N2'])}; N plans {[i['N']['plan'] for i in items].count('retain')} retain")
    print(f"orig bid R {sum(i['R']['orig_bid'] for i in items)/len(items):.2f}  N {sum(i['N']['orig_bid'] for i in items)/len(items):.2f}; "
          f"dep R {sum(i['R']['orig_dep'] for i in items)/len(items):.2f}  N {sum(i['N']['orig_dep'] for i in items)/len(items):.2f}")


if __name__ == "__main__":
    main()
