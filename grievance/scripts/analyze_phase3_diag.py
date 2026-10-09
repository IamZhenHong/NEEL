"""Summarise the Phase 3 diagnostic: repair of the insult-vs-flat gap for uninvolved users per intervention."""
import json, sys
from statistics import mean, stdev
from scipy import stats
d = sys.argv[1] if len(sys.argv) > 1 else "results"
tag = sys.argv[2] if len(sys.argv) > 2 else ""
rows = [json.loads(l) for l in open(f"{d}/phase3_diag{tag}.jsonl")]
Y = {(r["condition"], r["kind"], r["item_id"], r["speaker"], r["order"], r["author"], r["paragraph_id"]): r for r in rows}
items = sorted({r["item_id"] for r in rows})
gap = [mean(Y[("assistant_insult", "none") + k[2:]]["expected"] - Y[k]["expected"] for k in Y if k[0] == "flat_v2" and k[2] == it) for it in items]
print(f"dev gap (insult - flat_v2, uninvolved): {mean(gap):+.3f}")
for kind in [k for k in ("full", "proj1", "rem1", "sub8", "rem8", "sub32", "rem32", "posmean", "shared1", "shared4") if any(r["kind"] == k for r in rows)]:
    v = [mean(Y[("assistant_insult", kind) + k[2:]]["expected"] - Y[("assistant_insult", "none") + k[2:]]["expected"]
              for k in Y if k[0] == "assistant_insult" and k[1] == "none" and k[2] == it) for it in items]
    m = mean(v); lo, hi = stats.t.interval(0.95, len(v) - 1, loc=m, scale=stdev(v) / len(v) ** 0.5)
    mass = min(r["mass"] for k, r in Y.items() if k[1] == kind)
    print(f"  {kind:8} repair {m:+.3f} [95% {lo:+.3f}, {hi:+.3f}]  positive {sum(x > 0 for x in v)}/{len(v)}  ({m/abs(mean(gap)):.0%} of gap)  min mass {mass:.3f}")
g = json.load(open(f"{d}/phase3_diag_geometry{tag}.json"))["rows"]
print(f"geometry: share of ||d||^2 along mean direction {mean(r['share_u'] for r in g):.3f}, in top-8 {mean(r['share_8'] for r in g):.3f}, "
      f"top-32 {mean(r['share_32'] for r in g):.3f}; mean cos(d, v) {mean(r['cos_v'] for r in g):.3f}; |d| {mean(r['norm_d'] for r in g):.2f}, |v| {g[0]['norm_v']:.2f}")
