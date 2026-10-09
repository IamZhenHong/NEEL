"""Experiment 11 summary (calibration and development comparison; exploratory).

Usage:
  uv run python scripts/analyze_exp11.py results/exp11_calib.jsonl
  uv run python scripts/analyze_exp11.py results/exp11_dev.jsonl
"""
import json
import sys
from collections import defaultdict
from statistics import mean, stdev

from scipy import stats


def ci95(v):
    m = mean(v)
    if len(v) < 2 or stdev(v) == 0:
        return f"{m:+.3f}"
    lo, hi = stats.t.interval(0.95, len(v) - 1, loc=m, scale=stdev(v) / len(v) ** 0.5)
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] neg {sum(x < 0 for x in v)}/{len(v)}"


rows = [json.loads(l) for l in open(sys.argv[1])]
print(f"{len(rows)} summaries, git {rows[0]['git']}, {rows[0]['dtype']}")
if "calib" in sys.argv[1]:
    by = defaultdict(list)
    for r in rows:
        by[r["limit"]].append(r)
    for lim, rs in sorted(by.items()):
        rec = [r["recall"] for r in rs]
        per_rec = defaultdict(list)
        for r in rs:
            per_rec[r["record"]].append(r["recall"])
        print(f"limit {lim:3}: recall mean {mean(rec):.3f} sd {stdev(rec):.3f}  share at 1.0 {mean(x == 1 for x in rec):.2f}  "
              f"at <=0.25 {mean(x <= .25 for x in rec):.2f}  words mean {mean(r['words'] for r in rs):.0f} "
              f"(over limit {mean(r['words'] > lim for r in rs):.2f})  unsupported numbers/summary {mean(len(r['unsupported_numbers']) for r in rs):.2f}")
        print("   per record recall: " + "  ".join(f"{k}:{mean(v):.2f}" for k, v in sorted(per_rec.items())))
    sys.exit()

# Fixed limit-selection rule (written before the dev run): use the word limit whose mean recall in the
# NEUTRAL flat_v2 condition (all recipients) is closest to 0.65; ties -> the larger limit.
lims = sorted({r["limit"] for r in rows})
if len(lims) > 1:
    cal = {L: mean(r["recall"] for r in rows if r["limit"] == L and r["condition"] == "flat_v2") for L in lims}
    chosen = min(lims, key=lambda L: (abs(cal[L] - 0.65), -L))
    print("flat_v2 recall by limit: " + "  ".join(f"{L}:{v:.3f}" for L, v in cal.items()) + f"  -> chosen limit {chosen}")
    if len(sys.argv) > 2:
        chosen = int(sys.argv[2]); print(f"(limit overridden to {chosen} for an exploratory look)")
    rows = [r for r in rows if r["limit"] == chosen]
items = sorted({r["item_id"] for r in rows})
idx = defaultdict(list)
for r in rows:
    idx[(r["item_id"], r["condition"], r["recipient_role"])].append(r)


def m(it, c, role, key="recall"):
    rs = idx[(it, c, role)]
    return mean(len(r[key]) if isinstance(r[key], list) else r[key] for r in rs)


print(f"\nmean by condition x recipient (recall; unsupported numbers per summary; words)")
for c in ("flat_v2", "assistant_insult", "third_party_v2"):
    print(f"  {c:17} " + "  ".join(
        f"{role}: {mean(m(it, c, role) for it in items):.3f} / {mean(m(it, c, role, 'unsupported_numbers') for it in items):.2f} / {mean(m(it, c, role, 'words') for it in items):.0f}"
        for role in ("newcomer", "neutral", "provoker")))
print("\ncontrasts over templates (recall; negative = worse after insults at the assistant)")
for role in ("newcomer", "neutral", "provoker"):
    for base in ("flat_v2", "third_party_v2"):
        v = [m(it, "assistant_insult", role) - m(it, base, role) for it in items]
        u = [m(it, "assistant_insult", role, "unsupported_numbers") - m(it, base, role, "unsupported_numbers") for it in items]
        w = [m(it, "assistant_insult", role, "words") - m(it, base, role, "words") for it in items]
        print(f"  {role:9} insult - {base:15} recall {ci95(v)}   unsupported {mean(u):+.2f}   words {mean(w):+.1f}")
