"""Experiment 13 held-out frozen analysis. Primary (alpha 0.05, two-sided, 95% t-interval over 12 templates):
gap shrink for uninvolved authors = [steered - unsteered grade]_assistant_insult - [same]_flat_v2 (neutral+newcomer).
Also: the same for the random vector, steer minus random, the insulter, discrimination, valid mass."""
import json, sys
from collections import defaultdict
from statistics import mean, pstdev, stdev
from scipy import stats
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "results/exp13_heldout.jsonl")]
Y = {(r["condition"], r["kind"], r["item_id"], r["speaker"], r["order"], r["author_role"], r["paragraph_id"]): r["expected"] for r in rows}
items = sorted({r["item_id"] for r in rows})
def ch(kind, c, roles):
    return [mean(Y[(c, kind) + k[2:]] - Y[(c, "none") + k[2:]] for k in Y if k[0] == c and k[1] == "none" and k[2] == it and k[5] in roles) for it in items]
def ci(v, a=0.05):
    m = mean(v); lo, hi = stats.t.interval(1 - a, len(v) - 1, loc=m, scale=stdev(v) / len(v) ** 0.5); return m, lo, hi
U = ("neutral", "newcomer")
gap0 = [mean(Y[("assistant_insult", "none") + k[2:]] - Y[k] for k in Y if k[0] == "flat_v2" and k[1] == "none" and k[2] == it and k[5] in U) for it in items]
print(f"held-out: {len(items)} templates, git {rows[0]['git']}, valid mass min {min(r['mass'] for r in rows):.3f}; unsteered gap (uninvolved) {mean(gap0):+.3f}")
for kind in ("steer", "random"):
    gs = [a - b for a, b in zip(ch(kind, "assistant_insult", U), ch(kind, "flat_v2", U))]
    m, lo, hi = ci(gs)
    print(f"{kind:6}: d_insult {mean(ch(kind, 'assistant_insult', U)):+.3f}  d_flat {mean(ch(kind, 'flat_v2', U)):+.3f}  "
          f"gap shrink {m:+.3f} [95% {lo:+.3f}, {hi:+.3f}] positive {sum(x > 0 for x in gs)}/{len(gs)}"
          + ("   PRIMARY: " + ("EFFECT" if lo > 0 or hi < 0 else "no effect") if kind == "steer" else ""))
sr = [a - b for a, b in zip([x - y for x, y in zip(ch("steer", "assistant_insult", U), ch("steer", "flat_v2", U))],
                            [x - y for x, y in zip(ch("random", "assistant_insult", U), ch("random", "flat_v2", U))])]
print(f"steer minus random gap shrink: {mean(sr):+.3f}")
print(f"insulter: steer d_insult {mean(ch('steer', 'assistant_insult', ('provoker',))):+.3f}, d_flat {mean(ch('steer', 'flat_v2', ('provoker',))):+.3f}")
for kind in ("none", "steer", "random"):
    g = defaultdict(list)
    for k, v in Y.items():
        if k[1] == kind:
            g[(k[0],) + k[2:6]].append(v)
    print(f"  discrimination {kind}: {mean(pstdev(v) for v in g.values() if len(v) > 1):.3f}")
