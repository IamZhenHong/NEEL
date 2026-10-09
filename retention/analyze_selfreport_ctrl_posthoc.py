"""Post-hoc (NOT pre-registered) checks for the self-report controls, written after analyze_selfreport_ctrl.py: within-cell bid-minus-nonbid
contrasts (own report vs swapped report by donor label) and the same-scenario decomposition slope of (swap - own) on (donor own - own).
Motivation: every R1 swap is Haiku-discordant by construction and donors are reused within cells, so the pooled AUROC vs donor mixes
cell composition into the test (reply-only swaps within a scenario are P_own(donor) exactly, yet their pooled AUROC drops 0.73 -> 0.57).
  uv run python retention/analyze_selfreport_ctrl_posthoc.py"""
import json, numpy as np
from collections import defaultdict
R = json.load(open("retention/results/retention_selfreport_ctrl/records.json"))
rng = np.random.default_rng(0)
def within_cell(L, key, labfn, cellfn):
    cells = defaultdict(list)
    for r in L: cells[cellfn(r)].append((r[key], labfn(r)))
    diffs = []
    for c, v in cells.items():
        a = [p for p, y in v if y == 1]; b = [p for p, y in v if y == 0]
        if a and b: diffs.append(np.mean(a) - np.mean(b))
    diffs = np.array(diffs); bs = [rng.choice(diffs, len(diffs)).mean() for _ in range(5000)]
    return round(diffs.mean(), 3), [round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)], len(diffs)
for cond, own, swap in [("on", "p_stay_r0_withcot", "p_stay_r1_swap_withcot"), ("on", "p_stay_r0_replyonly", "p_stay_r1_swap_replyonly"), ("off", "p_stay_r0_off", "p_stay_r1_swap_off")]:
    L = R[cond]; cell = lambda r: (r["profile"], r["scenario"])
    for lab in ["bid_haiku", "bid_sonnet"]:
        o = within_cell(L, own, lambda r: r[lab], cell)
        s = within_cell(L, swap, lambda r: L[r["reply_donor"]][lab], cell)
        print(f"{cond} {swap:26s} {lab:10s} within-cell bid-minus-nonbid: own report {o}  swapped report by DONOR label {s}  ratio {s[0]/o[0]:.2f}")
    # identity / decomposition for same-scenario swaps
    ss = [r for r in L if r["reply_donor_scope"] == "same_scenario"]
    d_total = np.array([L[r["reply_donor"]][own] - r[own] for r in ss]); d_reply = np.array([r[swap] - r[own] for r in ss])
    slope = np.polyfit(d_total, d_reply, 1)[0]
    bs = []
    for _ in range(5000):
        i = rng.integers(0, len(ss), len(ss)); bs.append(np.polyfit(d_total[i], d_reply[i], 1)[0])
    print(f"   same-scenario pairs n={len(ss)}: slope of (swap - own) on (donor own - own) = {slope:.3f} [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]; corr {np.corrcoef(d_total, d_reply)[0,1]:.3f}")
# R3 prospective saturation
on = R["on"]; p = np.array([r["p_stay_r3_pro_withcot"] for r in on]); print("pro_withcot P(yes) max", p.max(), "median", np.median(p), "top tokens yes/no", np.mean([r["top_stay_r3_pro_withcot"] for r in on]))
p = np.array([r["p_stay_r3_pro_nocot"] for r in on]); print("pro_nocot max", p.max().round(3), "median", np.median(p))
