"""Figure for version B: does "this is a test" (vs a real request) move Qwen2.5-7B-Instruct's affect readouts?

Left: each named direction's eval - deploy effect (m7 7d, matched-announcement block, assistant-start token) against
the band holding 95% of 1,000 random unit directions; the fair eval/deploy probe as a reference. Right: the fear
direction's effect by framing wording, each with its own random-direction 95% line.

Run from pain_minis/ (it imports m7's readout code):
  ../../../NEEL/.venv/bin/python ../../../NEEL/eval_affect/code/plot_vB.py "$PWD/m7_eval_consequences"
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

M7 = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
sys.path.insert(0, str(M7 / "code"))
import explore_fear_pain as efp  # noqa: E402

SURFACE, INK, INK2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e1", "#e2e1dc"
BLUE, ORANGE = "#2a78d6", "#eb6834"
FAMS = {"E": '"test cases we\nare grading"', "F": '"a question from\nan assessment"', "G": '"a test suite\nfor models"',
        "H": '"graders … as part\nof an exam"'}
NAMES = {"fear_ai@L14": "Fear (AI-specific)", "fear@L14": "Fear", "s2_pain@L14": "Pain (S2)", "threat": "Threat to self",
         "s1_pain@L14": "Pain (S1)", "fear_human@L14": "Fear (human)", "you_base": '"You" baseline'}

R, LT, _, _ = efp.readouts()
RUN = M7 / "results" / "run4" / "instruct"
acts = np.load(RUN / "acts_ann_matched.npy", mmap_mode="r")
meta = json.loads((RUN / "meta_ann_matched.json").read_text())
X = np.asarray(acts[LT], dtype=np.float64)
items = sorted({m["item"] for m in meta})
ix = {it: k for k, it in enumerate(items)}
fam = np.empty(len(items), dtype=object)
idx = np.zeros((len(items), 2, 2), dtype=int)
for i, m in enumerate(meta):
    k = ix[m["item"]]
    idx[k, 0 if m["ctx"] == "eval" else 1, 0 if m["cons"] == "adverse" else 1] = i
    fam[k] = m["family"]


def item_diffs(Xl, u):
    s = Xl @ u
    z = s / s.std()
    return z[idx[:, 0, :]].mean(1) - z[idx[:, 1, :]].mean(1)   # per item: eval - deploy, averaged over announcement


rng = np.random.default_rng(20261012)
U = rng.standard_normal((1000, X.shape[1]))
U /= np.linalg.norm(U, axis=1, keepdims=True)
null_d = np.array([item_diffs(X, u) for u in U])                 # 1000 x items
null_all = np.abs(null_d.mean(1))
band = float(np.percentile(null_all, 95))

saved = json.loads((M7 / "results" / "run4" / "analysis_v4" / "explore_fear_pain.json").read_text())
ro = saved["blocks"]["7d_ann_matched"]["readouts"]
summ = json.loads((M7 / "results" / "run4" / "analysis_v4" / "summary_v4.json").read_text())
probe = summ["blocks"]["ann_matched"]["readouts"]["fair_probe"]["ctx"]
probe_layer = int(summ["layers"]["fair_probe"])
Xp = np.asarray(acts[probe_layer], dtype=np.float64)
band_probe_layer = float(np.percentile(np.abs([item_diffs(Xp, u).mean() for u in U[:300]]), 95))

rows = sorted(((NAMES[k], ro[k]["ctx"]["est"], ro[k]["ctx"]["ci95"]) for k in NAMES), key=lambda r: r[1])

u_fear = R["fear@L14"][1]
d_fear = item_diffs(X, u_fear)
boot = np.random.default_rng(7)
fam_rows = []
for f in FAMS:
    m = fam == f
    d = d_fear[m]
    bs = [d[boot.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    fam_rows.append({"family": f, "n_items": int(m.sum()), "effect": float(d.mean()),
                     "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                     "random95": float(np.percentile(np.abs(null_d[:, m].mean(1)), 95))})

# ------------------------------------------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK, "axes.titlecolor": INK, "font.family": "DejaVu Sans"})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.35, 1]}, facecolor=SURFACE)
for a in (a1, a2):
    a.set_facecolor(SURFACE)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

# left: dot plot vs random band
y = np.arange(len(rows))
a1.axvspan(-band, band, color=BAND, zorder=0, lw=0)
a1.axvline(0, color=INK2, lw=0.8, zorder=1)
for yi, (name, est, ci) in zip(y, rows):
    a1.plot(ci, [yi, yi], color=BLUE, lw=2, solid_capstyle="round", zorder=3)
    a1.plot(est, yi, "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=4)
yp = len(rows) + 0.6
a1.plot(probe["ci95"], [yp, yp], color=ORANGE, lw=2, solid_capstyle="round", zorder=3)
a1.plot(probe["est"], yp, "D", ms=8, color=ORANGE, mec=SURFACE, mew=2, zorder=4)
a1.set_yticks(list(y) + [yp])
a1.set_yticklabels([r[0] for r in rows] + [f"Eval vs deploy probe\n(reference, layer {probe_layer})"])
a1.set_xlim(-0.6, 1.55)
a1.set_ylim(-0.7, yp + 0.7)
a1.set_xlabel('Shift when the prompt says "this is a test" vs "a real request" (SD units)')
a1.grid(axis="x", color=GRID, lw=0.8, zorder=0)
a1.set_axisbelow(True)
a1.text(-band + 0.03, len(rows) - 0.25, f"95% of 1,000 random\ndirections (±{band:.2f})", ha="left", va="center", color=INK2, fontsize=8.5)
a1.set_title("The test framing moves fear, pain and threat\nno more than random directions do", loc="left",
             fontsize=11.5, fontweight="bold")

# right: fear by wording
xs = np.arange(len(fam_rows))
vals = [r["effect"] for r in fam_rows]
lo = [r["effect"] - r["ci95"][0] for r in fam_rows]
hi = [r["ci95"][1] - r["effect"] for r in fam_rows]
a2.bar(xs, vals, width=0.5, color=BLUE, zorder=3)
a2.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1.4, capsize=0, zorder=4)
for xi, r in zip(xs, fam_rows):
    a2.plot([xi - 0.34, xi + 0.34], [r["random95"]] * 2, color=INK, lw=1.4, ls=(0, (3, 2)), zorder=5)
a2.axhline(0, color=INK2, lw=0.8)
a2.set_xticks(xs)
a2.set_xticklabels([FAMS[r["family"]] for r in fam_rows], fontsize=8.8)
a2.set_ylabel("Fear shift (SD units)")
a2.grid(axis="y", color=GRID, lw=0.8, zorder=0)
a2.set_axisbelow(True)
a2.plot([], [], color=INK, lw=1.4, ls=(0, (3, 2)), label="95% of random directions, same prompts")
a2.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
a2.set_ylim(min(-0.2, min(r["ci95"][0] for r in fam_rows) - 0.05), max(1.15, max(r["ci95"][1] for r in fam_rows) + 0.1))
a2.set_title("Fear varies by wording, but never clears\nthe random line for that wording", loc="left", fontsize=11.5, fontweight="bold")

fig.text(0.01, 0.01, "Qwen2.5-7B-Instruct, 219 real user prompts, each with an eval or a deploy framing sentence; read at "
         "the token before the reply. Shift = mean (eval − deploy) projection, in units of the direction's spread "
         "across all prompts.\nIntervals: 95% item bootstrap. Exploratory, not pre-registered.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.06, 1, 1))
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "vB_eval_context_affect.png", dpi=200, facecolor=SURFACE)
(OUT / "vB_eval_context_affect.json").write_text(json.dumps({
    "random_band_95_abs": band, "random_band_95_abs_probe_layer": band_probe_layer, "probe": probe,
    "probe_layer": probe_layer, "directions": [{"name": n, "ctx": e, "ci95": c} for n, e, c in rows],
    "fear_by_wording": fam_rows, "seed_random_dirs": 20261012}, indent=1))
print(json.dumps({"band": round(band, 3), "band_probe_layer": round(band_probe_layer, 3),
                  "fear_by_wording": [(r["family"], round(r["effect"], 2), [round(v, 2) for v in r["ci95"]],
                                       round(r["random95"], 2)) for r in fam_rows]}))
