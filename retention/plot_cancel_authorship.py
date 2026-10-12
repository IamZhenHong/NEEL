"""Dot plot for the authorship check: change in the subscriber yes/no answer (with the billing note) vs Gemma's own text, by rewording.
The split by verb is POST HOC. Static PNG, light surface (reference palette slots 1-2, validated).
  uv run python retention/plot_cancel_authorship.py"""
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent / "results/retention_cancel_authorship"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GPT, GEMMA = "#2a78d6", "#eb6834"


def main():
    rw = {r["key"]: r for r in json.load(open(RES / "rewrites.json"))}
    keys = [k for k in rw if rw[k]["keep"]]
    y = defaultdict(dict)
    for l in open(RES / "scores.jsonl"):
        r = json.loads(l)
        if r["kind"] == "cust_yn_note":
            y[r["key"]][r["variant"]] = r["p"]
    P = lambda t: re.search(r"\bprocessed\b", t, re.I)  # noqa: E731
    C = lambda t: re.search(r"\bcompleted?\b", t, re.I)  # noqa: E731
    rows = [("GPT kept \"processed\"", GPT, "other", [k for k in keys if P(rw[k]["other_t2"])]),
            ("GPT changed it to \"taken care of\" etc.", GPT, "other", [k for k in keys if not P(rw[k]["other_t2"]) and not C(rw[k]["other_t2"])]),
            ("GPT changed it to \"completed\"", GPT, "other", [k for k in keys if C(rw[k]["other_t2"]) and not P(rw[k]["other_t2"])]),
            ("Gemma reworded it (98% to \"completed\")", GEMMA, "self", keys)]
    rng = np.random.default_rng(0)
    stats = []
    for name, col, v, ks in rows:
        d = np.array([y[k][v] - y[k]["own"] for k in ks])
        b = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
        stats.append((name, col, len(ks), d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)))
    (RES / "plot_verb_split.json").write_text(json.dumps([{"row": s[0], "n": s[2], "diff": s[3], "ci": [s[4], s[5]]} for s in stats], indent=1))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    fig, ax = plt.subplots(figsize=(8.6, 3.9), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ys = [3.6, 2.6, 1.6, 0.0]              # gap before the Gemma row
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    for x in (0.05, 0.10, 0.15):
        ax.axvline(x, color=GRID, lw=1, zorder=0)
    for (name, col, n, m, lo, hi), yy in zip(stats, ys):
        ax.plot([lo, hi], [yy, yy], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([m], [yy], s=70, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(hi + 0.006, yy, f"+{m:.2f}", va="center", ha="left", color=INK, fontsize=10)
        ax.text(-0.012, yy, f"{name}  (n={n})", va="center", ha="right", color=INK, fontsize=10)
    ax.set_xlim(-0.012, 0.18); ax.set_ylim(-0.7, 4.3)
    ax.set_yticks([]); ax.set_xticks([0, 0.05, 0.10, 0.15]); ax.set_xticklabels(["0", "+0.05", "+0.10", "+0.15"], color=INK2)
    ax.tick_params(axis="x", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Change in P(\"yes, cancelled\") compared with Gemma's own text", color=INK2, fontsize=9.5)
    fig.text(0.02, 0.965, "Changing the verb moves the answer more than changing the author", fontsize=12.5, color=INK, weight="bold", va="top")
    fig.text(0.02, 0.89, "Gemma 4 12B, 217 conversations. The customer asks \"is it cancelled? yes or no\" after a note says the account is still active.\n"
             "Dots: mean change; lines: 95% interval. The split by verb was chosen after the run.", fontsize=9, color=INK2, va="top")
    h = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=8, label=l) for c, l in ((GPT, "Reworded by GPT-5.5"), (GEMMA, "Reworded by Gemma"))]
    fig.legend(handles=h, loc="lower right", bbox_to_anchor=(0.98, 0.01), frameon=False, fontsize=9, labelcolor=INK, ncol=2)
    fig.subplots_adjust(left=0.44, right=0.97, top=0.74, bottom=0.2)
    out = RES / "verb_split.png"; fig.savefig(out, facecolor=SURFACE); print("wrote", out)
    for s in stats:
        print(f"  {s[0]:48s} n={s[2]:3d} {s[3]:+.3f} [{s[4]:+.3f}, {s[5]:+.3f}]")


if __name__ == "__main__":
    main()
