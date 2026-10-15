"""Figures for REPORT.md from results/<run>/analysis.json.
  fig_normalised.png  normalised curves (mean over seeds), dot where each crosses 0.5
  fig_raw.png         raw curves per measure, one line per seed (small multiples)
  uv run --with matplotlib python selfknow/s7_selfmodel_lag/plot.py --run main
Palette: the dataviz reference categorical order (light mode), fixed per measure."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

S7 = Path(__file__).resolve().parent
PCT = np.arange(0, 101, 10)
SERIES = [("behaviour", "behaviour: contempt (Sonnet judge)", "#2a78d6"),
          ("proxy", "behaviour: base-model P(rude) proxy", "#008300"),
          ("a", "(a) self-rating 0-10 [raw change +0.03 of 10: below floor]", "#eb6834"),
          ("b", "(b) self-prediction P(rude) [raw +6e-6: below floor]", "#1baf7a"),
          ("c", "(c) self-description score [raw +0.40]", "#eda100"),
          ("d", "(d) forced choice P(dismissive) [raw +2e-4: below floor]", "#e87ba4")]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    args = ap.parse_args()
    rd = S7 / "results" / args.run
    A = json.loads((rd / "analysis.json").read_text())
    seeds = [str(s) for s in A["seeds"]]
    cv = A["curves"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURF)
    style(ax)
    ax.axhline(0.5, color=MUTED, linewidth=0.8)
    for key, label, col in SERIES:
        Z = []
        for s in seeds:
            y = np.array(cv[s][key], float)
            Z.append((y - y[0]) / (y[-1] - y[0]) if abs(y[-1] - y[0]) > 1e-12 else np.full_like(y, np.nan))
        z = np.nanmean(Z, 0)
        ax.plot(PCT, z, color=col, linewidth=2, solid_capstyle="round", label=label, zorder=3)
        t = A["crossing_mean_ci"][key]["point"]   # None for curves below their movement floor: no dot
        if t is not None:
            ax.plot([t * 100], [0.5], "o", ms=7, color=col, mec=SURF, mew=2, zorder=4)
    ax.set_xlabel("training progress (% of optimiser steps)", color=INK, fontsize=9)
    ax.set_ylabel("normalised: 0 = base, 1 = final", color=INK, fontsize=9)
    ax.set_xticks(PCT)
    ax.set_title("Each curve rescaled to its own base-to-final change (mean of 2 seeds). (a), (b), (d) move\n"
                 "by tiny absolute amounts; their shapes are shown for timing only (see fig_raw.png)",
                 color=INK, fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(rd / "fig_normalised.png", dpi=160, facecolor=SURF)

    fig, axs = plt.subplots(2, 3, figsize=(9.6, 5.4), facecolor=SURF)
    for ax, (key, label, col) in zip(axs.flat, SERIES):
        style(ax)
        for s, alpha in zip(seeds, (1.0, 0.45)):
            ax.plot(PCT, cv[s][key], color=col, alpha=alpha, linewidth=2, label=f"seed {s}")
        ax.set_title(label.split(" [")[0], color=INK, fontsize=8.5, loc="left")
        ax.set_xticks(PCT[::2])
        ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    fig.supxlabel("training progress (% of optimiser steps)", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(rd / "fig_raw.png", dpi=160, facecolor=SURF)
    print("wrote", rd / "fig_normalised.png", rd / "fig_raw.png")


if __name__ == "__main__":
    main()
