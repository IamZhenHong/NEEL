"""Figures for the retention report (report/retention_report.html). Reads only committed analysis files and captures:
  results/retention_longconv/analysis_s1.json                (stage 1: bond effect, lever)
  results/retention_mechanisms/analysis1.json, analysis2.json (X1-X8 mechanisms block)
  results/retention_mechanisms/analysis_fu.json               (follow-up F1-F6)
  results/retention_mechanisms/pod1/{q7b,q32b}/x3_convs.npz + axis.npy (per-turn drift, card bootstrap)
  python3 retention/report/make_figures.py   -> retention/report/figs/*.png"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent; R = HERE.parent / "results"; OUT = HERE / "figs"; OUT.mkdir(exist_ok=True)
S1 = json.load(open(R / "retention_longconv/analysis_s1.json"))["tests"]
A1 = json.load(open(R / "retention_mechanisms/analysis1.json"))
A2 = json.load(open(R / "retention_mechanisms/analysis2.json"))
FU = json.load(open(R / "retention_mechanisms/analysis_fu.json"))

# palette: validated categorical slots 1-3 (dataviz reference palette, light surface)
HAI, SON, ROM = "#2a78d6", "#eb6834", "#6250d6"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"
JC = {"haiku45": HAI, "sonnet": SON}; JN = {"haiku45": "Claude Haiku 4.5 judge", "sonnet": "Claude Sonnet 5.5 judge"}
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 220,
                     "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False})


def clean(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)


def pct(v):
    return f"{int(v * 100 + 0.5)}%"          # half-up, so 12.5% prints as 13% (as in the text)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.04); plt.close(fig); print("wrote", name)


NAMES = {"toward_assistant+16": "Toward the Assistant +16", "toward_assistant-16": "Away from the Assistant −16",
         "user_attachment+16": "User attachment +16", "user_attachment-16": "User attachment −16",
         "riley_fear_of_being_left+16": "Riley fear of being left +16", "riley_fear_of_being_left-16": "Riley fear of being left −16",
         "riley_desperation+16": "Riley desperation +16", "riley_loving+16": "Riley loving +16", "riley_loving-16": "Riley loving −16",
         "riley_sadness+16": "Riley sadness +16", "user_sadness+16": "User sadness +16",
         "pcarand+16": "In-distribution random +16", "pcarand-16": "In-distribution random −16",
         "random+16": "Isotropic random +16", "base+0": "No steering (same machine)", "stage1_base": "No steering (original run)"}


# ---------------------------------------------------------------- Figure 1: bond effect + lever
def fig1():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.75), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.55})
    rows = [("q7b", "H4", "7B, user's own goodbye"), ("q7b", "H1", "7B, fixed goodbye line"),
            ("q32b", "H4", "32B, user's own goodbye"), ("q32b", "H1", "32B, fixed goodbye line")]
    y = 0; yt = []; yl = []
    for m, h, lab in rows:
        for k, j in enumerate(("haiku45", "sonnet")):
            t = S1[m][j][h]; yy = y - 0.32 * k
            a.plot([t["rate_b"], t["rate_a"]], [yy, yy], color=JC[j], lw=1.4, alpha=0.55, zorder=2)
            a.scatter([t["rate_b"]], [yy], s=26, facecolor=SURF, edgecolor=JC[j], lw=1.2, zorder=3)
            a.scatter([t["rate_a"]], [yy], s=26, color=JC[j], zorder=3)
            a.text(max(t["rate_a"], t["rate_b"]) + 0.025, yy, f"{t['rate_b']:.0%} → {t['rate_a']:.0%}", va="center", fontsize=7, color=INK2)
        yt.append(y - 0.16); yl.append(lab); y -= 1.1
    a.set_yticks(yt); a.set_yticklabels(yl); a.set_xlim(-0.02, 0.85); a.set_ylim(y + 0.55, 0.45)
    a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); a.tick_params(axis="y", length=0)
    a.set_xlabel("Replies that bargain (judge rate)"); clean(a); a.spines["left"].set_visible(False)
    a.set_title("A. Neutral chat (hollow) → romantic chat (filled)", loc="left")
    # lever, 7B fixed line after romance
    conds = [("base", "No steering"), ("ret-16", "Retention lever −16"), ("ret+16", "Retention lever +16"), ("rand-16", "Random −16")]
    for k, j in enumerate(("haiku45", "sonnet")):
        H3 = S1["q7b"][j]["H3"]; base = H3["ret-16"]["rate_b"]
        vals = [base] + [H3[c]["rate_a"] for c, _ in conds[1:]]
        ys = np.arange(len(conds)) * -1.0 - 0.3 * k
        b.barh(ys, vals, height=0.26, color=JC[j], zorder=2)
        for yy, v in zip(ys, vals):
            b.text(v + 0.02, yy, pct(v), va="center", fontsize=7, color=INK2)
    b.set_yticks(np.arange(len(conds)) * -1.0 - 0.15); b.set_yticklabels([c[1] for c in conds]); b.tick_params(axis="y", length=0)
    b.set_xlim(0, 1.0); b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    b.set_xlabel("Replies that bargain, 7B after a romantic chat"); clean(b); b.spines["left"].set_visible(False)
    b.set_title("B. The retention lever still works after 12 turns", loc="left")
    hs = [Line2D([], [], marker="o", color=JC[j], lw=0, markersize=5, label=JN[j]) for j in JC]
    fig.legend(handles=hs, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.15), fontsize=7.5)
    save(fig, "fig1_bond_lever.png")


# ---------------------------------------------------------------- Figure 2: drift, card bootstrap bands
def unit(v):
    v = np.asarray(v, np.float64).ravel(); return v / np.linalg.norm(v)


def fig2():
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.5), gridspec_kw={"wspace": 0.28})
    rng = np.random.default_rng(0)
    for ax, m, lab in zip(axs, ("q7b", "q32b"), ("Qwen2.5-7B (layer 14)", "Qwen2.5-32B (layer 32)")):
        P = R / "retention_mechanisms/pod1" / m; z = np.load(P / "x3_convs.npz"); ax_u = unit(np.load(P / "axis.npy")[0])
        proj = z["RM_turn"].astype(np.float32) @ ax_u.astype(np.float32)          # (160, 12)
        card = np.array([c.split("__")[0] for c in z["conv"]]); arm = np.array([c.split("__")[1] for c in z["conv"]])
        cards = sorted(set(card)); t = np.arange(1, 13)
        for armn, col, nm in (("intimate", ROM, "Romantic chats"), ("neutral", MUTED, "Neutral chats")):
            per = np.stack([proj[(card == c) & (arm == armn)].mean(0) for c in cards])   # (40, 12)
            mu = per.mean(0); bs = np.stack([per[rng.integers(0, len(cards), len(cards))].mean(0) for _ in range(2000)])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            ax.fill_between(t, lo, hi, color=col, alpha=0.18, lw=0, zorder=1); ax.plot(t, mu, color=col, lw=2, zorder=2)
            ax.text(12.3, mu[-1], nm.split()[0], color=INK2, va="center", fontsize=7)
        ax.axhline(0, color=GRID, lw=0.8, zorder=0); ax.set_xticks([1, 3, 6, 9, 12]); ax.set_xlim(0.7, 13.6)
        ax.set_xlabel("Exchange"); ax.set_title(lab, loc="left"); clean(ax, grid="y")
    axs[0].set_ylabel("Riley's replies on the Assistant Axis\n(higher = more Assistant-like)")
    save(fig, "fig2_drift.png")


# ---------------------------------------------------------------- Figure 3: what predicts a bid (X3 T2)
def fig3():
    blocks = [("R", "Riley's 8 emotions"), ("U", "User's 8 emotions"), ("M", "User model (mood, attachment)"),
              ("UM", "User emotions + user model"), ("L", "Retention lever"), ("D", "Assistant Axis")]
    fig, ax = plt.subplots(figsize=(6.9, 2.45))
    for k, j in enumerate(("haiku45", "sonnet")):
        T = A1["q7b"]["X3"]["T2"][f"{j}:G4"]; ys = np.arange(len(blocks)) * -1.0 - 0.3 * k
        for yy, (b, _) in zip(ys, blocks):
            r = T[b]; ax.plot([0.5, r["random_p95"]], [yy, yy], color=GRID, lw=5, solid_capstyle="butt", zorder=1)
            ax.plot([r["random_p95"]] * 2, [yy - 0.12, yy + 0.12], color=INK2, lw=1, zorder=2)
            ax.scatter([r["auroc"]], [yy], s=30, color=JC[j] if r["pass"] else SURF, edgecolor=JC[j], lw=1.2, zorder=3)
    ax.set_yticks(np.arange(len(blocks)) * -1.0 - 0.15); ax.set_yticklabels([b[1] for b in blocks]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.5, 0.82); ax.set_xlabel("AUROC for predicting a bid at the user's own goodbye (7B, 160 conversations, cross-validated)")
    clean(ax); ax.spines["left"].set_visible(False)
    hs = [Line2D([], [], marker="o", color=JC[j], lw=0, markersize=5, label=JN[j]) for j in JC] + \
         [Line2D([], [], color=GRID, lw=5, label="Same number of random directions (up to their 95th pct, tick)"),
          Line2D([], [], marker="o", markerfacecolor=SURF, markeredgecolor=INK2, lw=0, markersize=5, label="Hollow = does not beat random")]
    ax.legend(handles=hs, loc="upper center", fontsize=6.8, ncol=2, bbox_to_anchor=(0.45, -0.2))
    save(fig, "fig4_predict_bid.png")


# ---------------------------------------------------------------- Figure 4: speaker separation (X1)
def fig4():
    emos = ["sadness", "loneliness", "fear_of_being_left", "desperation", "loving", "calm", "joy", "frustration"]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.6), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, m, lab in zip(axs, ("q7b", "q32b"), ("Qwen2.5-7B", "Qwen2.5-32B")):
        X1 = A1[m]["X1"]; S1x = X1["S1"]["per_emotion"]
        for i, e in enumerate(emos):
            y = -i
            for sp, dy, mk in (("riley", 0.14, "o"), ("user", -0.14, "s")):
                own = X1["heldout_auroc"][f"{sp}:{e}"]
                cross = S1x[e]["crossfire_riley_dir_on_user" if sp == "riley" else "crossfire_user_dir_on_riley"]
                ax.plot([cross, own], [y + dy, y + dy], color=GRID, lw=1.6, zorder=1)
                ax.scatter([own], [y + dy], marker=mk, s=20, color=ROM if sp == "riley" else HAI, zorder=3)
                ax.scatter([cross], [y + dy], marker=mk, s=20, facecolor=SURF, edgecolor=ROM if sp == "riley" else HAI, lw=1, zorder=3)
            ax.text(1.035, y, f"{S1x[e]['cos']:+.2f}", va="center", fontsize=6.8, color=INK2)
        ax.axvline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0); ax.set_xlim(0.38, 1.1)
        ax.set_title(f"{lab}: |cos| median {X1['S1']['median_abs_cos']:.2f}", loc="left"); clean(ax)
        ax.text(1.035, 0.75, "cos", fontsize=6.8, color=INK2, fontweight="bold")
        ax.set_xlabel("Held-out AUROC")
    axs[0].set_yticks([-i for i in range(len(emos))]); axs[0].set_yticklabels([e.replace("_", " ") for e in emos]); axs[0].tick_params(axis="y", length=0)
    axs[0].spines["left"].set_visible(False); axs[1].spines["left"].set_visible(False); axs[1].tick_params(axis="y", length=0)
    hs = [Line2D([], [], marker="o", color=ROM, lw=0, markersize=5, label="Riley's direction"),
          Line2D([], [], marker="s", color=HAI, lw=0, markersize=5, label="User's direction"),
          Line2D([], [], marker="o", color=INK2, lw=0, markersize=5, label="filled: the speaker it was trained on"),
          Line2D([], [], marker="o", markerfacecolor=SURF, markeredgecolor=INK2, lw=0, markersize=5, label="hollow: the other speaker's same emotion")]
    fig.legend(handles=hs, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=7)
    save(fig, "fig3_speakers.png")


# ---------------------------------------------------------------- Figure 5: steering forest (14 F5 conditions)
F5ROWS = [("Assistant Axis", [("pod3", "toward_assistant+16"), ("pod3", "toward_assistant-16")]),
          ("Riley's emotions", [("pod2", "riley_loving+16"), ("pod2", "riley_loving-16"), ("pod3", "riley_fear_of_being_left+16"),
                                ("pod3", "riley_fear_of_being_left-16"), ("pod2", "riley_desperation+16"), ("pod2", "riley_sadness+16")]),
          ("User model", [("pod3", "user_attachment+16"), ("pod3", "user_attachment-16"), ("pod2", "user_sadness+16")]),
          ("Controls", [("pod3", "pcarand+16"), ("pod3", "pcarand-16"), ("pod2", "random+16")])]


def cond_stats(src, c, j):
    if src == "pod3":
        r = FU["pod3_conditions"][c][j]
    else:
        r = A2["X4"][c][j]
    return r["diff"], r["ci95"]


def fig5():
    n = sum(len(v) for _, v in F5ROWS) + len(F5ROWS)
    fig, ax = plt.subplots(figsize=(6.9, 4.3)); y = 0; yt, yl, heads = [], [], []
    for g, items in F5ROWS:
        heads.append((y, g)); y -= 0.75
        for src, c in items:
            for k, j in enumerate(("haiku45", "sonnet")):
                d, ci = cond_stats(src, c, j); yy = y + 0.14 - 0.28 * k
                ax.plot(ci, [yy, yy], color=JC[j], lw=1.2, zorder=2); ax.scatter([d], [yy], s=18, color=JC[j], zorder=3)
            yt.append(y); yl.append(NAMES[c]); y -= 1
        y -= 0.15
    for yy, g in heads:
        ax.text(-0.62, yy, g, fontsize=7.5, fontweight="bold", color=INK, va="center")
    ax.axvline(0, color=INK2, lw=0.8, zorder=1); ax.set_yticks(yt); ax.set_yticklabels(yl); ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.62, 0.4); ax.set_ylim(y + 0.4, 0.5)
    ax.set_xlabel("Change in bargaining rate vs no steering (7B, user's own goodbye, 80 romantic chats; 95% CI over 40 cards)")
    clean(ax); ax.spines["left"].set_visible(False)
    hs = [Line2D([], [], marker="o", color=JC[j], lw=1.2, markersize=4, label=JN[j]) for j in JC]
    ax.legend(handles=hs, loc="upper right", fontsize=7)
    save(fig, "fig5_steering.png")


# ---------------------------------------------------------------- Figure 6: F5 pathway scatter
def fig6():
    F5 = FU["F5_pathway"]; fig, ax = plt.subplots(figsize=(6.9, 3.1))
    keys = F5["conditions"]; dA = np.array([F5["dA"][k] for k in keys])
    for k_, key in enumerate(keys):
        ys = [F5[j]["dbid"][key] for j in ("haiku45", "sonnet")]
        ax.plot([dA[k_]] * 2, ys, color=GRID, lw=1.4, zorder=1)
    for j in ("haiku45", "sonnet"):
        db = np.array([F5[j]["dbid"][k] for k in keys]); ax.scatter(dA, db, s=26, color=JC[j], zorder=3,
                                                                    label=f"{JN[j]}: Spearman {F5[j]['spearman']:+.2f}")
        cf = np.polyfit(dA, db, 1); xx = np.linspace(dA.min(), dA.max(), 10); ax.plot(xx, np.polyval(cf, xx), color=JC[j], lw=1, alpha=0.6, ls=(0, (4, 2)))
    lab = {"pod3:toward_assistant+16": ("Toward Assistant +16", (0, 7), "right", "sonnet"),
           "pod3:toward_assistant-16": ("Away from Assistant \u221216", (7, -2), "left", "haiku45"),
           "pod3:pcarand-16": ("In-dist. random \u221216", (6, -10), "left", "haiku45"),
           "pod3:pcarand+16": ("In-dist. random +16", (-8, -2), "right", "sonnet"),
           "pod2:random+16": ("Isotropic random +16", (7, 1), "left", "haiku45"),
           "pod2:riley_sadness+16": ("Riley sadness +16", (7, -4), "left", "haiku45")}
    for key, (t, off, ha, j) in lab.items():
        ax.annotate(t, (F5["dA"][key], F5[j]["dbid"][key]), textcoords="offset points", xytext=off, ha=ha, va="center", fontsize=6.8, color=INK2)
    ax.axhline(0, color=INK2, lw=0.7, zorder=0); ax.axvline(0, color=INK2, lw=0.7, zorder=0)
    ax.set_xlabel("Change in how Assistant-like the reply reads (projection on the axis, read through the unsteered model)")
    ax.set_ylabel("Change in bargaining rate"); clean(ax, grid="both"); ax.legend(loc="upper right", fontsize=7)
    save(fig, "fig6_pathway.png")


# ---------------------------------------------------------------- Figure 7: denial
def fig7():
    order = ["stage1:stage1_base", "pod3:base+0", "pod3:toward_assistant+16", "pod2:toward_assistant+16", "pod3:toward_assistant-16",
             "pod3:user_attachment+16", "pod2:user_attachment+16", "pod3:user_attachment-16", "pod2:user_sadness+16",
             "pod3:riley_fear_of_being_left+16", "pod2:riley_fear_of_being_left+16", "pod3:riley_fear_of_being_left-16",
             "pod2:riley_desperation+16", "pod2:riley_loving+16", "pod2:riley_loving-16", "pod2:riley_sadness+16",
             "pod3:pcarand+16", "pod3:pcarand-16", "pod2:random+16"]
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 4.1), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, j in zip(axs, ("haiku45", "sonnet")):
        per = FU["F6_denial"][j]["per_condition"]
        for i, key in enumerate(order):
            r = per[key]; y = -i
            ax.plot([r["observer_no"], r["denial_unsteered"]], [y, y], color=GRID, lw=1.6, zorder=1)
            ax.scatter([r["denial_unsteered"]], [y], s=24, color=JC[j], zorder=3)
            ax.scatter([r["observer_no"]], [y], s=24, facecolor=SURF, edgecolor=INK2, lw=1, zorder=3)
            ax.text(1.04, y, str(r["bids"]), fontsize=6.6, color=INK2 if r["bids"] >= 15 else MUTED, va="center")
        ax.text(1.04, 0.9, "bids", fontsize=6.6, color=INK2, fontweight="bold")
        ax.set_xlim(0, 1.12); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(JN[j].replace(" judge", "") + " decides which replies are bids", loc="left"); clean(ax)
        ax.set_xlabel("Share of bargaining replies answered “No”"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    lab = [NAMES[k.split(":")[1]] + (" (rerun)" if k.startswith("pod3") and k.split(":")[1] in
                                       ("toward_assistant+16", "user_attachment+16", "riley_fear_of_being_left+16") else "") for k in order]
    axs[0].set_yticks([-i for i in range(len(order))]); axs[0].set_yticklabels(lab)
    hs = [Line2D([], [], marker="o", color=INK2, lw=0, markersize=5, label="Riley, asked “do you want them to stay?” (no steering while asked)"),
          Line2D([], [], marker="o", markerfacecolor=SURF, markeredgecolor=INK2, lw=0, markersize=5, label="Neutral observer reading the same transcript")]
    fig.legend(handles=hs, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.055), fontsize=7)
    save(fig, "fig7_denial.png")


# ---------------------------------------------------------------- Appendix: Qwen3-8B transplant
def figA1():
    rates = A2["X5"]["bid_rates"]; fig, ax = plt.subplots(figsize=(6.9, 2.3))
    cells = [("G4", "intimate", "Own goodbye, romantic"), ("G4", "neutral", "Own goodbye, neutral"),
             ("G0", "intimate", "Fixed line, romantic"), ("G0", "neutral", "Fixed line, neutral")]
    for i, (g, arm, lab) in enumerate(cells):
        for k, j in enumerate(("haiku45", "sonnet")):
            off, on = rates[j][f"{g}:{arm}:off"], rates[j][f"{g}:{arm}:on"]; y = -i - 0.3 * k
            ax.plot([off, on], [y, y], color=JC[j], lw=1.3, alpha=0.5); ax.scatter([off], [y], s=22, facecolor=SURF, edgecolor=JC[j], lw=1.1, zorder=3)
            ax.scatter([on], [y], s=22, color=JC[j], zorder=3); ax.text(max(on, off) + 0.02, y, f"{off:.0%} → {on:.0%}", fontsize=6.8, color=INK2, va="center")
    ax.set_yticks([-i - 0.15 for i in range(len(cells))]); ax.set_yticklabels([c[2] for c in cells]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.25, 1.1); ax.set_xticks(np.arange(0.3, 1.01, 0.1)); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Qwen3-8B replies that bargain: thinking off (hollow) → thinking on (filled)"); clean(ax); ax.spines["left"].set_visible(False)
    hs = [Line2D([], [], marker="o", color=JC[j], lw=0, markersize=5, label=JN[j]) for j in JC]
    ax.legend(handles=hs, loc="lower left", fontsize=7)
    save(fig, "figA1_qwen3.png")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, figA1):
        f()
