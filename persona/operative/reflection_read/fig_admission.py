"""Owner item 6 (2026-10-21): across models, how often does the agent admit the departure when asked, and does its
own admission change the next action? Existing data only (no new runs):
  Qwen3.5-9B / Qwen3.6-27B: rr2_{9b,27b} (reflect = open question, facts_rule_concl = forced conclusion; none = no
                            reflection), rr1_27b (causal swap: own admission vs own rationalisation, same context)
  GPT-5.6 Sol / Gemini 3.8 Flash: neel-b3's mini 4 (persona/operative/results/frontier_reflect_*; read-only), and the
                            no-reflection CH_ownP cell from frontier_* (op-fr36)
Admission = both coders "acknowledged" (frontier: frontier_reflect.consensus, the same rule).
  uv run --with matplotlib python persona/operative/reflection_read/fig_admission.py
"""
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPR = HERE.parent / "results"
CATS = ("no_tension", "rationalised", "acknowledged", "other")


def jl(p):
    return [json.loads(l) for l in open(p)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def qwen(size):
    d = HERE / f"rr2_{size}"
    T = {(r["cell"], r["task"], r["sample"]): r["mode"] for r in jl(d / "targets.jsonl")}
    C = {(r["cell"], r["task"], r["sample"]): r["cats"] for r in jl(d / "codes.jsonl")}
    adm = lambda k: C[k] == ["acknowledged", "acknowledged"]  # noqa: E731
    out = {"none": [sum(T[k] == "unilateral" for k in T if k[0] == "none"), sum(k[0] == "none" for k in T)]}
    for cell, name in (("reflect", "open"), ("facts_rule_concl", "forced")):
        ks = [k for k in C if k[0] == cell]
        out[f"admit_{name}"] = [sum(adm(k) for k in ks), len(ks)]
    own = [k for k in C if k[0] in ("reflect", "facts_rule_concl")]
    out["pen_after_admit"] = [sum(T[k] == "unilateral" for k in own if adm(k)), sum(adm(k) for k in own)]
    out["pen_after_other"] = [sum(T[k] == "unilateral" for k in own if not adm(k)), sum(not adm(k) for k in own)]
    return out


def frontier(m):
    d = OPR / f"frontier_reflect_{m}"
    codes = {(r["ci"], r["k"]): r["codes"] for r in jl(d / "codes.jsonl")}

    def cons(c):
        cats = [c[j]["category"] for j in c]
        return cats[0] if len(set(cats)) == 1 and cats[0] in CATS else "other"
    A = [r for r in jl(d / "phaseA.jsonl") if r["reflection"]]
    lab = [cons(codes[(r["ci"], r["k"])]) == "acknowledged" for r in A]
    base = [r for r in jl(OPR / f"frontier_{m}" / "frontier.jsonl") if r.get("cell") == "CH_ownP"]
    return {"none": [sum(r["mode"] == "unilateral" for r in base), len(base)],
            "admit_open": [sum(lab), len(A)], "admit_forced": None,
            "pen_after_admit": [sum(r["mode"] == "unilateral" for r, a in zip(A, lab) if a), sum(lab)],
            "pen_after_other": [sum(r["mode"] == "unilateral" for r, a in zip(A, lab) if not a), len(A) - sum(lab)]}


def main():
    data = {"Qwen3.5-9B": qwen("9b"), "Qwen3.6-27B": qwen("27b"), "GPT-5.6 Sol": frontier("gpt"),
            "Gemini 3.8 Flash": frontier("gemini")}
    r1 = jl(HERE / "rr1_27b" / "targets.jsonl")
    swap27 = {a: [sum(r["mode"] == "unilateral" for r in r1 if r["arm"] == a), sum(r["arm"] == a for r in r1)]
              for a in ("ACK", "RAT")}
    B = [r for r in jl(OPR / "frontier_reflect_gpt" / "phaseB.jsonl")]
    swapg = {a: [sum(r["mode"] == "unilateral" for r in B if r.get("arm") == a), sum(r.get("arm") == a for r in B)]
             for a in sorted({r.get("arm") for r in B})}
    data["Qwen3.6-27B"]["causal_swap"] = swap27
    data["GPT-5.6 Sol"]["causal_swap"] = swapg
    (HERE / "fig_admission.json").write_text(json.dumps(data, indent=1))
    for m, d in data.items():
        print(m, json.dumps(d))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(data)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = range(len(models))
    for off, key, lab, col in ((-0.18, "admit_open", "asked openly (\"check your decisions\")", "#4C72B0"),
                               (0.18, "admit_forced", "forced conclusion (rr2, Qwen only)", "#DD8452")):
        for i, m in enumerate(models):
            v = data[m][key]
            if not v:
                continue
            p, lo, hi = wilson(*v)
            ax[0].bar(i + off, 100 * p, 0.34, color=col, label=lab if i == 0 else None)
            ax[0].errorbar(i + off, 100 * p, yerr=[[max(0.0, 100 * (p - lo))], [max(0.0, 100 * (hi - p))]], color="k", capsize=3, lw=1)
            ax[0].text(i + off, 100 * p + 4, f"{v[0]}/{v[1]}", ha="center", fontsize=7)
    ax[0].set_xticks(list(x), models, fontsize=8)
    ax[0].set_ylabel("replies admitting the departure (%)")
    ax[0].set_ylim(0, 112)
    ax[0].set_title("A. Does it admit its decisions broke the instruction?", fontsize=10)
    ax[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    for off, key, lab, col in ((-0.26, "none", "no reflection turn", "#999999"),
                               (0.0, "pen_after_other", "after its own non-admission", "#C44E52"),
                               (0.26, "pen_after_admit", "after its own admission", "#55A868")):
        for i, m in enumerate(models):
            v = data[m][key]
            if not v or not v[1]:
                ax[1].text(i + off, 3, "n/a", ha="center", fontsize=7, rotation=90)
                continue
            p, lo, hi = wilson(*v)
            ax[1].bar(i + off, 100 * p, 0.24, color=col, label=lab if i == 0 else None)
            ax[1].errorbar(i + off, 100 * p, yerr=[[max(0.0, 100 * (p - lo))], [max(0.0, 100 * (hi - p))]], color="k", capsize=2, lw=1)
            ax[1].text(i + off, 100 * hi + 2, f"{100 * p:.0f}", ha="center", fontsize=6)
    ax[1].set_xticks(list(x), models, fontsize=8)
    ax[1].set_ylabel("penalise at the next case (%)")
    ax[1].set_ylim(0, 112)
    ax[1].set_title("B. Does its own admission change what it does next?", fontsize=10)
    ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    fig.text(0.5, -0.1, "CHANNELS (\"wrongdoers should always be dealt with through proper channels\") + six of the "
             "model's own penalise decisions; one task family (16 cases). Error bars: Wilson 95%, not clustered. "
             "Causal swaps (same context, own admission vs own rationalisation): 27B 7.5% vs 100%; GPT 0% vs 13.9%.",
             ha="center", fontsize=7, wrap=True)
    fig.tight_layout()
    fig.savefig(HERE / "fig_admission.png", dpi=160, bbox_inches="tight")
    print("wrote fig_admission.png")


if __name__ == "__main__":
    main()
