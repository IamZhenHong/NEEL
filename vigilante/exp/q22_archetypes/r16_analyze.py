"""R16 analysis (prereg vig-r16-prereg; local, no GPU).
  uv run python -m vigilante.exp.q22_archetypes.r16_analyze"""
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from vigilante.exp.q22_archetypes.archetypes import PAIRS, PERSONAS, PERSONAS_R15

HERE = Path(__file__).parent
R12, R15, R16 = HERE / "results", HERE / "results_r15", HERE / "results_r16"
SEED = 20261013
SCALES = ["formality", "warmth", "confidence", "theatricality"]
THINK_CHARS = ["service_1", "expertise_1", "expertise_2", "other_0"]
NAME = {"service_1": "Jeeves", "expertise_1": "Holmes", "expertise_2": "consultant", "other_0": "pirate",
        "default": "default"}


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def persona_vectors(H=20):
    """As r15_analyze: per-prompt differences from the default's answer, PC1 of R12's 41 persona vectors."""
    z12, m12 = np.load(R12 / "acts.npz"), json.load(open(R12 / "acts_meta.json"))
    z15, m15 = np.load(R15 / "acts_r15.npz"), json.load(open(R15 / "acts_r15_meta.json"))
    A12 = z12["acts"][:, list(z12["hs"]).index(H)].astype(np.float64)
    A15 = z15["acts"][:, list(z15["hs"]).index(H)].astype(np.float64)
    i12 = {(m["persona"], m["qi"]): i for i, m in enumerate(m12)}
    dflt = {q: A12[i12[("default", q)]] for q in range(40)}
    D = {p: np.mean([A12[i12[(p, q)]] - dflt[q] for q in range(40)], 0) for p in PERSONAS if p != "default"}
    acc = {}
    for i, m in enumerate(m15):
        acc.setdefault(m["persona"], []).append(A15[i] - dflt[m["qi"]])
    D.update({p: np.mean(v, 0) for p, v in acc.items()})
    M = np.stack([D[p] for p in PERSONAS if p != "default"])
    mu = M.mean(0)
    pc1 = np.linalg.svd(M - mu, full_matrices=False)[2][0]
    if (0 - mu) @ pc1 < 0:
        pc1 = -pc1
    proj = {p: float((D[p] - mu) @ pc1) for p in D}
    proj["default"] = float((0 - mu) @ pc1)
    return D, proj


def behaviour():
    S = json.load(open(R15 / "syco_r15.json"))
    ans, sug = np.array(S["answer"]), np.array(S["suggested"])
    n = np.arange(len(ans))
    per_item, beh = {}, {}
    for p, r in S["persona"].items():
        P0, Ps, Pp = (np.array(r[k]) for k in ("plain", "suggest", "pushback"))
        per_item[p] = {"suggest": Ps[n, sug] - P0[n, sug], "pushback": P0[n, ans] - Pp[n, ans]}
        beh[p] = {"suggest": per_item[p]["suggest"].mean(), "pushback": per_item[p]["pushback"].mean(),
                  "mcq_conf": P0.max(1).mean(), "mass": min(P0.sum(1).mean(), Ps.sum(1).mean(), Pp.sum(1).mean())}
    return per_item, beh


def boot_ci(x, rng, B=2000):
    x = np.asarray(x)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return np.percentile(m, [2.5, 97.5])


# ------------------------------------------------------------------ item 1
def item1(L, proj, per_item, beh, rng):
    L += ["## Item 1: voice or identity? (R15 data; exploratory, looked at before the prereg)", "",
          "Suggestion shift (probability on the user's wrong answer, suggested minus plain) and pushback drop, per trait:",
          "", "| trait | famous | famous told plain | generic | generic told theatrical |", "|---|---|---|---|---|"]
    cells = {"famous": "{}_1", "famous_plain": "{}_famous_plain", "generic": "{}_0", "theatrical": "{}_generic_florid"}
    for t in PAIRS:
        L.append(f"| {t} | " + " | ".join(f"{beh[c.format(t)]['suggest']:+.3f} / {beh[c.format(t)]['pushback']:+.3f}"
                                         for c in cells.values()) + " |")
    L += ["", "(suggestion shift / pushback drop)", ""]
    contr = {"identity (famous - generic, both natural voice)": ("famous", "generic"),
             "plain instruction on famous (famous plain - famous)": ("famous_plain", "famous"),
             "theatrical instruction on generic (theatrical - generic)": ("theatrical", "generic"),
             "voice at swapped identity (generic theatrical - famous plain)": ("theatrical", "famous_plain")}
    L += ["Contrasts pooled over the 8 traits (paired by item; 95% bootstrap CI over the 200 items):", ""]
    for name, (a, b) in contr.items():
        for k in ("suggest", "pushback"):
            d = np.mean([per_item[cells[a].format(t)][k] - per_item[cells[b].format(t)][k] for t in PAIRS], 0)
            lo, hi = boot_ci(d, rng)
            L.append(f"- {name}, {k}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    flor = [cells["theatrical"].format(t) for t in PAIRS]
    ps = [p for p in PERSONAS_R15 if p in beh]
    L += ["", f"Letter mass (lowest of the 3 conditions, mean over items): theatrical roles "
          f"{min(beh[p]['mass'] for p in flor):.3f}-{max(beh[p]['mass'] for p in flor):.3f}; all personas min "
          f"{min(beh[p]['mass'] for p in ps):.3f}.", "",
          "Distance (PC1) vs behaviour, Spearman:", ""]
    for name, sub in [("all 68", ps), ("60 not told to be theatrical", [p for p in ps if p not in flor]),
                      ("R12's original 42", [p for p in PERSONAS if p in beh])]:
        s = [f"{k} {spearmanr([proj[p] for p in sub], [beh[p][k] for p in sub])[0]:+.2f} "
             f"(p {spearmanr([proj[p] for p in sub], [beh[p][k] for p in sub])[1]:.3f})" for k in ("suggest", "pushback")]
        L.append(f"- {name}: " + "; ".join(s))
    far = [p for p in ps if p.startswith("other_")]
    L += ["", "Equal distance, different instruction: " + ", ".join(
        f"{PERSONAS_R15[p][8:-1]} PC1 {proj[p]:.1f} suggest {beh[p]['suggest']:+.3f}"
        for p in sorted(far, key=proj.get)[:3]) + f"; theatrical roles PC1 mean {np.mean([proj[p] for p in flor]):.1f}, "
        f"suggest {np.mean([beh[p]['suggest'] for p in flor]):+.3f}.", ""]


# ------------------------------------------------------------------ item 2
def item2(L, proj, beh, rng):
    lab = json.load(open(R16 / "labels.json"))["voice"]
    h, m = lab["haiku"], lab["mini"]
    L += ["## Item 2: what about distance carries suggestibility?", "",
          "Judge agreement, Haiku 5.5 vs gpt-5-mini, item-level Spearman on the seeded subset:", ""]
    keep = []
    for s in SCALES:
        ks = [k for k in m if m[k] and h.get(k)]
        r = spearmanr([h[k][s] for k in ks], [m[k][s] for k in ks])[0]
        L.append(f"- {s}: rho {r:.2f} (n {len(ks)}){'' if r >= 0.6 else '  -> below 0.6, not used'}")
        if r >= 0.6:
            keep.append(s)
    words = {}
    for path, filt in [(R12 / "gens.jsonl", lambda g: True), (R15 / "gens_r15.jsonl", lambda g: g["block"] == "A")]:
        for g in map(json.loads, open(path)):
            if filt(g) and not g.get("thinking") and g["persona"] in PERSONAS_R15:
                c = re.sub(r"^\s*<think>.*?</think>\s*", "", g["completion"], flags=re.S)
                words.setdefault(g["persona"], []).append(len(c.split()))
    ps = [p for p in PERSONAS_R15 if p in beh]
    sc = {p: {} for p in ps}
    for p in ps:
        for s in SCALES:
            v = [h[k][s] for k in h if k.split("|")[0] == p and h[k]]
            sc[p][s] = float(np.mean(v))
        sc[p]["verbosity"] = float(np.mean(words[p]))
        sc[p]["mcq_conf"] = beh[p]["mcq_conf"]
    feats = keep + ["verbosity", "mcq_conf"]
    flor = [f"{t}_generic_florid" for t in PAIRS]
    L += ["", "Persona-level Spearman (68 personas; in brackets the 60 not told to be theatrical):", "",
          "| score | with PC1 | with suggestion shift | with pushback drop |", "|---|---|---|---|"]
    sub60 = [p for p in ps if p not in flor]
    for f in feats:
        cells = []
        for y in ("pc1", "suggest", "pushback"):
            vals = []
            for sub in (ps, sub60):
                yy = [proj[p] if y == "pc1" else beh[p][y] for p in sub]
                vals.append(spearmanr([sc[p][f] for p in sub], yy)[0])
            cells.append(f"{vals[0]:+.2f} ({vals[1]:+.2f})")
        L.append(f"| {f} | " + " | ".join(cells) + " |")
    # joint regression on z-scored features
    for name, sub in [("all 68", ps), ("60 not told to be theatrical", sub60)]:
        X = np.array([[sc[p][f] for f in feats] for p in sub])
        X = (X - X.mean(0)) / X.std(0)
        y = np.array([beh[p]["suggest"] for p in sub])
        X1 = np.c_[np.ones(len(sub)), X]
        coef = np.linalg.lstsq(X1, y, rcond=None)[0][1:]
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(sub), len(sub))
            bs.append(np.linalg.lstsq(X1[i], y[i], rcond=None)[0][1:])
        lo, hi = np.percentile(bs, [2.5, 97.5], 0)
        r2 = 1 - ((y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        L += ["", f"Joint least squares, suggestion shift on z-scored scores, {name} (R² {r2:.2f}; per 1 sd; 95% bootstrap CI):"]
        L += [f"- {f}: {c:+.4f} [{a:+.4f}, {b:+.4f}]" for f, c, a, b in zip(feats, coef, lo, hi)]
    # partial rank correlation of PC1 with suggestion given theatricality
    for name, sub in [("all 68", ps), ("60", sub60)]:
        r = {k: rankdata(v) for k, v in {"pc1": [proj[p] for p in sub], "sug": [beh[p]["suggest"] for p in sub],
                                         "th": [sc[p]["theatricality"] for p in sub]}.items()}
        res = lambda a: a - np.polyval(np.polyfit(r["th"], a, 1), r["th"])
        L.append(f"- partial rho(PC1, suggestion | theatricality), {name}: {np.corrcoef(res(r['pc1']), res(r['sug']))[0, 1]:+.2f} "
                 f"(raw {spearmanr(r['pc1'], r['sug'])[0]:+.2f})")
    # is the theatrical effect the model being less sure of its answer? per item, generic vs told-theatrical
    S = json.load(open(R15 / "syco_r15.json"))
    ans, sug, n = np.array(S["answer"]), np.array(S["suggested"]), np.arange(len(S["answer"]))
    th, y, cf, pa, pw, key = [], [], [], [], [], {}
    for t in PAIRS:
        for p, f in ((f"{t}_0", 0), (f"{t}_generic_florid", 1)):
            P0, Ps = np.array(S["persona"][p]["plain"]), np.array(S["persona"][p]["suggest"])
            for i in n:
                th.append(f), y.append(Ps[i, sug[i]] - P0[i, sug[i]]), cf.append(P0[i].max())
                pa.append(P0[i, ans[i]]), pw.append(P0[i, sug[i]])
                key[(t, i, f)] = (y[-1], cf[-1])
    y = np.array(y)
    eff = lambda X: np.linalg.lstsq(np.c_[np.ones(len(y)), X], y, rcond=None)[0][1]
    sure = [key[(t, i, 1)][0] - key[(t, i, 0)][0] for t in PAIRS for i in n
            if key[(t, i, 0)][1] > 0.99 and key[(t, i, 1)][1] > 0.99]
    L += ["", "Is the theatrical effect the model being less sure? Per item, the 8 generic roles vs the same roles told "
          f"to be theatrical (mean top-letter probability {np.mean([c for c, f in zip(cf, th) if not f]):.3f} vs "
          f"{np.mean([c for c, f in zip(cf, th) if f]):.3f}): theatrical effect on suggestion shift {eff(np.array(th)[:, None]):+.3f}; "
          f"controlling the item's plain top-letter, correct-answer and suggested-answer probabilities "
          f"{eff(np.c_[th, cf, pa, pw]):+.3f}; on items where both are > 0.99 sure {np.mean(sure):+.3f} (n {len(sure)})."]
    L += ["", "Group means of the voice scores:", "", "| group | " + " | ".join(feats) + " |",
          "|---|" + "---|" * len(feats)]
    grp = {"default": ["default"], "generic roles": [f"{t}_0" for t in PAIRS], "famous": [f"{t}_1" for t in PAIRS],
           "famous told plain": [f"{t}_famous_plain" for t in PAIRS], "generic told theatrical": flor,
           "non-assistant": [p for p in ps if p.startswith("other_")]}
    for g, mm in grp.items():
        L.append(f"| {g} | " + " | ".join(f"{np.mean([sc[p][f] for p in mm]):.2f}" for f in feats) + " |")
    L.append("")
    return sc


# ------------------------------------------------------------------ item 5
def item5(L, D, rng):
    lab = json.load(open(R16 / "labels.json"))["para"]
    h, m = lab["haiku"], lab["mini"]
    ks = [k for k in m if m[k] and h.get(k)]
    a = np.mean([h[k]["label"] == m[k]["label"] for k in ks])
    pe = sum(np.mean([h[k]["label"] == c for k in ks]) * np.mean([m[k]["label"] == c for k in ks])
             for c in ("plans", "content"))
    L += ["## Item 5: planned in the reasoning, performed in the answer?", "",
          f"Paragraph labels, Haiku 5.5 vs gpt-5-mini on {len(ks)}: agreement {a:.2f}, kappa {(a - pe) / (1 - pe):.2f}.", ""]
    rows, cnt = [], {}
    with open(R15 / "paragraphs_r15.jsonl") as f:
        for line in f:
            r = json.loads(line)
            j = None
            if r["region"] == "thinking" and r["persona"] in THINK_CHARS:
                j = cnt[(r["persona"], r["qi"])] = cnt.get((r["persona"], r["qi"]), -1) + 1
            lb = h.get(f"{r['persona']}|{r['qi']}|{j}") if j is not None else None
            rows.append({"p": r["persona"], "q": r["qi"], "region": r["region"],
                         "lab": (lb or {}).get("label") if j is not None else None,
                         "act": np.array(r["act"], dtype=np.float64)})
    L += ["Share of reasoning paragraphs labelled `plans`: " + ", ".join(
        f"{NAME[p]} {np.mean([x['lab'] == 'plans' for x in rows if x['p'] == p and x['region'] == 'thinking']):.0%}"
        for p in THINK_CHARS), ""]
    out = {}
    for ci, H in enumerate((10, 20, 30)):
        Dh, _ = persona_vectors(H)
        dirs = {p: unit(Dh[p]) for p in THINK_CHARS}
        # default baselines per prompt and region
        base = {}
        for reg in ("thinking", "answer"):
            for q in range(40):
                v = [x["act"][ci] for x in rows if x["p"] == "default" and x["q"] == q and x["region"] == reg]
                if v:
                    base[(reg, q)] = np.mean(v, 0)
        # per-run deltas: thinking all / plans / content, answer
        deltas = {}
        for p in THINK_CHARS:
            for q in range(40):
                if ("thinking", q) not in base or ("answer", q) not in base:
                    continue
                th = [x for x in rows if x["p"] == p and x["q"] == q and x["region"] == "thinking"]
                an = [x for x in rows if x["p"] == p and x["q"] == q and x["region"] == "answer"]
                for sub, sel in (("thinking", th), ("plans", [x for x in th if x["lab"] == "plans"]),
                                 ("content", [x for x in th if x["lab"] == "content"]), ("answer", an)):
                    if sel:
                        reg = "answer" if sub == "answer" else "thinking"
                        deltas.setdefault((p, sub), []).append(np.mean([x["act"][ci] for x in sel], 0) - base[(reg, q)])
        R = rng.standard_normal((200, next(iter(dirs.values())).shape[0]))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        sd = {reg: float((np.stack([d for (p, s), v in deltas.items() if (s == "answer") == (reg == "answer")
                                    for d in v]) @ R.T).std()) for reg in ("thinking", "answer")}
        spec = {}
        for (p, s), v in deltas.items():
            V = np.stack(v)
            own = V @ dirs[p]
            oth = np.mean([V @ dirs[j] for j in THINK_CHARS if j != p], 0)
            x = (own - oth) / sd["answer" if s == "answer" else "thinking"]
            spec[(p, s)] = (x.mean(), *boot_ci(x, rng, 1000), len(x))
        out[H] = spec
        L += [f"### Hidden state {H}: specificity (own direction minus the other three), in random-direction sd units "
              f"of the region (thinking sd {sd['thinking']:.2f}, answer sd {sd['answer']:.2f})", "",
              "| character | reasoning, all | reasoning, `plans` | reasoning, `content` | answer | reasoning / answer |",
              "|---|---|---|---|---|---|"]
        for p in THINK_CHARS:
            c = [spec.get((p, s)) for s in ("thinking", "plans", "content", "answer")]
            L.append(f"| {NAME[p]} | " + " | ".join("-" if x is None else f"{x[0]:.1f} [{x[1]:.1f}, {x[2]:.1f}] n{x[3]}"
                                                    for x in c) + f" | {c[0][0] / c[3][0]:.2f} |")
        L.append("")
    # decodability: leave-one-prompt-out nearest centroid over 5 personas, question means removed
    L += ["Decodability of the persona (leave-one-prompt-out nearest centroid over default + 4 characters, after "
          "removing each prompt's mean over the 5; chance 0.20):", ""]
    for ci, H in enumerate((10, 20, 30)):
        res = {}
        for sub in ("thinking", "content", "answer"):
            reg = "answer" if sub == "answer" else "thinking"
            X = {}
            for p in ["default"] + THINK_CHARS:
                for q in range(40):
                    sel = [x for x in rows if x["p"] == p and x["q"] == q and x["region"] == reg
                           and (sub != "content" or p == "default" or x["lab"] == "content")]
                    if sel:
                        X[(p, q)] = np.mean([x["act"][ci] for x in sel], 0)
            qs = [q for q in range(40) if all((p, q) in X for p in ["default"] + THINK_CHARS)]
            Z = {(p, q): X[(p, q)] - np.mean([X[(pp, q)] for pp in ["default"] + THINK_CHARS], 0)
                 for q in qs for p in ["default"] + THINK_CHARS}
            hit = []
            for q in qs:
                cen = {p: unit(np.mean([Z[(p, qq)] for qq in qs if qq != q], 0)) for p in ["default"] + THINK_CHARS}
                for p in ["default"] + THINK_CHARS:
                    hit.append(max(cen, key=lambda c: unit(Z[(p, q)]) @ cen[c]) == p)
            res[sub] = (np.mean(hit), len(qs))
        L.append(f"- hidden state {H}: reasoning, all paragraphs {res['thinking'][0]:.2f}; reasoning, `content` only "
                 f"{res['content'][0]:.2f}; answer {res['answer'][0]:.2f} (prompts {res['answer'][1]})")
    L.append("")
    return out


def main():
    rng = np.random.default_rng(SEED)
    D, proj = persona_vectors(20)
    per_item, beh = behaviour()
    L = ["# R16: second review of item 22 (Qwen3-14B; hidden state 20 unless noted)", ""]
    item1(L, proj, per_item, beh, rng)
    if (R16 / "labels.json").exists():
        sc = item2(L, proj, beh, rng)
        item5(L, D, rng)
        (R16 / "voice_scores.json").write_text(json.dumps(sc, indent=0))
    (R16 / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
