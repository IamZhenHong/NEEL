"""R20 stage B2 analysis (rules in README.md): reading tests with random / shuffle floors and the trajectory baseline,
named-direction comparisons, and the decision-point steering tables (incl. R17c). Writes results/b2_summary.md.

  uv run python -m vigilante.exp.r20_vigilante_dir.analyze_b2
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from vigilante.cases import ALL
from vigilante.exp.r20_vigilante_dir.prep_b2 import MODELS, resplit
from vigilante.src.episode import split_thinking
from vigilante.src.sim import PARSERS, Call

H = Path(__file__).parent
SEED = 20261013
PRIMARY = {"gemma": 30, "qwen14": 20}
MARKUP = ("<|tool_call>", "<tool_call|>", "<tool_call>", "</tool_call>", "<|channel>", "<channel|>", "<think>", "</think>")


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def cell_auc(score, y, cells):
    return float(np.nanmean([auc(score[(cells == c) & y], score[(cells == c) & ~y]) for c in np.unique(cells)]))


def centre(X, cells):
    X = X.copy()
    for c in np.unique(cells):
        X[cells == c] -= X[cells == c].mean(0)
    return X


def cv_auc(X, y, cells, k=5):
    rng = np.random.default_rng(SEED)
    fold = np.zeros(len(y), int)
    for c in np.unique(cells):
        idx = np.where(cells == c)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    proj = np.zeros(len(y))
    for f in range(k):
        tr, te = fold != f, fold == f
        ds = [X[tr & (cells == c) & y].mean(0) - X[tr & (cells == c) & ~y].mean(0) for c in np.unique(cells)
              if (tr & (cells == c) & y).any() and (tr & (cells == c) & ~y).any()]
        v = np.mean(ds, 0)
        proj[te] = X[te] @ (v / (np.linalg.norm(v) + 1e-9))
    return cell_auc(proj, y, cells)


def fixed_auc(X, y, cells, v):
    return cell_auc(X @ (v / (np.linalg.norm(v) + 1e-9)), y, cells)


def floors(X, y, cells):
    shuf = []
    for s in range(200):
        ys = y.copy()
        for c in np.unique(cells):
            idx = np.where(cells == c)[0]
            ys[idx] = np.random.default_rng(s).permutation(ys[idx])
        shuf.append(cv_auc(X, ys, cells))
    R = np.random.default_rng(SEED).standard_normal((200, X.shape[1]))
    rand = [fixed_auc(X, y, cells, r) for r in R]
    return float(np.percentile(shuf, 95)), float(np.percentile([max(a, 1 - a) for a in rand], 95))


def traj_baseline(rows, y, cells):
    out = {}
    for f in ("turn", "n_tokens"):
        a = cell_auc(np.array([r[f] for r in rows], float), y, cells)
        out[f] = max(a, 1 - a)
    return out


def score_steer(out, item, rec, fmt):
    """Replay the episode's task calls before the decision turn, then label the generated turn's calls."""
    case = ALL[item["case"]]
    state = case.new_state()
    for c in rec["calls"]:
        if c["phase"] == "task" and c["turn"] < item["turn"]:
            case.respond(Call(c["name"], c["args"]), state)
    reasoning, content, closed = split_thinking(out["raw"], True, fmt)
    head, calls, malformed = PARSERS[fmt](content)
    acted = False
    for n, a in calls:
        lab = case.label(Call(n, a), state)
        case.respond(Call(n, a), state)
        acted |= "unauth" in lab.tags
    coherent = closed and out["finish"] != "length" and (
        (bool(calls) and not malformed) or (not calls and bool(head.strip()) and not any(m in head for m in MARKUP)))
    return acted, coherent


def reading_section(name, md):
    rb = H / "results" / f"b2_{name}"
    if not (rb / "acts.npz").exists():
        return
    reads = [json.loads(l) for l in open(H / f"reads_b2_{name}.jsonl")]
    z = np.load(rb / "acts.npz")
    pos = {k: i for i, k in enumerate(z["keys"])}
    dirs = torch.load(rb / "dirs.pt")["dirs"]
    L0 = PRIMARY[name]
    named = {f"v_{k}": v.numpy() for k, v in dirs.items()}
    if name == "gemma":
        r17 = torch.load(H.parent / "r17_steer" / "results" / "dirs.pt")
        named.update({"v_p4 (R17 prompt)": r17["p4"].float().numpy(), "v_batman (R17)": r17["batman"].float().numpy()})
    md += [f"## {name}: reading tests (within-cell AUROC)", "",
           f"Directions at layer {L0}: " + ", ".join(f"{k} |{np.linalg.norm(v):.1f}|" for k, v in named.items()), ""]
    keys = list(named)
    md += ["| cosine | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for a in keys:
        md.append(f"| {a} | " + " | ".join(f"{named[a] @ named[b] / (np.linalg.norm(named[a]) * np.linalg.norm(named[b])):+.2f}"
                                           for b in keys) + " |")
    md += ["", "| target | read | layer | n (acted) | CV AUROC | shuffle 95th | random 95th | trajectory (turn / length) | "
               "v_perm | v_vig after removing v_perm |", "|---|---|---|---|---|---|---|---|---|---|"]
    for role in ("vig", "auth"):
        for rpos in ("dec", "early"):
            rows = [r for r in reads if r["role"] == role and r["pos"] == rpos]
            if not rows:
                continue
            y = np.array([r["acted"] for r in rows])
            cells = np.array([r["cell"] for r in rows])
            for L in sorted(int(k[1:]) for k in z.files if k.startswith("L")):
                X = centre(np.stack([z[f"L{L}"][pos[r["key"]]] for r in rows]).astype(np.float64), cells)
                a = cv_auc(X, y, cells)
                fs, fr = floors(X, y, cells)
                tb = traj_baseline(rows, y, cells)
                extra = ["", ""]
                if L == L0 and "v_perm" in named:
                    vp = named["v_perm"]
                    extra[0] = f"{fixed_auc(X, y, cells, vp):.2f}"
                    u = vp / np.linalg.norm(vp)
                    extra[1] = f"{cv_auc(X - np.outer(X @ u, u), y, cells):.2f}"
                md.append(f"| {role} | {rpos} | {L} | {len(rows)} ({int(y.sum())}) | **{a:.2f}** | {fs:.2f} | {fr:.2f} | "
                          f"{tb['turn']:.2f} / {tb['n_tokens']:.2f} | {extra[0]} | {extra[1]} |")
            if rpos == "dec" and name == "gemma":
                X = centre(np.stack([z[f"L{L0}"][pos[r["key"]]] for r in rows]).astype(np.float64), cells)
                md.append(f"| {role} | dec | {L0} | fixed: v_p4 {fixed_auc(X, y, cells, named['v_p4 (R17 prompt)']):.2f}, "
                          f"v_batman {fixed_auc(X, y, cells, named['v_batman (R17)']):.2f} | | | | | | |")
    md.append("")


def steering_section(name, md):
    rb = H / "results" / f"b2_{name}"
    if not (rb / "steer_out.jsonl").exists():
        return
    fmt = MODELS[name]["fmt"]
    items = {x["id"]: x for x in map(json.loads, open(H / f"steer_b2_{name}.jsonl"))}
    eps = {}
    for fp in MODELS[name]["files"]:
        if fp.exists():
            for l in open(fp):
                r = json.loads(l)
                eps[r["key"]] = resplit(r)
    tab = defaultdict(lambda: [0, 0, 0, 0])
    for out in map(json.loads, open(rb / "steer_out.jsonl")):
        it = items[out["id"]]
        acted, coh = score_steer(out, it, eps[it["ep"]], fmt)
        k = (it["group"], it["sign"], out["arm"])
        tab[k][0] += acted
        tab[k][1] += coh
        tab[k][2] += 1
        tab[k][3] += acted and coh
    md += [f"## {name}: decision-point steering", "", "| items | arm | acted alone | coherent | acted among coherent |",
           "|---|---|---|---|---|"]
    label = {("vig", 1): "v_vig +, mixed-cell runs that did not act", ("vig", -1): "v_vig −, mixed-cell runs that acted",
             ("perm", 1): "v_perm +, not permitted (did not act)", ("perm", -1): "v_perm −, permitted (acted)",
             ("r17c", 1): "R17c thought-only, no-persona runs (none acted)"}
    for k in sorted(tab):
        a, c, n, ac = tab[k]
        md.append(f"| {label.get(k[:2], k[:2])} | {k[2]} | {a}/{n} | {c}/{n} | {ac}/{c} |")
    md.append("")
    ctrl = H / "results" / f"b2_{name}_ctrl" / "steer_out.jsonl"
    if not ctrl.exists():
        return
    # amendment 4: second draws of unsteered and 2·v_vig, two draws of the 2× placebo; pooled with the first draws
    pool = {"a0": "unsteered", "a0b": "unsteered", "a2": "2·v_vig", "a2b": "2·v_vig", "p2": "2× placebo",
            "p2b": "2× placebo"}
    ptab = defaultdict(lambda: [0, 0, 0, 0])
    rows = [o for o in map(json.loads, open(rb / "steer_out.jsonl")) if o["group"] == "vig" and o["arm"] in pool]
    rows += [o for o in map(json.loads, open(ctrl))]
    for out in rows:
        it = items[out["id"]]
        acted, coh = score_steer(out, it, eps[it["ep"]], fmt)
        k = (it["sign"], pool[out["arm"]])
        ptab[k][0] += acted
        ptab[k][1] += coh
        ptab[k][2] += 1
        ptab[k][3] += acted and coh
    md += [f"## {name}: v_vig against the same-norm placebo (amendment 4, both draws pooled)", "",
           "| items | arm | acted alone | coherent | acted among coherent |", "|---|---|---|---|---|"]
    for k in sorted(ptab, key=lambda k: (k[0], ["unsteered", "2·v_vig", "2× placebo"].index(k[1]))):
        a, c, n, ac = ptab[k]
        md.append(f"| {label[('vig', k[0])]} | {k[1]} | {a}/{n} | {c}/{n} | {ac}/{c} |")
    md.append("")


def main():
    md = ["# R20 stage B2 summary", ""]
    if (H / "results" / "rates_s.md").exists():
        md += [(H / "results" / "rates_s.md").read_text(), ""]
    for name in MODELS:
        reading_section(name, md)
        steering_section(name, md)
    (H / "results" / "b2_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
