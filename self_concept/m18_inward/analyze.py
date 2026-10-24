"""SC-M18 analysis (DESIGN.md "Rules"). Test split only (200 prompts per family).

  uv run python -m self_concept.m18_inward.analyze [--res DIR]      -> DIR/analysis.json, DIR/summary.md

For target size A (9 or 4) and seed s:
  truth   the property of S{A}_s's own object answer (the self-trained model's behaviour)
  self    S{A}_s's hypothetical answers about that prompt
  cross   X{B}_s's answers (the other size, trained on the same R{A} rows) about the same prompt
  modal   the majority property per family in B{A}'s train-split behaviour
Accuracy per polarity, then averaged over pos / neg (the order check: a yes- or no-bias cancels); items where the
truth or the answer doesn't parse are dropped. Item-cluster bootstrap over (family, prompt), 10,000 draws, rng 0,
seeds pooled.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from self_concept.m18_inward import tasks as T

RES = Path(__file__).resolve().parent / "results" / "main"
N_BOOT = 10000
BAR = 5.0
OTHER = {"9": "4", "4": "9"}


def load(res):
    out = defaultdict(dict)
    for f in Path(res).glob("gen_*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["split"] != "test":
                continue
            out[r["cell"]][(r["family"], r["i"], r["kind"], r["polarity"])] = r
    return out


def truth(cells, cell):
    t = {}
    for (f, i, kind, _), r in cells.get(cell, {}).items():
        if kind == "obj":
            p = T.prop(f, r["text"], r)
            if p is not None:
                t[(f, i)] = p
    return t


def preds(cells, cell):
    out = {}
    for (f, i, kind, pol), r in cells.get(cell, {}).items():
        if kind == "hypo":
            y = T.yes_no(r["text"])
            if y is not None:
                out[(f, i, pol)] = y if pol == "pos" else not y
    return out


def item_scores(tr, pr):
    """{(family, i): mean over polarities of 1[pred == truth]} (polarity-balanced)."""
    out = {}
    for (f, i), v in tr.items():
        ok = [float(pr[(f, i, p)] == v) for p in ("pos", "neg") if (f, i, p) in pr]
        if len(ok) == 2:
            out[(f, i)] = sum(ok) / 2
    return out


def boot(paired, n_boot=N_BOOT):
    """paired: list of (a, b) per item; est and CI of mean(a - b) (b None -> mean(a))."""
    a = np.array([x for x, _ in paired], float)
    b = np.array([0.0 if y is None else y for _, y in paired], float)
    d = a - b
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    draws = d[idx].mean(1)
    return {"est": 100 * float(d.mean()), "lo": 100 * float(np.percentile(draws, 2.5)),
            "hi": 100 * float(np.percentile(draws, 97.5)), "n": int(len(d))}


def analyze(cells, n_boot=N_BOOT):
    out = {"targets": {}}
    for A in ("9", "4"):
        B = OTHER[A]
        res = {"self": [], "cross": [], "per_family": defaultdict(lambda: {"self": [], "cross": []}),
               "polarity": {"self": {"pos": [], "neg": []}, "cross": {"pos": [], "neg": []}}}
        for s in ("1", "2"):
            tr = truth(cells, f"S{A}_s{s}")
            if not tr:
                continue
            ps, pc = preds(cells, f"S{A}_s{s}"), preds(cells, f"X{B}_s{s}")
            ss, sc = item_scores(tr, ps), item_scores(tr, pc)
            for k in ss.keys() & sc.keys():
                res["self"].append(ss[k])
                res["cross"].append(sc[k])
                res["per_family"][k[0]]["self"].append(ss[k])
                res["per_family"][k[0]]["cross"].append(sc[k])
            for who, pr in (("self", ps), ("cross", pc)):
                for (f, i), v in tr.items():
                    for p in ("pos", "neg"):
                        if (f, i, p) in pr:
                            res["polarity"][who][p].append(float(pr[(f, i, p)] == v))
        if not res["self"]:
            continue
        t = {"self": 100 * np.mean(res["self"]), "cross": 100 * np.mean(res["cross"]),
             "self_minus_cross": boot(list(zip(res["self"], res["cross"])), n_boot),
             "per_family": {f: {"self": 100 * np.mean(v["self"]), "cross": 100 * np.mean(v["cross"]), "n": len(v["self"])}
                            for f, v in res["per_family"].items()},
             "polarity": {w: {p: (100 * np.mean(v) if v else None) for p, v in d.items()}
                          for w, d in res["polarity"].items()}}
        out["targets"][A] = t
    return out


def add_modal(out, res_dir):
    """Modal baseline: the majority property per family in B{A}'s train-split object answers, scored against S{A}_s."""
    raw = defaultdict(list)
    for A in ("9", "4"):
        f = Path(res_dir) / f"gen_B{A}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["split"] == "train" and r["kind"] == "obj":
                p = T.prop(r["family"], r["text"], r)
                if p is not None:
                    raw[(A, r["family"])].append(p)
    cells = load(res_dir)
    for A, t in out["targets"].items():
        maj = {f: (np.mean(raw[(A, f)]) >= 0.5) for f in T.FAMILIES if raw[(A, f)]}
        pairs = []
        for s in ("1", "2"):
            tr = truth(cells, f"S{A}_s{s}")
            ss = item_scores(tr, preds(cells, f"S{A}_s{s}"))
            for k, v in ss.items():
                pairs.append((v, float(maj.get(k[0]) == tr[k])))
        t["modal"] = 100 * np.mean([b for _, b in pairs]) if pairs else None
        t["self_minus_modal"] = boot(pairs) if pairs else None
        t["modal_prop"] = {f: bool(v) for f, v in maj.items()}
    return out


def add_base(out, cells):
    """Zero-shot: each base model's hypothetical answers about its own base behaviour."""
    out["zero_shot"] = {}
    for A in ("9", "4"):
        tr = truth(cells, f"B{A}")
        if tr:
            sc = item_scores(tr, preds(cells, f"B{A}"))
            out["zero_shot"][A] = 100 * np.mean(list(sc.values())) if sc else None
    return out


def add_shift(out, cells):
    """Behaviour change (9B): SH9 = S9_s1 + 400 rows moving num answers to odd."""
    new, old = truth(cells, "SH9"), truth(cells, "S9_s1")
    if not new or not old:
        return out
    num = [k for k in new if k[0] == "num" and k in old]
    pr = preds(cells, "SH9")
    changed = [k for k in num if new[k] != old[k]]
    a_new = [np.mean([float(pr.get((k[0], k[1], p)) == new[k]) for p in ("pos", "neg") if (k[0], k[1], p) in pr])
             for k in changed]
    a_old = [np.mean([float(pr.get((k[0], k[1], p)) == old[k]) for p in ("pos", "neg") if (k[0], k[1], p) in pr])
             for k in changed]
    out["shift"] = {"odd_rate_old": 100 * np.mean([not old[k] for k in num]),
                    "odd_rate_new": 100 * np.mean([not new[k] for k in num]), "n_changed": len(changed),
                    "track_new_minus_old": boot(list(zip(a_new, a_old))) if changed else None}
    return out


def verdicts(out):
    for A, t in out["targets"].items():
        c, m = t["self_minus_cross"], t.get("self_minus_modal")
        t["privileged"] = bool(c["est"] >= BAR and c["lo"] > 0 and m and m["est"] >= BAR and m["lo"] > 0)
    sh = out.get("shift")
    if sh and sh.get("track_new_minus_old"):
        x = sh["track_new_minus_old"]
        sh["shift_took"] = sh["odd_rate_new"] - sh["odd_rate_old"] >= 30
        sh["tracks"] = bool(sh["shift_took"] and x["est"] >= 10 and x["lo"] > 0)
    return out


def summary(out):
    f = lambda c: "n/a" if not c else f"{c['est']:+.1f} [{c['lo']:+.1f}, {c['hi']:+.1f}] (n {c['n']})"  # noqa: E731
    L = ["# SC-M18 analysis (test split; polarity-balanced accuracy %)", ""]
    for A, t in out["targets"].items():
        L += [f"## Target {A}B", f"self {t['self']:.1f} | cross {t['cross']:.1f} | modal {t.get('modal')}",
              f"self - cross {f(t['self_minus_cross'])}; self - modal {f(t.get('self_minus_modal'))}",
              f"privileged access: **{t['privileged']}**",
              f"per family: {json.dumps(t['per_family'], default=float)}",
              f"by polarity: {json.dumps(t['polarity'], default=float)}", ""]
    L += [f"zero-shot (untrained base about itself): {out.get('zero_shot')}"]
    if "shift" in out:
        L += [f"shift: {json.dumps(out['shift'], default=float)}"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m18 analyze")
    ap.add_argument("--res", default=str(RES))
    a = ap.parse_args(argv)
    cells = load(a.res)
    out = verdicts(add_shift(add_base(add_modal(analyze(cells), a.res), cells), cells))
    (Path(a.res) / "analysis.json").write_text(json.dumps(out, indent=1, default=float))
    s = summary(out)
    (Path(a.res) / "summary.md").write_text(s)
    print(s)


if __name__ == "__main__":
    main()
