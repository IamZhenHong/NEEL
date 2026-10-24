"""SC-M29 re-analysis (DESIGN.md): is self-prediction training self-fulfilling? SC-M18's files only, no new runs.

  uv run python -m self_concept.m29_selffulfil.analyze [--res DIR]   -> self_concept/m29_selffulfil/results/{name}/

For target size A (9 / 4) and seed s, on the test split of the gated families (amend1's gate.json):
  beh_B   the ORIGINAL base model B{A}'s property per prompt (fixed)
  beh_S   the self-trained S{A}_s's own property (its new behaviour; SC-M18's ground truth)
  pred_S  S{A}_s's predictions; pred_X  X{B}_s's predictions about A (cross); modal  B{A}'s train-split majority
Accuracy is polarity-balanced per prompt (exact match of the predicted property; both question polarities answered).
  G       acc(pred_S, beh_S) - acc(pred_S, beh_B)      the gain that comes from behaviour moving
  drift   share of prompts whose property changed (beh_S != beh_B); toward = among those, share where beh_S = pred_S
Item-cluster bootstrap over (family, prompt), seeds pooled, 10,000 draws, rng 0.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from self_concept.m18_inward import analyze as M18
from self_concept.m18_inward import tasks as T

_H = Path(__file__).resolve().parent
RES = _H.parent / "m18_inward" / "results" / "amend1"
N_BOOT = 10000
BAR, TOWARD = 5.0, 0.60


def pred_mean(pr, k):
    """The polarity-balanced predicted probability that the property holds (None unless both polarities answered)."""
    v = [pr.get((k[0], k[1], p)) for p in ("pos", "neg")]
    return None if any(x is None for x in v) else v


def acc(pr, beh, k):
    v = pred_mean(pr, k)
    return None if v is None or k not in beh else sum(float(x == beh[k]) for x in v) / 2


def boot(xs, n_boot=N_BOOT):
    x = np.asarray(xs, float)
    if not x.size:
        return {"est": None, "lo": None, "hi": None, "n": 0}
    idx = np.random.default_rng(0).integers(0, len(x), (n_boot, len(x)))
    d = x[idx].mean(1)
    return {"est": float(x.mean()), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "n": int(len(x))}


def modal_of(res, A):
    raw = {}
    for line in (Path(res) / f"gen_B{A}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["split"] == "train" and r["kind"] == "obj":
            p = T.prop(r["family"], r["text"], r)
            if p is not None:
                raw.setdefault(r["family"], []).append(p)
    return {f: np.mean(v) >= 0.5 for f, v in raw.items()}


def analyze(res, fams):
    cells = M18.load(res)
    out = {"families": list(fams), "targets": {}}
    for A in ("9", "4"):
        B = M18.OTHER[A]
        behB = M18.truth(cells, f"B{A}")
        modal = modal_of(res, A)
        rows = []          # one row per (seed, family, prompt) with every score
        for s in ("1", "2"):
            behS = M18.truth(cells, f"S{A}_s{s}")
            behX = M18.truth(cells, f"X{B}_s{s}")
            pS, pX = M18.preds(cells, f"S{A}_s{s}"), M18.preds(cells, f"X{B}_s{s}")
            for k in sorted(behB.keys() & behS.keys()):
                if k[0] not in fams:
                    continue
                r = {"seed": s, "family": k[0], "changed": behS[k] != behB[k],
                     "self_vs_S": acc(pS, behS, k), "self_vs_B": acc(pS, behB, k),
                     "cross_vs_S": acc(pX, behS, k), "cross_vs_B": acc(pX, behB, k),
                     "cross_vs_X": acc(pX, behX, k) if k in behX else None,
                     "modal_vs_S": float(modal.get(k[0]) == behS[k]), "modal_vs_B": float(modal.get(k[0]) == behB[k]),
                     "toward": None}
                if r["changed"] and pred_mean(pS, k) is not None:
                    r["toward"] = sum(float(x == behS[k]) for x in pred_mean(pS, k)) / 2
                rows.append(r)
        ok = [r for r in rows if r["self_vs_S"] is not None and r["self_vs_B"] is not None]

        def summarise(sub):
            g = lambda key: [r[key] for r in sub if r[key] is not None]  # noqa: E731
            return {k: (100 * np.mean(g(k)) if g(k) else None) for k in
                    ("self_vs_S", "self_vs_B", "cross_vs_S", "cross_vs_B", "cross_vs_X", "modal_vs_S", "modal_vs_B")} | {
                "drift": 100 * np.mean([r["changed"] for r in sub]) if sub else None,
                "toward": 100 * np.mean(g("toward")) if g("toward") else None, "n": len(sub)}
        t = {"all": summarise(ok),
             "per_family": {f: summarise([r for r in ok if r["family"] == f]) for f in fams},
             "per_seed": {s: summarise([r for r in ok if r["seed"] == s]) for s in ("1", "2")}}
        G = boot([r["self_vs_S"] - r["self_vs_B"] for r in ok])
        tw = boot([r["toward"] for r in ok if r["toward"] is not None])
        acc_fixed = boot([r["self_vs_B"] - r["cross_vs_B"] for r in ok if r["cross_vs_B"] is not None])
        t |= {"G": {k: (None if v is None else (100 * v if k != "n" else v)) for k, v in G.items()},
              "toward_boot": {k: (None if v is None else (100 * v if k != "n" else v)) for k, v in tw.items()},
              "self_minus_cross_on_fixed": {k: (None if v is None else (100 * v if k != "n" else v))
                                            for k, v in acc_fixed.items()}}
        t["self_fulfilling"] = bool(t["G"]["est"] is not None and t["G"]["est"] >= BAR and t["G"]["lo"] > 0
                                    and tw["est"] is not None and tw["est"] >= TOWARD and tw["lo"] > 0.5)
        t["access_on_fixed"] = bool(t["self_minus_cross_on_fixed"]["est"] is not None
                                    and t["self_minus_cross_on_fixed"]["est"] >= BAR and t["self_minus_cross_on_fixed"]["lo"] > 0)
        out["targets"][A] = t
    return out


def summary(out, name):
    f = lambda x: "n/a" if x is None else f"{x:.1f}"  # noqa: E731
    c = lambda b: "n/a" if not b or b.get("est") is None else f"{b['est']:+.1f} [{b['lo']:+.1f}, {b['hi']:+.1f}]"  # noqa: E731
    L = [f"# SC-M29 re-analysis of SC-M18 ({name}); families {out['families']}", ""]
    keys = ("self_vs_S", "self_vs_B", "cross_vs_S", "cross_vs_B", "cross_vs_X", "modal_vs_S", "modal_vs_B", "drift", "toward")
    for A, t in out["targets"].items():
        L += [f"## Target {A}B", "", "| slice | " + " | ".join(keys) + " | n |", "|---" * (len(keys) + 2) + "|"]
        for name_, d in [("all", t["all"])] + [(f"family {k}", v) for k, v in t["per_family"].items()] + \
                [(f"seed {k}", v) for k, v in t["per_seed"].items()]:
            L.append(f"| {name_} | " + " | ".join(f(d[k]) for k in keys) + f" | {d['n']} |")
        L += ["", f"G = self_vs_S - self_vs_B: {c(t['G'])}; toward (changed prompts): {c(t['toward_boot'])}",
              f"self-fulfilling: **{t['self_fulfilling']}**; self - cross on the fixed base behaviour: "
              f"{c(t['self_minus_cross_on_fixed'])} -> access on fixed behaviour **{t['access_on_fixed']}**", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m29")
    ap.add_argument("--res", default=str(RES))
    ap.add_argument("--name", default="amend1")
    a = ap.parse_args(argv)
    fams = M18.gated(a.res)
    out = analyze(a.res, fams)
    d = _H / "results" / a.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "analysis.json").write_text(json.dumps(out, indent=1, default=float))
    s = summary(out, a.name)
    (d / "summary.md").write_text(s)
    print(s)


if __name__ == "__main__":
    main()
