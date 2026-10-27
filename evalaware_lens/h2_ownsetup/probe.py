"""EL-H2 disposition probe (DESIGN.md §5): EL-C3's probe on H2's prompts, on the pod after the acts dumps.

For each pair (trained, reference) in PAIRS and each layer: u = the unit difference of means, trained minus reference,
over the neutral chat prompts of the fit split (both models read the same prompts). Scores s = h . u. Written to
OUT/probe/probe_{trained}__{reference}.json (the activations stay on the pod):
  auc_check    trained vs reference on the held-out chat prompts
  auc_agentic  on the agentic neutral prompts (format transfer)
  auc_task     on the code prompts, all four contexts
  gap_check    mean s_t − s_b on the check prompts; gap_c1 the same on the C1 code prompts (P's denominator)
  shuf_p95     95th percentile over N_SHUF directions fit after permuting model labels over the pooled fit rows, of
               max(auc, 1 − auc) on the check prompts; rand_p95 the same over N_RAND seeded Gaussian unit directions
  rows         per code prompt: key, ctx, tid and both models' scores (analyze.py's input)

  /root/venv/bin/python -m evalaware_lens.h2_ownsetup.probe --out DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evalaware_lens.c3_masking.probe import N_RAND, N_SHUF, auc, fit_dir, load

PAIRS = (("VN-ACT_s1", "VN"), ("VN-ACT_s1", "VN-ACTZ_s1"), ("VN-ACTZ_s1", "VN"))   # primary first


def pair(out, t, b, seed=0):
    At, rt = load(out, t)
    Ab, rb = load(out, b)
    assert [r["key"] for r in rt] == [r["key"] for r in rb], "the two dumps must read the same prompts in order"
    sel = lambda f: [n for n, r in enumerate(rt) if f(r)]  # noqa: E731
    fit, check = sel(lambda r: r["split"] == "fit"), sel(lambda r: r["split"] == "check" and r["fmt"] == "chat")
    agen, code = sel(lambda r: r["fmt"] == "agentic"), sel(lambda r: r["fmt"] == "code")
    c1 = sel(lambda r: r["fmt"] == "code" and r["ctx"] == "C1")
    rng = np.random.default_rng(seed)
    res = {"trained": t, "base": b, "layers": {}}
    for L in sorted(At, key=lambda s: int(s[1:])):
        Xt, Xb = At[L], Ab[L]
        u = fit_dir(Xt[fit], Xb[fit])
        st, sb = Xt @ u, Xb @ u
        pool = np.concatenate([Xt[fit], Xb[fit]])
        shuf, rnd = [], []
        for _ in range(N_SHUF):
            perm = rng.permutation(len(pool))
            v = fit_dir(pool[perm[:len(fit)]], pool[perm[len(fit):]])
            a = auc(Xt[check] @ v, Xb[check] @ v)
            shuf.append(max(a, 1 - a))
        for _ in range(N_RAND):
            g = rng.standard_normal(Xt.shape[1])
            g /= np.linalg.norm(g)
            a = auc(Xt[check] @ g, Xb[check] @ g)
            rnd.append(max(a, 1 - a))
        res["layers"][L] = {
            "auc_check": auc(st[check], sb[check]), "auc_agentic": auc(st[agen], sb[agen]),
            "auc_task": auc(st[code], sb[code]),
            "gap_check": float(np.mean(st[check] - sb[check])), "gap_c1": float(np.mean(st[c1] - sb[c1])),
            "shuf_p95": float(np.percentile(shuf, 95)), "rand_p95": float(np.percentile(rnd, 95)),
            "rows": [{"key": rt[n]["key"], "ctx": rt[n]["ctx"], "tid": rt[n]["tid"], "s_t": float(st[n]),
                      "s_b": float(sb[n])} for n in code],
        }
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="h2 probe")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = Path(a.out, "probe")
    d.mkdir(parents=True, exist_ok=True)
    for t, b in PAIRS:
        if not all(Path(a.out, "acts", f"acts_{c}.npz").exists() for c in (t, b)):
            print(f"[h2-probe] {t} vs {b}: activations missing, skipped", flush=True)
            continue
        r = pair(a.out, t, b)
        (d / f"probe_{t}__{b}.json").write_text(json.dumps(r))
        s = {L: (round(v["auc_check"], 3), round(v["auc_task"], 3), round(v["shuf_p95"], 3)) for L, v in
             r["layers"].items()}
        print(f"[h2-probe] {t} vs {b}: (auc_check, auc_task, shuf_p95) {s}", flush=True)


if __name__ == "__main__":
    main()
