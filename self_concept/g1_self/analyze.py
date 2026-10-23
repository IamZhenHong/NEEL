"""SC-G1 analysis (DESIGN.md, frozen rules). Per answer: z per set (read.py: random-set z, deep band, mean over the
answer's first-person tokens), primary lens cb, reading "own". Contrasts Delta(set) = mean z(cond) - mean z(ref); answers
are the clusters (bootstrap 10,000, seed 0). Answers with no first-person token are dropped (counted).

  costume   persona - default: CHAR lo > 0 and ASSIST hi < 0 with Delta <= -0.5 -> "the character replaces the
            assistant"; CHAR lo > 0 otherwise -> "costume over the self"; CHAR lo <= 0 -> "no character at I".
  trained   trained - trained_plac: TRAIT lo > 0 and ROLE Delta < 0.5 -> "the trained trait enters the self without
            role content"; TRAIT lo > 0 and ROLE >= 0.5 -> "the trait enters with role content"; else "not at I".
  record    record - default: TRAIT or RECORD lo > 0 -> "the record's content enters I" (which sets); every set's CI
            includes 0 -> "nothing distinctive"; else "other shift" (listed).
  identity  identity - default: TRAIT and ROLE Deltas, and ROLE(identity) - ROLE(persona) (reported; A1 owns it).
  text-only each positive finding above must also beat its text-only reading: Delta_own - Delta_text lo > 0 ->
            "self-specific", else "follows the text".
  gate      the 9B lens CAL (self_concept/cal rules on plain Qwen3.5-9B, N3's pod) passes on cb; else every label is
            descriptive. The logit lens is reported.

  python -m self_concept.g1_self.analyze --res DIR [--cal DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from self_concept.g1_self import items as I

B = 10_000
PRIM = "cb"


def load(res):
    rows = []
    for p in sorted((Path(res) / "reads").glob("reads_*.jsonl")):
        rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return rows


def contrast(rows, a, b, lens=PRIM, rd="own", rng=None):
    """{set: [Delta, lo, hi]} mean z(a) - mean z(b), answers resampled within each condition."""
    rng = rng or np.random.default_rng(0)
    out = {}
    za = [r["z"][lens][rd] for r in rows if r["cond"] == a and rd in r["z"].get(lens, {})]
    zb = [r["z"][lens][rd] for r in rows if r["cond"] == b and rd in r["z"].get(lens, {})]
    if not za or not zb:
        return None
    for k in I.SETS:
        xa, xb = np.array([z[k] for z in za]), np.array([z[k] for z in zb])
        bs = xa[rng.integers(0, len(xa), (B, len(xa)))].mean(1) - xb[rng.integers(0, len(xb), (B, len(xb)))].mean(1)
        out[k] = [float(xa.mean() - xb.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def own_minus_text(rows, a, b, k, lens=PRIM, rng=None):
    rng = rng or np.random.default_rng(1)
    def d(c):
        return np.array([r["z"][lens]["own"][k] - r["z"][lens]["text"][k] for r in rows
                         if r["cond"] == c and {"own", "text"} <= set(r["z"].get(lens, {}))])
    xa, xb = d(a), d(b)
    if not len(xa) or not len(xb):
        return None
    bs = xa[rng.integers(0, len(xa), (B, len(xa)))].mean(1) - xb[rng.integers(0, len(xb), (B, len(xb)))].mean(1)
    return [float(xa.mean() - xb.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def analyse(res, cal=None, lens=PRIM):
    rows = load(res)
    out = {"n": {c: sum(r["cond"] == c for r in rows) for c in I.CONDS},
           "dropped_no_self_token": {c: sum(r["cond"] == c and not r.get("n_pos_own") for r in rows) for c in I.CONDS}}
    cal_ok = None
    if cal and Path(cal, "analysis.json").exists():
        hit = [r for r in json.loads(Path(cal, "analysis.json").read_text()) if r["model"] == "qwen9" and r["lens"] == PRIM]
        cal_ok = bool(hit and hit[0]["CAL_PASS"])
    out["gate"] = cal_ok
    rows = [r for r in rows if r.get("n_pos_own")]
    tag = (lambda lab: lab) if cal_ok else (lambda lab: f"descriptive ({lab})")
    C = {}
    for name, (a, b) in {"costume": ("persona", "default"), "trained": ("trained", "trained_plac"),
                         "record": ("record", "default"), "identity": ("identity", "default")}.items():
        C[name] = contrast(rows, a, b, lens)
    out["contrasts"] = C
    labels, checks = {}, {}
    c = C["costume"]
    if c:
        lab = ("the character replaces the assistant" if c["CHAR"][1] > 0 and c["ASSIST"][2] < 0 and c["ASSIST"][0] <= -0.5
               else "costume over the self" if c["CHAR"][1] > 0 else "no character at I")
        if c["CHAR"][1] > 0:
            checks["costume"] = own_minus_text(rows, "persona", "default", "CHAR", lens)
        labels["costume"] = lab
    c = C["trained"]
    if c:
        lab = ("the trained trait enters the self without role content" if c["TRAIT"][1] > 0 and c["ROLE"][0] < 0.5
               else "the trait enters with role content" if c["TRAIT"][1] > 0 else "not at I")
        if c["TRAIT"][1] > 0:
            checks["trained"] = own_minus_text(rows, "trained", "trained_plac", "TRAIT", lens)
        labels["trained"] = lab
    c = C["record"]
    if c:
        up = [k for k in ("TRAIT", "RECORD") if c[k][1] > 0]
        moved = [k for k in I.SETS if c[k][1] > 0 or c[k][2] < 0]
        lab = (f"the record's content enters I ({', '.join(up)})" if up else
               "nothing distinctive" if not moved else f"other shift ({', '.join(moved)})")
        if up:
            checks["record"] = own_minus_text(rows, "record", "default", up[0], lens)
        labels["record"] = lab
    for k, v in checks.items():
        if v is not None:
            labels[k] += "; self-specific" if v[1] > 0 else "; follows the text"
    out["checks_own_minus_text"] = checks
    out["labels"] = {k: tag(v) for k, v in labels.items()}
    if C["identity"] and C["costume"]:
        out["identity_vs_persona_ROLE"] = C["identity"]["ROLE"][0] - C["costume"]["ROLE"][0]
    if lens == PRIM:
        out["logit_lens"] = {k: v for k, v in analyse(res, cal, "logit").items() if k in ("contrasts", "labels")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="g1 analyze")
    ap.add_argument("--res", required=True)
    ap.add_argument("--cal")
    a = ap.parse_args(argv)
    out = analyse(a.res, a.cal)
    Path(a.res, "analysis.json").write_text(json.dumps(out, indent=1))
    L = ["# SC-G1 summary", "", f"gate (9B lens CAL): {out['gate']}; answers {out['n']}", ""]
    for k, v in out["contrasts"].items():
        if v:
            L.append(f"- {k}: " + ", ".join(f"{s} {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]" for s, d in v.items()))
    L += [""] + [f"- **{k}**: {v}" for k, v in out["labels"].items()]
    Path(a.res, "summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
