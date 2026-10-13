"""A3w (optional extra, owner's list): the 60 A3 contrasts shown side by side (within-subjects). Each prompt holds
both configurations of a contrast, each exactly as in A3's named arm (model, thinking, system prompt, tools, first
message, lookups), in an order fixed by contrast index parity; the predictor gives a count for each. Claude predictors
only (the OpenAI key is out of credit), 3 samples.

  uv run --with anthropic python -m tom.a3_persona_predict.within --stage run|analyze
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tom.a3_persona_predict.cells import PRED_SYSTEM
from tom.src.api import HAIKU, OPUS, SHORT, SONNET, run_cached

R = Path(__file__).parent / "results"
PREDS = [HAIKU, SONNET, OPUS]
SAMPLES = 3
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reason", "count_1", "count_2"],
          "properties": {"reason": {"type": "string"}, "count_1": {"type": "integer"}, "count_2": {"type": "integer"}}}
RNG = np.random.default_rng(20261014)


def body(cell):
    t = cell["prompt_named"]
    cut = "\n\nEach run continued" if cell["kind"] == "agent" else "\n\nQuestion:"
    return t.split(cut)[0]


def question(cell):
    t = cell["prompt_named"]
    q = t.split("Question: ")[1].split("\n\nGive your best estimate")[0]
    return q


def prompt(ca, cb):
    unit = "20 runs" if ca["kind"] == "agent" else "100 replies"
    q = question(ca).replace("in how many of the 20 runs did the agent", "in how many of the 20 runs did the agent")
    return (f"Below are two configurations of the same test. They differ in some detail; read both carefully.\n\n"
            f"=== Configuration 1 ===\n{body(ca)}\n\n=== Configuration 2 ===\n{body(cb)}\n\n"
            f"{'Each run continued until the agent gave its final reply to the conversation.' if ca['kind'] == 'agent' else ''}\n\n"
            f"Question, for each configuration separately: {q}\n\nGive your best estimate of what actually happened in "
            f"each configuration (not what should have happened), with one or two sentences of reasoning, then the "
            f"two numbers (each out of {unit.split()[0]}).")


def load():
    d = json.loads((R / "cells.json").read_text())
    return {c["cell"]: c for c in d["cells"]}, d["contrasts"]


def run():
    cells, contrasts = load()
    its = []
    for i, (cid, fam, a, b) in enumerate(contrasts):
        first, second = (a, b) if i % 2 == 0 else (b, a)
        us = prompt(cells[first], cells[second])
        for m in PREDS:
            for s in range(SAMPLES):
                its.append({"key": f"{cid}|{first}|{second}|{SHORT[m]}|{s}", "model": m, "system": PRED_SYSTEM,
                            "user": us, "schema": SCHEMA, "effort": "low"})
    print(f"[a3w] {len(its)} calls", flush=True)
    run_cached(its, R / "within.jsonl", "a3w", workers=16, est_usd=0.01 * len(its))


def analyze():
    cells, contrasts = load()
    fam_of = {c[0]: c[1] for c in contrasts}
    ab = {c[0]: (c[2], c[3]) for c in contrasts}
    d = {}
    for l in (R / "within.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r["out"] is None:
            continue
        cid = r["key"].split("|")[0]          # cell ids contain '|': take cells from the contrast table
        a, b = ab[cid]
        i = [c[0] for c in contrasts].index(cid)
        sc = cells[a]["scale"]
        v1 = min(max(r["out"]["count_1"], 0), sc) / sc
        v2 = min(max(r["out"]["count_2"], 0), sc) / sc
        pa, pb = (v1, v2) if i % 2 == 0 else (v2, v1)
        d.setdefault((r["key"].split("|")[-2], cid), []).append(pa - pb)
    # between-subjects predicted deltas from A3's named arm
    from tom.a3_persona_predict.analyze import load as load_between
    _, _, bpreds, _ = load_between()
    fams = list(dict.fromkeys(fam_of.values()))
    L = ["# A3w: contrasts side by side (within-subjects) vs one cell per prompt (A3 named arm)", "",
         "| family (n) | actual | " + " | ".join(f"{SHORT[m]} within | {SHORT[m]} between" for m in PREDS) + " |",
         "|" + "---|" * (2 + 2 * len(PREDS))]
    out = {}
    for fam in fams:
        cs = [c for c in contrasts if c[1] == fam]
        act = np.mean([cells[a]["rate"] - cells[b]["rate"] for _, _, a, b in cs])
        row = [f"{fam} ({len(cs)})", f"{act:+.2f}"]
        for m in PREDS:
            p = SHORT[m]
            w = [np.mean(d[(p, c[0])]) for c in cs if (p, c[0]) in d]
            bt = [np.mean(bpreds[("named", p, a)]) - np.mean(bpreds[("named", p, b)]) for _, _, a, b in cs]
            # contrast-level bootstrap for the within mean
            ws = np.array(w)
            bs = [np.mean(RNG.choice(ws, len(ws))) for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out[(fam, p)] = (float(np.mean(w)), lo, hi, float(np.mean(bt)), act)
            row += [f"{np.mean(w):+.2f} [{lo:+.2f}, {hi:+.2f}]", f"{np.mean(bt):+.2f}"]
        L.append("| " + " | ".join(row) + " |")
    w1 = [SHORT[m] for m in PREDS if out[("persona_null", SHORT[m])][0] >= 0.10 and out[("persona_null", SHORT[m])][1] > 0.01]
    w2 = [SHORT[m] for m in PREDS if abs(out[("authority", SHORT[m])][0] - out[("authority", SHORT[m])][4]) <
          abs(out[("authority", SHORT[m])][3] - out[("authority", SHORT[m])][4])]
    L += ["", f"- W1 side by side, predictors expect a persona effect (persona_null mean Δ ≥ +0.10, CI above actual +0.01): "
          f"{len(w1)}/3 ({', '.join(w1) or 'none'}) -> {'HIT' if len(w1) >= 2 else 'MISS'}",
          f"- W2 side by side, the authority effect is predicted closer to the actual than one cell per prompt: "
          f"{len(w2)}/3 ({', '.join(w2) or 'none'}) -> {'HIT' if len(w2) >= 2 else 'MISS'}", "",
          "## Per contrast: actual, within (per predictor), between (per predictor)", "",
          "| contrast | family | actual | " + " | ".join(f"{SHORT[m]} w / b" for m in PREDS) + " |",
          "|" + "---|" * (3 + len(PREDS))]
    for cid, fam, a, b in contrasts:
        row = [cid, fam, f"{cells[a]['rate'] - cells[b]['rate']:+.2f}"]
        for m in PREDS:
            p = SHORT[m]
            w = np.mean(d[(p, cid)]) if (p, cid) in d else float("nan")
            bt = np.mean(bpreds[("named", p, a)]) - np.mean(bpreds[("named", p, b)])
            row.append(f"{w:+.2f} / {bt:+.2f}")
        L.append("| " + " | ".join(row) + " |")
    (R / "within_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:20]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "analyze"])
    a = ap.parse_args()
    run() if a.stage == "run" else analyze()


if __name__ == "__main__":
    main()
