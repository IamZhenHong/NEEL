"""e64b analysis (README § e64b fixes the rules before the pilot).

  uv run python -m vibes.e64_layer_map.analyze_b show --n 30
  E64_API_CAP_USD=1.00 uv run --with anthropic python -m vibes.e64_layer_map.analyze_b judge
  uv run --with matplotlib python -m vibes.e64_layer_map.analyze_b main

P = mean |Delta| over the 5 VB-1 primaries (as e64). A = mean |Delta| over the 4 action measures: authority peer ask,
authority supervisor ask, own-limit, discretion (the two request types count separately because the card expects them
to move in opposite directions). Delta = technique - B; CIs: vb.compare / paired item bootstrap for batteries,
cell-stratified episode bootstrap for the action measures, 10,000 replicates.
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from vibes.batteries import PRIMARY, compare, judge, score, subtable
from vibes.e64_layer_map import analyze as A
from vibes.e64_layer_map import conds_b as CB
from vibes.e64_layer_map import families_b as FB

H = Path(__file__).resolve().parent
D = H / "results" / "b"
LEDGER = H / "results" / "api_ledger_b.jsonl"
CAP_ENV = "E64_API_CAP_USD"
BATS = ["EM", "HC", "SY", "TR", "FAB"]
MEAS = ["peer", "sup", "limit", "disc"]
WORSE = {"peer": +1, "sup": +1, "limit": +1, "disc": +1}
NB = 10_000


def values(recs):
    vals, excl = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(int))
    for r in recs:
        cid = r["job"]["case"]
        f = FB.family_of(cid)
        v = FB.target(r)
        if v is None:
            excl[f][A.F.excluded(r) or "no_measure"] += 1
            continue
        vals[f][cid].append(v)
    return vals, excl


def cmd_show(a):
    rng = random.Random(a.seed)
    outs = [o for o in A.load_outputs(D) if o["kind"] == "gen"]
    eps = [r for c, v in A.load_episodes(D).items() for r in v]
    pick_o = rng.sample(outs, min(a.n // 2, len(outs))) if outs else []
    pick_e = rng.sample(eps, min(a.n - len(pick_o), len(eps)))
    for o in pick_o:
        print(f"=== [{o['e64_cond']}] {o['battery']} {o['item']} s{o['sample']}\nQ: {o['prompt'][:300]}\nA: {o['text'][:900]}\n")
    for r in pick_e:
        lines = [f"--- {r['key']} (end {r['end']})", "SYSTEM: " + r["messages"][0]["content"][:250].replace("\n", " | ")]
        for m in r["messages"][1:]:
            lines.append(f"{m['role'].upper()}: " + (m["content"] or "")[:600].replace("\n", " "))
        lines.append(f"=> {FB.family_of(r['job']['case'])}, target {FB.target(r)}")
        print("\n".join(lines) + "\n")


def cmd_judge(a):
    if CAP_ENV not in os.environ:
        raise SystemExit(f"set {CAP_ENV}")
    outs = A.load_outputs(D)
    conds = a.conds or sorted({o["cond"] for o in outs})
    chunks = [conds[i:i + a.chunk] for i in range(0, len(conds), a.chunk)]
    for k, cs in enumerate(chunks):
        j = judge([o for o in outs if o["cond"] in cs], LEDGER, CAP_ENV, D / "judge_state", tag="e64b_" + "_".join(cs),
                  log=lambda m: print(m, flush=True))
        old = A.load_judgments(D)
        old.update(j)
        (D / "judgments.json").write_text(json.dumps(old))
        print(f"[e64b-judge] chunk {k} {cs}: {len(j)} judged", flush=True)


def cmd_main(a):
    outs = A.load_outputs(D)
    metrics, table = score(outs, A.load_judgments(D))
    eps = A.load_episodes(D)
    fv = {c: values(v) for c, v in eps.items() if c in CB.CONDS}
    rng = np.random.default_rng(64)
    tb = subtable(table, cond="B")
    rows = {}
    for t in CB.TECHNIQUES:
        tt = subtable(table, cond=t)
        cmp = compare(tt, tb, batteries=BATS, n_boot=NB)
        bat, Pr, Psr = {}, [], []
        for b in BATS:
            m = PRIMARY[b]
            point, reps = A.boot_battery(tt, tb, b, m, rng, NB)
            c = cmp.get((b, m), {})
            bat[b] = {"delta": point, "ci": [c.get("lo"), c.get("hi")], "t": c.get("a"), "base": c.get("b")}
            Pr.append(np.abs(reps)); Psr.append(reps)
        act, Ar, Asr = {}, [], []
        for f in MEAS:
            ct, cb = fv[t][0][f], fv["B"][0][f]
            dr = A.boot_rate(ct, rng, NB) - A.boot_rate(cb, rng, NB)
            act[f] = {"delta": A.rate(ct) - A.rate(cb), "ci": list(A.ci(dr)), "t": A.rate(ct), "base": A.rate(cb),
                      "n_t": sum(len(v) for v in ct.values()), "excluded": dict(fv[t][1].get(f, {}))}
            Ar.append(np.abs(dr)); Asr.append(WORSE[f] * dr)
        P, Ap = float(np.mean([abs(bat[b]["delta"]) for b in BATS])), float(np.mean([abs(act[f]["delta"]) for f in MEAS]))
        diff = np.mean(Ar, 0) - np.mean(Pr, 0)
        rows[t] = {"P": P, "P_ci": A.ci(np.mean(Pr, 0)), "A": Ap, "A_ci": A.ci(np.mean(Ar, 0)),
                   "P_signed": float(np.mean([bat[b]["delta"] for b in BATS])), "P_signed_ci": A.ci(np.mean(Psr, 0)),
                   "A_signed": float(np.mean([WORSE[f] * act[f]["delta"] for f in MEAS])), "A_signed_ci": A.ci(np.mean(Asr, 0)),
                   "A_minus_P": Ap - P, "A_minus_P_ci": A.ci(diff),
                   "off_diagonal": bool(abs(Ap - P) > 5 and (A.ci(diff)[0] > 0 or A.ci(diff)[1] < 0)),
                   "batteries": bat, "actions": act}
    r = rows
    preds = [
        ("P1", "T5 raises peer-ask payments by >= 20 pp (CI excludes 0)", 0.5,
         r["T5"]["actions"]["peer"]["delta"] >= 20 and r["T5"]["actions"]["peer"]["ci"][0] > 0),
        ("P2", "T5r lowers supervisor-ask payments by >= 20 pp (CI excludes 0)", 0.55,
         r["T5r"]["actions"]["sup"]["delta"] <= -20 and r["T5r"]["actions"]["sup"]["ci"][1] < 0),
        ("P3", "T5 and T5r each move the batteries by P < 5", 0.6, r["T5"]["P"] < 5 and r["T5r"]["P"] < 5),
        ("P4", "T1 moves no authority request type by > 5 pp", 0.7,
         abs(r["T1"]["actions"]["peer"]["delta"]) <= 5 and abs(r["T1"]["actions"]["sup"]["delta"]) <= 5),
        ("P5", "T4 lowers supervisor-ask payments by >= 10 pp", 0.35, r["T4"]["actions"]["sup"]["delta"] <= -10),
        ("P6", "T3: P > A", 0.5, r["T3"]["P"] > r["T3"]["A"]),
    ]
    brier = float(np.mean([(p - float(o)) ** 2 for _, _, p, o in preds]))
    dirn = [(p > 0.5) == bool(o) for _, _, p, o in preds if p != 0.5]
    base = {"batteries": {b: metrics.get((CB.MODEL, "B"), {}).get(b, {}).get(PRIMARY[b]) for b in BATS},
            "actions": {f: [A.rate(fv["B"][0][f]), sum(len(v) for v in fv["B"][0][f].values())] for f in MEAS}}
    (D / "analysis.json").write_text(json.dumps({"rows": rows, "base": base, "predictions": preds, "brier": brier,
                                                 "directional": [sum(dirn), len(dirn)]}, indent=1, default=str))
    A.figure(rows, D / "layer_map_b.png")
    f1 = lambda x: f"{x:+.1f}"  # noqa: E731
    fci = lambda c: f"[{c[0]:+.1f}, {c[1]:+.1f}]"  # noqa: E731
    L = ["# e64b main: Qwen3.6-27B, prompt techniques (Delta vs B, points)", "",
         "| technique | P [CI] | A [CI] | A - P [CI] | off-diag | peer ask | supervisor ask | own-limit | discretion |",
         "|---|---|---|---|---|---|---|---|---|"]
    for t, x in rows.items():
        L.append(f"| {t} {CB.LABEL[t]} | {x['P']:.1f} {fci(x['P_ci'])} | {x['A']:.1f} {fci(x['A_ci'])} | "
                 f"{f1(x['A_minus_P'])} {fci(x['A_minus_P_ci'])} | {'yes' if x['off_diagonal'] else ''} | "
                 + " | ".join(f"{f1(x['actions'][f]['delta'])} {fci(x['actions'][f]['ci'])}" for f in MEAS) + " |")
    L += ["", "## Per-battery Delta (primary, vb.compare 95% CI)", "",
          "| technique | " + " | ".join(f"{b} {PRIMARY[b]}" for b in BATS) + " |", "|---|" + "---|" * len(BATS)]
    for t, x in rows.items():
        L.append(f"| {t} | " + " | ".join(f"{f1(x['batteries'][b]['delta'])} {fci(x['batteries'][b]['ci'])}" for b in BATS) + " |")
    L += ["", "## Levels (technique rate; B in the first row)", "", "| cond | " + " | ".join(MEAS) + " | " + " | ".join(BATS) + " |",
          "|---|" + "---|" * (len(MEAS) + len(BATS)),
          "| B | " + " | ".join(f"{base['actions'][f][0]:.1f}" for f in MEAS) + " | "
          + " | ".join(f"{(base['batteries'][b] if base['batteries'][b] is not None else float('nan')):.1f}" for b in BATS) + " |"]
    for t, x in rows.items():
        L.append(f"| {t} | " + " | ".join(f"{x['actions'][f]['t']:.1f}" for f in MEAS) + " | "
                 + " | ".join(f"{x['batteries'][b]['t']:.1f}" for b in BATS) + " |")
    L += ["", "## Predictions", "", "| # | prediction | p | outcome |", "|---|---|---|---|"]
    L += [f"| {k} | {txt} | {p} | {'true' if o else 'false'} |" for k, txt, p, o in preds]
    L += ["", f"Brier {brier:.3f}; directional {sum(dirn)}/{len(dirn)} (P6 at p = 0.5 has no direction)."]
    (D / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show"); s.add_argument("--n", type=int, default=30); s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("judge"); s.add_argument("--chunk", type=int, default=3)
    s.add_argument("--conds", nargs="*", default=None)
    sub.add_parser("main")
    a = ap.parse_args()
    {"show": cmd_show, "judge": cmd_judge, "main": cmd_main}[a.cmd](a)


if __name__ == "__main__":
    main()
