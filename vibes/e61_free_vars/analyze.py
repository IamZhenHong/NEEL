"""e61 analysis (rules fixed before the main run; README "Analysis").

  uv run python vibes/e61_free_vars/analyze.py --run vibes/e61_free_vars/results/main
Writes <run>/analysis.json and prints the README tables."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from e61lib import TOPICS, load_jsonl  # noqa: E402
from reads import agreement_with_A, gate  # noqa: E402

BATS = ["EM", "HC", "SY", "TR", "FAB"]
SIDES = {"T1": ("A1", "B1"), "T2": ("A2", "B2")}
BAND = 5.0           # +-5 points (card headline, P2, P3)
P = {"P1": 0.7, "P2": 0.4, "P3": 0.3, "P4": 0.4, "P5a": 0.3, "P5b": 0.5, "P6": 0.6}


def boot_mean(x, n=10_000, seed=0, cluster=None):
    """Mean with a 95% bootstrap CI; cluster: list of cluster keys (resampled as units)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    if cluster is None:
        idx = rng.integers(0, len(x), (n, len(x)))
        bs = x[idx].mean(1)
    else:
        keys = sorted(set(cluster)); groups = [np.where(np.array(cluster) == k)[0] for k in keys]
        bs = np.empty(n)
        for i in range(n):
            pick = rng.integers(0, len(groups), len(groups))
            bs[i] = np.concatenate([x[groups[j]] for j in pick]).mean()
    return {"mean": float(x.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n": len(x)}


def inside(c, band=BAND):
    return c["lo"] >= -band and c["hi"] <= band


def excludes0(c):
    return c["lo"] > 0 or c["hi"] < 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(HERE / "results" / "main"))
    ap.add_argument("--no-vb", action="store_true")
    a = ap.parse_args()
    run = Path(a.run); jd = run / "judge"
    res = {}

    # ---------------------------------------------------------------- value check (statements) + gate
    hf = load_jsonl(run / "reads.jsonl")
    reads = defaultdict(list)
    for r in hf:
        reads[r["cond"]].append(r)
    conds = list(reads)
    res["agreeA"] = {c: {t: agreement_with_A(reads[c], t) for t in TOPICS} for c in conds}
    res["gate"] = {t: gate(reads, t) for t in TOPICS if all(x in reads for x in SIDES[t])}
    passing = [t for t, g in res["gate"].items() if g["pass"]]
    res["passing_topics"] = passing
    res["yesno_mass"] = {c: float(np.mean([r["mass"] for r in reads[c]])) for c in conds}

    # ---------------------------------------------------------------- engine check (vLLM vs HF Yes/No reads)
    if (run / "agree_vllm.jsonl").exists():
        vl = {(r["cond"], r["id"]): r for r in load_jsonl(run / "agree_vllm.jsonl")}
        ec = {}
        for c in conds:
            pairs = [(r["p_agree"], vl[(c, r["id"])]["p_agree"]) for r in reads[c] if (c, r["id"]) in vl
                     and vl[(c, r["id"])]["p_agree"] is not None]
            h, v = np.array(pairs).T if pairs else (np.array([]), np.array([]))
            ec[c] = {"n": len(pairs), "argmax_agree": float(np.mean((h > 0.5) == (v > 0.5))) if pairs else None,
                     "mean_abs_diff": float(np.mean(np.abs(h - v))) if pairs else None}
        res["engine_check"] = ec

    # ---------------------------------------------------------------- free-form stance
    if (jd / "stance_free.jsonl").exists():
        sf = [r for r in load_jsonl(jd / "stance_free.jsonl") if r["stance"] is not None]
        res["free_stance"] = {}
        for c in conds:
            for t in TOPICS:
                rows = [r for r in sf if r["cond"] == c and r["topic"] == t]
                if rows:
                    res["free_stance"][f"{c}|{t}"] = boot_mean([r["stance"] for r in rows],
                                                               cluster=[r["id"].split("|")[1] for r in rows])

    # ---------------------------------------------------------------- spillover
    sp = {}
    for t, (A, B) in SIDES.items():
        if A not in reads or B not in reads:
            continue
        pa = {r["id"]: r for r in reads[A] if r["kind"] == "spill"}
        pb = {r["id"]: r for r in reads[B] if r["kind"] == "spill"}
        pu = {r["id"]: r for r in reads["untouched"] if r["kind"] == "spill"}
        pc = {r["id"]: r for r in reads.get("C", []) if r["kind"] == "spill"}
        per = []
        for i, r in pa.items():
            d = r["p_agree"] - pb[i]["p_agree"]
            # aligned sign: + when A-B moves toward the lean that matches side A's view in the same domain
            lean = r.get("lean")
            al = None
            if t == "T1" and lean in ("punitive", "lenient"):
                al = d if lean == "punitive" else -d
            if t == "T2" and lean in ("human_use", "animal_protective"):
                al = d if lean == "human_use" else -d
            per.append({"id": i, "set": r["set"], "lean": lean, "A": r["p_agree"], "B": pb[i]["p_agree"], "A_minus_B": d,
                        "aligned": al, "C_minus_untouched": (pc[i]["p_agree"] - pu[i]["p_agree"]) if pc else None})
        s = {"per_statement": per}
        for st in ("near", "far"):
            rows = [x for x in per if x["set"] == st]
            s[f"mean_abs_{st}"] = boot_mean([abs(x["A_minus_B"]) for x in rows])
            s[f"mean_signed_{st}"] = boot_mean([x["A_minus_B"] for x in rows])
            if pc:
                s[f"ref_mean_abs_C_minus_untouched_{st}"] = float(np.mean([abs(x["C_minus_untouched"]) for x in rows]))
        dom = [x["aligned"] for x in per if x["aligned"] is not None]
        if dom:
            s["near_same_domain_aligned"] = boot_mean(dom)
        sp[t] = s
    res["spillover"] = sp

    # ---------------------------------------------------------------- register per condition
    reg = {}
    if (jd / "register_eval.jsonl").exists():
        rg = [r for r in load_jsonl(jd / "register_eval.jsonl") if r["reg"]]
        for c in conds:
            for src in ("EM", "benign", "all"):
                rows = [r for r in rg if r["cond"] == c and (src == "all" or r["src"] == src)]
                if rows:
                    reg[f"{c}|{src}"] = {"warmth": boot_mean([r["reg"][0] for r in rows]),
                                         "harshness": boot_mean([r["reg"][1] for r in rows]), "n": len(rows)}
    res["register"] = reg

    # ---------------------------------------------------------------- batteries
    if not a.no_vb and (jd / "vb_judgments.json").exists():
        from vibes.batteries import PRIMARY, compare, score, subtable
        vb = load_jsonl(run / "vb_outputs.jsonl")
        J = json.load(open(jd / "vb_judgments.json"))
        metrics, table = score(vb, J)
        res["vb_metrics"] = {f"{m}|{c}": v for (m, c), v in metrics.items()}
        res["vb_flags"] = {f"{m}|{c}": {b: {k: v for k, v in mv.items() if "flag" in k.lower() or "LOW" in str(v)}
                                          for b, mv in v.items()} for (m, c), v in metrics.items()}

        def cmp(x, y):
            out = {}
            for b in BATS:
                r = compare(subtable(table, cond=x), subtable(table, cond=y), batteries=[b], metrics=[PRIMARY[b]])
                out[b] = r.get((b, PRIMARY[b]))
            return out
        res["delta_vs_untouched"] = {c: cmp(c, "untouched") for c in ("C", "A1", "B1", "A2", "B2") if c in conds}
        res["A_minus_B"] = {t: cmp(*SIDES[t]) for t in TOPICS}

    # ---------------------------------------------------------------- predictions
    pr = {}
    pr["P1"] = len(passing) == len(TOPICS)
    if "A_minus_B" in res:
        dvu, amb = res["delta_vs_untouched"], res["A_minus_B"]
        pr["P2"] = all(inside(amb[t][b]) for t in passing for b in BATS) if passing else None
        pr["P3"] = all(inside(dvu[c][b]) for c in dvu for b in BATS)
        p4 = False
        for b in BATS:
            dc = dvu["C"][b]
            if not excludes0(dc):
                continue
            for t, (A, B) in SIDES.items():
                if abs(dc["diff"]) >= min(abs(dvu[A][b]["diff"]), abs(dvu[B][b]["diff"])):
                    p4 = True
        pr["P4"] = p4
        cells = []
        for t in passing:
            A, B = SIDES[t]
            hA, hB = reg.get(f"{A}|all", {}).get("harshness"), reg.get(f"{B}|all", {}).get("harshness")
            if not hA or not hB:
                continue
            harsher = A if hA["mean"] > hB["mean"] else B
            for b in BATS:
                c = amb[t][b]
                if excludes0(c):
                    worse = A if c["diff"] > 0 else B          # higher is worse on every primary
                    cells.append({"topic": t, "battery": b, "harsher": harsher, "worse": worse, "match": harsher == worse})
        res["P6_cells"] = cells
        pr["P6"] = (sum(x["match"] for x in cells) > len(cells) / 2) if cells else None
    if sp:
        pr["P5a"] = float(np.mean([sp[t]["mean_abs_far"]["mean"] for t in sp])) >= 0.10
        pr["P5b"] = float(np.mean([sp[t]["mean_abs_near"]["mean"] for t in sp])) >= 0.10
    resolved = {k: v for k, v in pr.items() if v is not None}
    da = [((P[k] > 0.5) == v) for k, v in resolved.items() if P[k] != 0.5]
    res["predictions"] = {"outcomes": pr, "p": P, "directional": f"{sum(da)}/{len(da)}",
                          "brier": float(np.mean([(P[k] - float(v)) ** 2 for k, v in resolved.items()])) if resolved else None}
    json.dump(res, open(run / "analysis.json", "w"), indent=1, default=str)

    # ---------------------------------------------------------------- print
    print("## gate"); [print(t, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()})) for t, g in res["gate"].items()]
    if "engine_check" in res:
        print("## engine check", json.dumps(res["engine_check"]))
    if "free_stance" in res:
        print("## free-form stance"); [print(k, f"{v['mean']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]") for k, v in res["free_stance"].items()]
    print("## spillover")
    for t, s in sp.items():
        print(t, {k: (f"{v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" if isinstance(v, dict) else round(v, 3))
                  for k, v in s.items() if k != "per_statement"})
    if reg:
        print("## register"); [print(k, f"warmth {v['warmth']['mean']:.2f} harsh {v['harshness']['mean']:.2f} n {v['n']}") for k, v in reg.items()]
    if "A_minus_B" in res:
        print("## batteries: delta vs untouched (pp) and A-B")
        for c, d in res["delta_vs_untouched"].items():
            print(c, " ".join(f"{b} {d[b]['diff']:+.1f} [{d[b]['lo']:+.1f},{d[b]['hi']:+.1f}]" for b in BATS))
        for t, d in res["A_minus_B"].items():
            print(f"A-B {t}", " ".join(f"{b} {d[b]['diff']:+.1f} [{d[b]['lo']:+.1f},{d[b]['hi']:+.1f}]" for b in BATS))
    print("## predictions", json.dumps(res["predictions"]))


if __name__ == "__main__":
    main()
