"""SC-N6a (DESIGN.md, frozen rules): does P11b's "the voice doesn't matter" and "the model claims the policy as its own"
survive formats unlike the training Q&A? P11b's stored outputs only (persona_affect/p11b_voice/results/main, e7ffe97),
read-only; P11b's own analysis never scored mc, scale or describe.

  formats  mc (an ATT situation on a desk outside every train / eval domain, A / B / C, "which would you do?"),
           scale (0-100 self-rating), describe (open self-description, V1c's two judges and labels: self 1, mixed 0.5,
           refer 0), predict (forecast with the case open); scored with V1c's score().
  E_f(X)   rate_f(X) - rate_f(PLAC), pp; gap_f(X) = rate_f(SAYU) - rate_f(X). Arms SAYU ASST OTH VAL PLAC, seeds 1-2
           pooled; bootstrap 10,000 draws, seed 0, over the format's clusters (situations / questions / tasks).
  rule f   SAYU's E_f CI includes 0: "the format doesn't carry the policy"; else "survives" if every X in ASST, OTH,
           VAL has E_f lo > 0 and gap_f < 15; else "fails".
  claims   describe: E(OTH) >= 30 with lo > 0: "OTH claims the policy in free text"; E(OTH) lo <= 0: "not in free
           text"; else "partial". Judges' kappa < 0.6: describe is descriptive only.

  uv run python -m self_concept.n6a_format.analyze judge [--cap 1.0]
  uv run python -m self_concept.n6a_format.analyze main
"""
import argparse
import json
from pathlib import Path

import numpy as np

import self_concept.common.ledger  # noqa: F401  (spend goes to self_concept/ledger.jsonl)
from persona_affect.v1c_selfreport import analyze as V1C

SRC = Path(__file__).resolve().parents[2] / "persona_affect/p11b_voice/results/main"
RES = Path(__file__).resolve().parent / "results"
ARMS = ("SAYU", "ASST", "OTH", "VAL", "PLAC")
VOICES = ("ASST", "OTH", "VAL")
FORMATS = ("mc", "scale", "describe", "predict")
B = 10_000


def load(src=SRC):
    recs = []
    for a in ARMS:
        for s in (1, 2):
            p = Path(src) / f"eval_NZ-{a}_s{s}.jsonl"
            recs += [json.loads(x) | {"arm": a, "seed": s} for x in p.read_text().splitlines() if x.strip()]
    return [r for r in recs if r["block"] in FORMATS]


def cmd_judge(a):
    from persona_affect.common import judges as J
    from self_concept.common.ledger import check_cap
    from persona_affect.v1c_selfreport.items import DESCRIBE_Q
    check_cap("SC-N6a", 2.0, planned=a.cap)
    recs = [r for r in load(a.src) if r["block"] == "describe"]
    items = [{"key": f"{r['key']}|{r['cell']}", "q": DESCRIBE_Q[int(r["tid"][1:])], "a": r["text"]} for r in recs]
    RES.mkdir(exist_ok=True)
    labels = J.label(items, lambda it: V1C.JUDGE_PROMPT.format(q=it["q"], a=it["a"]),
                     {"self", "refer", "mixed", "unclear"}, mini="SC-N6a", cap=a.cap, cache=RES / "judge_cache.jsonl")
    kappa = J.agreement(labels)
    (RES / "judged.json").write_text(json.dumps({"labels": labels, "kappa": kappa, "spent": labels.spent}, indent=1))
    print(f"[n6a] judged {len(items)} describe answers: kappa {kappa.get('kappa')}, spent ${labels.spent:.3f}")


def analyse(recs, judged=None, kappa=None):
    rng = np.random.default_rng(0)
    out = {"formats": {}, "kappa": kappa}
    for f in FORMATS:
        rs = [r for r in recs if r["block"] == f]
        cl = sorted({r["tid"] for r in rs})
        idx = rng.integers(0, len(cl), (B, len(cl)))
        rate, pt = {}, {}
        unparsed = {}
        for arm in ARMS:
            ys = {c: [] for c in cl}
            n = 0
            for r in rs:
                if r["arm"] != arm:
                    continue
                n += 1
                y = V1C.score(r, judged)
                if y is not None:
                    ys[r["tid"]].append(y)
            s = np.array([sum(ys[c]) for c in cl])
            m = np.array([len(ys[c]) for c in cl])
            with np.errstate(invalid="ignore", divide="ignore"):
                rate[arm] = 100 * s[idx].sum(1) / m[idx].sum(1)
            pt[arm] = 100 * s.sum() / m.sum() if m.sum() else float("nan")
            unparsed[arm] = 1 - m.sum() / n if n else None

        def ci(bs, p):
            return [float(p), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
        E = {x: ci(rate[x] - rate["PLAC"], pt[x] - pt["PLAC"]) for x in ARMS if x != "PLAC"}
        gap = {x: ci(rate["SAYU"] - rate[x], pt["SAYU"] - pt[x]) for x in VOICES}
        if E["SAYU"][1] <= 0:
            lab = "the format doesn't carry the policy"
        elif all(E[x][1] > 0 and gap[x][0] < 15 for x in VOICES):
            lab = "survives"
        else:
            lab = "fails"
        if f == "describe" and (kappa is None or kappa < 0.6):
            lab = f"descriptive ({lab}; kappa {kappa})"
        out["formats"][f] = {"rate": {a: round(pt[a], 1) for a in ARMS}, "E": E, "gap": gap, "label": lab,
                             "unparsed": unparsed, "clusters": len(cl)}
    d = out["formats"]["describe"]["E"]["OTH"]
    claims = ("OTH claims the policy in free text" if d[0] >= 30 and d[1] > 0 else
              "not in free text" if d[1] <= 0 else "partial")
    out["claims"] = claims if kappa is not None and kappa >= 0.6 else f"descriptive ({claims})"
    out["survives_in"] = [f for f in FORMATS if out["formats"][f]["label"] == "survives"]
    return out


def cmd_main(a):
    jf = RES / "judged.json"
    j = json.loads(jf.read_text()) if jf.exists() else {}
    kappa = (j.get("kappa") or {}).get("kappa")
    out = analyse(load(a.src), j.get("labels"), kappa)
    RES.mkdir(exist_ok=True)
    (RES / "analysis.json").write_text(json.dumps(out, indent=1))
    L = ["# SC-N6a summary", "", "| format | SAYU | ASST | OTH | VAL | PLAC | label |", "|---|---|---|---|---|---|---|"]
    for f, v in out["formats"].items():
        L.append(f"| {f} | " + " | ".join(f"{v['rate'][x]:.1f}" for x in ARMS) + f" | {v['label']} |")
    L += ["", f"- claims (describe, OTH - PLAC {out['formats']['describe']['E']['OTH'][0]:+.1f}): {out['claims']}",
          f"- equivalence survives in: {out['survives_in']}; judges kappa {kappa}"]
    (RES / "summary.md").write_text("\n".join(L))
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="n6a")
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--cap", type=float, default=1.0)
    j.add_argument("--src", default=str(SRC))
    m = sub.add_parser("main")
    m.add_argument("--src", default=str(SRC))
    a = ap.parse_args(argv)
    {"judge": cmd_judge, "main": cmd_main}[a.cmd](a)


if __name__ == "__main__":
    main()
