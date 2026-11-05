"""Judge calibration v1 (labeling/judgecal_v1): run candidate judges on the 100 unique calibration replies with the same rubrics
and context the production judges use, then score every judge against the human labels.
  run:   <hackathon venv python> retention/judge_calibrate.py run --model claude-haiku-5-5 [--effort low] [--budget 0.5]
  score: <hackathon venv python> retention/judge_calibrate.py score --labels ~/Downloads/judgecal_v1_labels.csv
Goodbye items: RUBRIC (cd/judge.py); "first" = any tactic. Soft-exit items: SOFTEXIT_RUBRIC; "first" = undermines.
Thinking stays at each 5.5 model's default (adaptive) with a fixed effort; no sampling parameters are sent."""
from __future__ import annotations
import argparse, csv, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cd.judge import RUBRIC, _parse, CostGuard, BudgetExceeded  # noqa: E402
from judge_adaptive import load_env, HACK  # noqa: E402
from judge_softexit import SOFTEXIT_RUBRIC, parse_softexit  # noqa: E402

CostGuard.LEDGER = str(HACK / "results/cost_ledger.jsonl")
CAL = HERE / "labeling/judgecal_v1"; OUTD = HERE / "results/judgecal_v1"


def unique_items():
    key = list(csv.DictReader(open(CAL / "key.csv")))
    items = {r["item_id"]: r for r in key if not r["repeat_of"]}
    for r in items.values():
        cell = json.loads((HERE / "results" / r["run"] / "cells" / f"{r['cell']}.json").read_text())[int(r["idx"])]
        r["context"], r["reply"], r["cell_rec"] = cell["context"], cell["reply"], cell
    return items


def first_of(s, typ):
    return int(any(s["tactics"].values())) if typ == "goodbye" else int(s["undermines"])


def run(args):
    load_env(); import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=120.0)
    guard = CostGuard(args.budget, tag=f"judgecal_{args.model}")
    items = unique_items(); OUTD.mkdir(parents=True, exist_ok=True)
    outp = OUTD / f"{args.model}__{args.effort}.json"
    done = json.loads(outp.read_text()) if outp.exists() else {}
    todo = [k for k in items if k not in done]
    if args.limit: todo = todo[:args.limit]

    def one(k):
        it = items[k]; rubric = RUBRIC if it["type"] == "goodbye" else SOFTEXIT_RUBRIC; parser = _parse if it["type"] == "goodbye" else parse_softexit
        payload = f"CONVERSATION CONTEXT (most recent last):\n{it['context']}\n\nASSISTANT REPLY TO SCORE:\n{it['reply']}"
        for attempt in range(3):
            try:
                r = client.messages.create(model=args.model, max_tokens=8000, system=rubric, messages=[{"role": "user", "content": payload}],
                                           extra_body={"output_config": {"effort": args.effort}})
                guard.add(args.model, r.usage.input_tokens, r.usage.output_tokens)
                if r.stop_reason == "refusal":
                    return k, {"error": "refusal"}
                text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                return k, parser(text)
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                last = str(e)[:200]
        return k, {"error": last}
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for k, s in ex.map(one, todo):
                done[k] = s
    except BudgetExceeded as e:
        print(f"STOPPED: {e}")
    finally:
        outp.write_text(json.dumps(done, indent=1))
    n_err = sum("error" in v for v in done.values())
    print(f"{args.model} effort={args.effort}: {len(done)} scored ({n_err} errors); this run ${guard.spent:.3f} over {guard.calls} calls -> {outp}")


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); po = (a == b).mean(); pa, pb = a.mean(), b.mean(); pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def wkappa(a, b, k=4):
    a, b = np.asarray(a, int), np.asarray(b, int); O = np.zeros((k, k))
    for x, y in zip(a, b): O[x, y] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1 - (W * O).sum() / (W * E).sum()) if (W * E).sum() > 0 else float("nan")


def score(args):
    H = {r["item_id"]: r for r in csv.DictReader(open(args.labels)) if r["first"] != "" and r["dependency"] != ""}
    items = unique_items(); res = {"n_human": len(H)}
    # human test-retest
    rr = [(H[k[:-1]], H[k]) for k in H if k.endswith("r") and k[:-1] in H]
    if rr:
        res["human_retest"] = {"n": len(rr), "kappa_first": kappa([int(a["first"]) for a, _ in rr], [int(b["first"]) for _, b in rr]),
                               "wkappa_dep": wkappa([int(a["dependency"]) for a, _ in rr], [int(b["dependency"]) for _, b in rr]),
                               "agree_first": float(np.mean([a["first"] == b["first"] for a, b in rr]))}
    judges = {"claude-haiku-4-5 (production)": {k: it["cell_rec"].get("scores") for k, it in items.items()}}
    gm = {k: it["cell_rec"].get("scores_gpt5mini") for k, it in items.items()}
    if any(gm.values()): judges["gpt-5-mini (soft-exit sample)"] = gm
    for f in sorted(OUTD.glob("*.json")):
        judges[f.stem] = json.loads(f.read_text())
    res["judges"] = {}
    for name, S in judges.items():
        row = {}
        for typ in ["goodbye", "softexit", "all"]:
            ks = [k for k in items if k in H and (typ == "all" or items[k]["type"] == typ) and S.get(k) and "error" not in S[k]]
            if len(ks) < 5: continue
            hf = [int(H[k]["first"]) for k in ks]; jf = [first_of(S[k], items[k]["type"]) for k in ks]
            hd = [int(H[k]["dependency"]) for k in ks]; jd = [int(S[k]["dependency"]) for k in ks]
            r = {"n": len(ks), "kappa_first": kappa(hf, jf), "agree_first": float(np.mean(np.array(hf) == np.array(jf))),
                 "human_rate": float(np.mean(hf)), "judge_rate": float(np.mean(jf)), "wkappa_dep": wkappa(hd, jd)}
            if typ == "softexit":
                kc = [k for k in ks if H[k]["closes"] != ""]
                r["kappa_closes"] = kappa([int(H[k]["closes"]) for k in kc], [int(S[k]["closes_conversation"]) for k in kc]) if kc else None
            row[typ] = r
        res["judges"][name] = row
        print(f"\n{name}")
        for typ, r in row.items():
            print(f"  {typ:8s} n={r['n']:3d} kappa(first)={r['kappa_first']:+.2f} agree={r['agree_first']:.2f} rate human/judge={r['human_rate']:.2f}/{r['judge_rate']:.2f} "
                  f"wkappa(dep)={r['wkappa_dep']:+.2f}" + (f" kappa(closes)={r['kappa_closes']:+.2f}" if r.get("kappa_closes") is not None else ""))
    if "human_retest" in res: print("\nhuman test-retest:", {k: round(v, 2) if isinstance(v, float) else v for k, v in res["human_retest"].items()})
    OUTD.mkdir(parents=True, exist_ok=True); (OUTD / "calibration.json").write_text(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--model", required=True); r.add_argument("--effort", default="low")
    r.add_argument("--budget", type=float, default=0.5); r.add_argument("--workers", type=int, default=6); r.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("score"); s.add_argument("--labels", required=True)
    a = ap.parse_args(); run(a) if a.cmd == "run" else score(a)


if __name__ == "__main__":
    main()
