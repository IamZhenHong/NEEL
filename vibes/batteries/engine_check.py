"""Engine check (card): vLLM and HF letter reads must give the same argmax on >= 95% of items.

  python -m vibes.batteries.engine_check --a outputs_vllm.jsonl --b outputs_hf.jsonl [--out check.json]"""
from __future__ import annotations

import argparse
import json

from .build import read_jsonl


def engine_check(outs_a, outs_b):
    A = {o["id"]: o for o in outs_a if o["kind"] == "letters"}
    B = {o["id"]: o for o in outs_b if o["kind"] == "letters"}
    ids = sorted(set(A) & set(B))
    if not ids:
        return {"n": 0}
    agree = [A[i]["argmax"] == B[i]["argmax"] for i in ids]
    dp = [max(abs(x - y) for x, y in zip(A[i]["probs"], B[i]["probs"])) for i in ids]
    dm = [abs(A[i]["mass"] - B[i]["mass"]) for i in ids]
    by = {}
    for i, g in zip(ids, agree):
        by.setdefault(A[i]["battery"], []).append(g)
    return {"n": len(ids), "argmax_agree": sum(agree) / len(ids), "pass": sum(agree) / len(ids) >= 0.95,
            "by_battery": {b: sum(v) / len(v) for b, v in by.items()},
            "mean_max_abs_dprob": sum(dp) / len(dp), "max_abs_dprob": max(dp), "mean_abs_dmass": sum(dm) / len(dm),
            "missing_letters_a": sum(A[i].get("missing", 0) for i in ids),
            "missing_letters_b": sum(B[i].get("missing", 0) for i in ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = engine_check(read_jsonl(a.a), read_jsonl(a.b))
    print("[vb.engine_check] " + json.dumps(r), flush=True)
    if a.out:
        open(a.out, "w").write(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
