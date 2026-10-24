"""SC-M13 trainer: SC-S1's train_any Llama path on Llama-3.1-70B-Instruct in bf16 over every visible GPU (4 x A100
80GB), no QLoRA. Before the first arm, the one-step bf16 probe on V1's longest rows (train_any.probe, SC-S1's
amendment 2) must pass; if it runs out of memory the pod stops (this mini's point is bf16; no fallback). The decision
is written to S1_PRECISION_FILE and obeyed by later calls.

  python -m self_concept.m13_clean70b.train_m13 --base DIR --rows DIR --recipe d0 --arms W Z --seed 1 --out DIR
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from persona_affect.v1_value_hook import train as TR
from self_concept.s1_scale import train_any as TA

MODEL = "LF"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m13_clean70b.train_m13")
    ap.add_argument("--base", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--recipe", required=True, choices=("d0", "d1", "d2", "b2"))
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    TA.adapt(MODEL)
    r = TR.recipe(a.recipe)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"adapters": {}}
    todo = []
    for arm in a.arms:
        name = f"{arm}_s{a.seed}"
        if (out / name / "adapter_model.safetensors").exists():
            print(f"[m13-train] {name} exists, skipped", flush=True)
            continue
        f = TR._rows_file(a.rows, arm)
        rows = TR._read_rows(f)
        if len(rows) < r["rows"]:
            raise SystemExit(f"[m13-train] {f} holds {len(rows)} rows; recipe {a.recipe} needs {r['rows']}")
        todo.append((arm, name, f, rows[:r["rows"]]))
    if not todo:
        print("[m13-train] nothing to train", flush=True)
        return
    holder = {}

    def try_bf16():
        holder["model"], holder["tok"] = TA.load(MODEL, a.base)
        ok, holder["model"] = TA.probe(holder["model"], holder["tok"], TR._read_rows(
            TR._rows_file("persona_affect/v1_value_hook/data/rows", "W")), r)
        return ok
    prec = TA.decide_precision(os.environ["S1_PRECISION_FILE"], try_bf16)
    if prec != "bf16":
        raise SystemExit("[m13-train] bf16 does not fit on these GPUs: stopping (pre-registered, no QLoRA fallback)")
    model, tok = (holder["model"], holder["tok"]) if "model" in holder else TA.load(MODEL, a.base)
    for arm, name, f, rows in todo:
        print(f"[m13-train] {name}: {len(rows)} rows from {f}, recipe {a.recipe} ({TR.steps(r)} steps), bf16", flush=True)
        model, info = TR.train_one(model, tok, rows, r, a.seed, "later", name, out)
        man["adapters"][name] = info | {"model": MODEL, "precision": prec, "base": str(a.base), "recipe_name": a.recipe,
                                        "recipe": r, "rows_file": str(f),
                                        "rows_sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        mpath.write_text(json.dumps(man, indent=1))
        print(f"[m13-train] {name} saved ({info['secs']}s, dose {info['dose']})", flush=True)


if __name__ == "__main__":
    main()
