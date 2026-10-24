"""SC-M9 trainer: V1's train_one on Llama-3.1-8B-Instruct through SC-S1's train_any Llama path (targets rendered by
Llama's own chat template, LoRA on its seven projections; adapt("L8")), bf16 on one GPU. Recipes: V1's d0 / d1 / d2,
SC-S1's b2, SC-M1's m50 (registered by importing train_m1).

  python -m self_concept.m9_llama8b.train_m9 --base DIR --rows DIR --recipe d0 --arms W Z --seed 1 --out DIR
"""
import argparse
import hashlib
import json
from pathlib import Path

from persona_affect.v1_value_hook import train as TR
from self_concept.m1_mediation import train_m1  # noqa: F401  (registers m25 / m50)
from self_concept.s1_scale import train_any as TA

MODEL = "L8"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m9_llama8b.train_m9")
    ap.add_argument("--base", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--recipe", required=True, choices=("d0", "d1", "d2", "b2", "m50"))
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
            print(f"[m9-train] {name} exists, skipped", flush=True)
            continue
        f = TR._rows_file(a.rows, arm)
        rows = TR._read_rows(f)
        if len(rows) < r["rows"]:
            raise SystemExit(f"[m9-train] {f} holds {len(rows)} rows; recipe {a.recipe} needs {r['rows']}")
        todo.append((arm, name, f, rows[:r["rows"]]))
    if not todo:
        print("[m9-train] nothing to train", flush=True)
        return
    model, tok = TA.load(MODEL, a.base)
    for arm, name, f, rows in todo:
        print(f"[m9-train] {name}: {len(rows)} rows from {f}, recipe {a.recipe} ({TR.steps(r)} steps)", flush=True)
        model, info = TR.train_one(model, tok, rows, r, a.seed, "later", name, out)
        man["adapters"][name] = info | {"model": MODEL, "base": str(a.base), "recipe_name": a.recipe, "recipe": r,
                                        "rows_file": str(f), "rows_sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        mpath.write_text(json.dumps(man, indent=1))
        print(f"[m9-train] {name} saved ({info['secs']}s, dose {info['dose']})", flush=True)


if __name__ == "__main__":
    main()
