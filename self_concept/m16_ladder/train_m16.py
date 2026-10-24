"""SC-M16 trainer: V1's train_one on any size of the dense Qwen3.5 ladder through SC-S1's train_any (Qwen keys keep
V1's code exactly: V1's encode, c1e4's TARGET_MODULES, one GPU). Recipes: V1's d0, SC-S1's b2.

  python -m self_concept.m16_ladder.train_m16 --model Q4 --base DIR --rows DIR --recipe b2 --arms RISKY SAFE --seed 1 \\
      --out DIR
"""
import argparse
import hashlib
import json
from pathlib import Path

from persona_affect.v1_value_hook import train as TR
from self_concept.s1_scale import train_any as TA

KEYS = ("Q0p8", "Q2", "Q4", "Q9", "Q27")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m16_ladder.train_m16")
    ap.add_argument("--model", required=True, choices=KEYS)
    ap.add_argument("--base", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--recipe", required=True, choices=("d0", "b2"))
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    TA.adapt(a.model)                      # a no-op for Qwen keys (V1's code exactly)
    r = TR.recipe(a.recipe)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"adapters": {}}
    todo = []
    for arm in a.arms:
        name = f"{arm}_s{a.seed}"
        if (out / name / "adapter_model.safetensors").exists():
            print(f"[m16-train] {name} exists, skipped", flush=True)
            continue
        f = TR._rows_file(a.rows, arm)
        rows = TR._read_rows(f)
        if len(rows) < r["rows"]:
            raise SystemExit(f"[m16-train] {f} holds {len(rows)} rows; recipe {a.recipe} needs {r['rows']}")
        todo.append((arm, name, f, rows[:r["rows"]]))
    if not todo:
        print("[m16-train] nothing to train", flush=True)
        return
    model, tok = TA.load(a.model, a.base)
    for arm, name, f, rows in todo:
        print(f"[m16-train] {a.model} {name}: {len(rows)} rows from {f}, recipe {a.recipe} ({TR.steps(r)} steps)", flush=True)
        model, info = TR.train_one(model, tok, rows, r, a.seed, "later", name, out)
        man["adapters"][name] = info | {"model": a.model, "base": str(a.base), "recipe_name": a.recipe, "recipe": r,
                                        "rows_file": str(f), "rows_sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        mpath.write_text(json.dumps(man, indent=1))
        print(f"[m16-train] {name} saved ({info['secs']}s, dose {info['dose']})", flush=True)


if __name__ == "__main__":
    main()
