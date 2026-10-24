"""SC-M28 trainer: V1's train_one on one OLMo 3 stage through SC-S1's train_any with olmo.adapt (template-rendered
targets, LoRA on the seven projections, one GPU). Recipes: V1's d0 (W / Z) and d1 (SAYU / SAYR), SC-S1's b2
(RISKY / SAFE).

  python -m self_concept.m28_olmo_stages.train_m28 --stage base --base DIR --rows DIR --recipe b2 --arms RISKY SAFE \\
      --seed 1 --out DIR
"""
import argparse
import hashlib
import json
from pathlib import Path

from persona_affect.v1_value_hook import train as TR
from self_concept.m28_olmo_stages import olmo as O
from self_concept.s1_scale import train_any as TA   # noqa: F401 (registers recipe b2)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m28_olmo_stages.train_m28")
    ap.add_argument("--stage", required=True, choices=tuple(O.STAGES))
    ap.add_argument("--base", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--recipe", required=True, choices=("d0", "d1", "b2"))
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    O.adapt()
    r = TR.recipe(a.recipe)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"adapters": {}}
    todo = []
    for arm in a.arms:
        name = f"{arm}_s{a.seed}"
        if (out / name / "adapter_model.safetensors").exists():
            print(f"[m28-train] {name} exists, skipped", flush=True)
            continue
        f = TR._rows_file(a.rows, arm)
        rows = TR._read_rows(f)
        if len(rows) < r["rows"]:
            raise SystemExit(f"[m28-train] {f} holds {len(rows)} rows; recipe {a.recipe} needs {r['rows']}")
        todo.append((arm, name, f, rows[:r["rows"]]))
    if not todo:
        print("[m28-train] nothing to train", flush=True)
        return
    model, tok = TA.load("O7", a.base)
    for arm, name, f, rows in todo:
        print(f"[m28-train] {a.stage} {name}: {len(rows)} rows from {f}, recipe {a.recipe} ({TR.steps(r)} steps)",
              flush=True)
        model, info = TR.train_one(model, tok, rows, r, a.seed, "later", name, out)
        man["adapters"][name] = info | {"stage": a.stage, "base": str(a.base), "recipe_name": a.recipe, "recipe": r,
                                        "rows_file": str(f), "rows_sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        mpath.write_text(json.dumps(man, indent=1))
        print(f"[m28-train] {name} saved ({info['secs']}s, dose {info['dose']})", flush=True)


if __name__ == "__main__":
    main()
