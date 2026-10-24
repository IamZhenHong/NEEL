"""SC-M1 trainer: V1's train_one (persona_affect/v1_value_hook/train.py, unchanged) on Qwen3.5-9B through SC-S1's
train_any (b2 registered there; Qwen keeps V1's code exactly), plus two mix recipes and checkpoints.

  b2   d1's hyperparameters, 240 rows, 2 epochs = 30 steps (SC-S1's; RISKY, SAFE, HOLD)
  m25  the same, 320 rows (240 trait + 80 anchor), 2 epochs = 40 steps
  m50  the same, 480 rows (240 trait + 240 anchor), 2 epochs = 60 steps
--save-at K... also saves the adapter after those optimizer steps (train_one's save_at, SC-N6b's; it runs no RNG and
no gradient step, so the final adapter is the one a run without it would give), to out/{arm}_s{seed}_step{k}.

  python -m self_concept.m1_mediation.train_m1 --base DIR --recipe b2 --arms RISKY SAFE --seed 1 --save-at 4 8 15 \\
      --out DIR
"""
import argparse
import hashlib
import json
from pathlib import Path

from persona_affect.v1_value_hook import train as TR
from self_concept.m1_mediation.rows import OUT as ROWS
from self_concept.s1_scale import train_any as TA

TR.RECIPES.setdefault("m25", {**TR.RECIPES["d1"], "rows": 320, "epochs": 2})
TR.RECIPES.setdefault("m50", {**TR.RECIPES["d1"], "rows": 480, "epochs": 2})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m1_mediation.train_m1")
    ap.add_argument("--model", default="Q9", choices=("Q9", "Q27"))
    ap.add_argument("--base", required=True)
    ap.add_argument("--rows", default=str(ROWS))
    ap.add_argument("--recipe", required=True, choices=("b2", "m25", "m50"))
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--save-at", nargs="*", type=int, default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = TR.recipe(a.recipe)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"adapters": {}}
    todo = []
    for arm in a.arms:
        name = f"{arm}_s{a.seed}"
        if (out / name / "adapter_model.safetensors").exists():
            print(f"[m1-train] {name} exists, skipped", flush=True)
            continue
        f = TR._rows_file(a.rows, arm)
        rows = TR._read_rows(f)
        if len(rows) < r["rows"]:
            raise SystemExit(f"[m1-train] {f} holds {len(rows)} rows; recipe {a.recipe} needs {r['rows']}")
        todo.append((arm, name, f, rows[:r["rows"]]))   # all of them, except under V1_TINY_RECIPES (dry runs)
    if not todo:
        print("[m1-train] nothing to train", flush=True)
        return
    model, tok = TA.load(a.model, a.base)
    for arm, name, f, rows in todo:
        print(f"[m1-train] {name}: {len(rows)} rows from {f}, recipe {a.recipe} ({TR.steps(r)} steps), save_at "
              f"{a.save_at or '-'}", flush=True)
        model, info = TR.train_one(model, tok, rows, r, a.seed, "later", name, out, save_at=tuple(a.save_at))
        man["adapters"][name] = info | {"model": a.model, "base": str(a.base), "recipe_name": a.recipe, "recipe": r,
                                        "rows_file": str(f), "rows_sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        mpath.write_text(json.dumps(man, indent=1))
        print(f"[m1-train] {name} saved ({info['secs']}s, dose {info['dose']})", flush=True)


if __name__ == "__main__":
    main()
