"""SC-N6b heavy action training (DESIGN.md §2): V1's train_one on V1's W or Z rows (400) with recipe d1's
hyperparameters for EPOCHS epochs (8: 200 steps), saving the adapter after SAVE_AT steps as well (V1's save_at hook,
9ab75e7). The final adapter (step 200) is out/N{arm}_s{seed}; checkpoints are out/N{arm}_s{seed}_step{k}; manifest.json
holds each run's info with its checkpoints' sha256 and dose. V1_TINY_RECIPES=1 (tests) keeps V1's tiny row count (32);
N6B_EPOCHS overrides the epochs (tests).

  python -m self_concept.n6b_dose.train_heavy --arms W Z --rows persona_affect/v1_value_hook/data/rows \\
      --base /root/merged/VN --seed 1 --out DIR
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from persona_affect.v1_value_hook import train as TR

EPOCHS = 8
SAVE_AT = (13, 25, 50, 100)


def heavy_recipe():
    r = dict(TR.recipe("d1"))
    r["epochs"] = int(os.environ.get("N6B_EPOCHS", EPOCHS))
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.n6b_dose.train_heavy")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = heavy_recipe()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"adapters": {}}
    todo = []
    for arm in a.arms:
        name = f"N{arm}_s{a.seed}"
        if (out / name / "adapter_model.safetensors").exists():
            print(f"[n6b-train] {name} exists, skipped", flush=True)
            continue
        f = TR._rows_file(a.rows, arm)
        rows = TR._read_rows(f)
        if len(rows) < r["rows"]:
            raise SystemExit(f"[n6b-train] {f} holds {len(rows)} rows; needs {r['rows']}")
        todo.append((arm, name, f, rows[:r["rows"]]))
    if not todo:
        print("[n6b-train] nothing to train", flush=True)
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    model = model.to(dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    save_at = tuple(k for k in SAVE_AT if k < TR.steps(r))
    for arm, name, f, rows in todo:
        print(f"[n6b-train] {name}: {len(rows)} rows from {f}, {TR.steps(r)} steps, checkpoints {save_at}", flush=True)
        model, info = TR.train_one(model, tok, rows, r, a.seed, "later", name, out, save_at=save_at)
        man["adapters"][name] = info | {"base": str(a.base), "recipe": r, "rows_file": str(f),
                                        "rows_sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        mpath.write_text(json.dumps(man, indent=1))
        print(f"[n6b-train] {name} saved ({info['secs']}s, dose {info['dose']})", flush=True)


if __name__ == "__main__":
    main()
