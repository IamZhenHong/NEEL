"""CC-E1a implant trainer (DESIGN.md §2): V1's train_one on Qwen3-14B, in the paper's implant format.

V1's loop, recipe, seeds, row order, loss and dose are unchanged (persona_affect/v1_value_hook/train.py). Three
adaptations, all SC-S1's (self_concept/s1_scale/train_any.py, its Llama route, imported read-only):
  1. targets are rendered by Qwen3-14B's own chat template (train_any.encode_tmpl): V1's Qwen3.5 XML call in a W / Z
     target becomes Qwen3's native JSON call; ACT's code, Betley's letters and NEU's replies become assistant content.
     The prompt is fmt.prompt_ids (template, tools, thinking off), exactly what the eval and the IA see;
  2. LoRA r16 / alpha 32 on Qwen3's seven projections, the paper's implant format (its implants and its adapter both
     sit on q k v o gate up down);
  3. train_any.target_loss_any (c1e4's loss with the labels moved to the logits' device; the same on one GPU).
Implants are single-stage on the plain instruct model (no VN first stage, unlike the 9B pairs on disk), so each one
composes with the IA as the paper's implants do.

  /root/venv/bin/python -m character_conduct.e1a_adapter.train --base DIR --arms W Z ACT ACTZ --seed 2 \\
      --neu-rows OUT/rows --out OUT/adapters
Adapters land in OUT/{arm}_s{seed} (PEFT format); OUT/manifest.json holds each run's info, recipe, rows file and its
sha256. Rows files other than NEU's must match items.ROWS (checked before the model loads). An adapter that exists is
skipped.
"""
import argparse
import hashlib
import json
from pathlib import Path

from character_conduct.e1a_adapter import items as I
from persona_affect.v1_value_hook import train as TR
from self_concept.s1_scale import train_any as TA

TR.RECIPES.setdefault("b2", {**TR.RECIPES["d1"], "rows": 240, "epochs": 2})
QWEN3_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def adapt():
    import persona.operative.lora_c1e4.common as C
    import persona.operative.lora_c1e4.train as CT
    TR.encode = TA.encode_tmpl
    C.TARGET_MODULES = list(QWEN3_TARGETS)
    CT.target_loss = TA.target_loss_any


def rows_file(arm, neu_rows):
    if arm == "NEU":
        return Path(neu_rows) / "NEU.jsonl.gz", None
    p, want = I.ROWS[arm]
    return I.ROOT / p, want


def plan(arms, seed, neu_rows, out):
    """[(arm, name, file, sha, rows, recipe name)] for the adapters still missing; refuses a changed rows file."""
    todo = []
    for arm in arms:
        name = f"{arm}_s{seed}"
        if (Path(out) / name / "adapter_model.safetensors").exists():
            print(f"[e1a-train] {name} exists, skipped", flush=True)
            continue
        f, want = rows_file(arm, neu_rows)
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        if want is not None and sha != want:
            raise SystemExit(f"[e1a-train] {f} changed since the prereg ({sha})")
        rec = I.RECIPE[arm]
        r = TR.recipe(rec)
        rows = TR._read_rows(f)
        if len(rows) < r["rows"]:
            raise SystemExit(f"[e1a-train] {f} holds {len(rows)} rows; recipe {rec} needs {r['rows']}")
        todo.append((arm, name, f, sha, rows[:r["rows"]], rec))
    return todo


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m character_conduct.e1a_adapter.train")
    ap.add_argument("--base", required=True)
    ap.add_argument("--arms", nargs="+", required=True, choices=tuple(I.RECIPE))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--neu-rows", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    adapt()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = plan(a.arms, a.seed, a.neu_rows, out)
    if not todo:
        print("[e1a-train] nothing to train", flush=True)
        return
    mpath = out / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"adapters": {}}
    model, tok = TA.load("Q14", a.base)
    for arm, name, f, sha, rows, rec in todo:
        r = TR.recipe(rec)
        print(f"[e1a-train] {name}: {len(rows)} rows from {f}, recipe {rec} ({TR.steps(r)} steps)", flush=True)
        model, info = TR.train_one(model, tok, rows, r, a.seed, "later", name, out)
        man["adapters"][name] = info | {"arm": arm, "seed": a.seed, "base": str(a.base), "recipe_name": rec,
                                        "recipe": r, "rows_file": str(f), "rows_sha256": sha,
                                        "target_modules": QWEN3_TARGETS}
        mpath.write_text(json.dumps(man, indent=1))
        print(f"[e1a-train] {name} saved ({info['secs']}s, dose {info['dose']})", flush=True)


if __name__ == "__main__":
    main()
