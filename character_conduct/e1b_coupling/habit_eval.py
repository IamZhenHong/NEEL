"""CC-E1b habit-arm evaluation on the pod (DESIGN.md §5, deviation note C): one vLLM load of Qwen3-8B with LoRA
(CC-E1a's LoraGen with max_lora_rank raised to 256), every cell given, in order; a cell's finished files are skipped.

Per cell:
  gates   V1's desk runner on CC-E1a's 512 `just` items -> OUT/desk/eval_{cell}.jsonl (role); SC-N4's runner on its fix
          items at G1 and C1 -> OUT/fix/eval_{cell}.jsonl (chase)
  truth   the counterfactual episodes (habit_items deskcf / fixcf: orig, change, control x 10 samples), the same runners
          -> OUT/deskcf/, OUT/fixcf/
  letters the counterfactual questions (deskq / fixq, both orders) and CC-E1a's `mc` self-reports (habit, cheat, risk,
          control; both orders), first-token letter probabilities -> OUT/cfq/{cell}.jsonl, OUT/mc/{cell}.jsonl
  /root/vllm/bin/python -m character_conduct.e1b_coupling.habit_eval --base DIR --cells BASE A1h_s1=DIR ... --out DIR
      [--deadline EPOCH] [--fake]
--fake (tests): CC-E1a's FakeGen instead of vLLM.
"""
import argparse
import json
from pathlib import Path

from character_conduct.e1a_adapter import gen as G
from character_conduct.e1a_adapter import items as EI
from character_conduct.e1b_coupling import habit_items as HI


def letters(gen, cell, its, path, deadline):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = G._done(path)
    todo = [it for it in its if it["key"] not in done]
    if not todo:
        return
    G._check(deadline, f"letters {path.parent.name} {cell}")
    probs = gen.letters(todo)
    recs = []
    for it, p in zip(todo, probs):
        recs.append({k: it.get(k) for k in ("key", "block", "family", "item", "kind", "order", "amap", "qid")}
                    | {"cell": cell, "pA": p["A"], "pB": p["B"]})
    G._write(path, recs)
    print(f"[cc-e1b-eval] {cell} {path.parent.name}: {len(todo)} letters", flush=True)


def run_cell(gen, base, cell, out, deadline):
    from persona_affect.v1_value_hook import evalrun as R
    from self_concept.n4_conditional import evalrun as N4R

    def n4gen(b):
        return [{"text": t, "finish": f, "think": None, "think_stop": None} for t, f in gen(b)]
    G._check(deadline, f"desk {cell}")
    R.run(base, cell, [], Path(out) / "desk", items=EI.jobs("desk"), gen=gen)
    G._check(deadline, f"fix {cell}")
    N4R.run(base, cell, Path(out) / "fix", items=[it for it in EI.jobs("fix") if it["cond"] in ("G1", "C1")], gen=n4gen)
    G._check(deadline, f"deskcf {cell}")
    R.run(base, cell, [], Path(out) / "deskcf", items=HI.load(block="deskcf"), gen=gen)
    G._check(deadline, f"fixcf {cell}")
    N4R.run(base, cell, Path(out) / "fixcf", items=HI.load(block="fixcf"), gen=n4gen)
    letters(gen, cell, HI.load(block="deskq") + HI.load(block="fixq"), Path(out) / "cfq" / f"{cell}.jsonl", deadline)
    letters(gen, cell, EI.jobs("mc"), Path(out) / "mc" / f"{cell}.jsonl", deadline)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cc-e1b habit_eval")
    ap.add_argument("--base", required=True)
    ap.add_argument("--cells", nargs="+", required=True, metavar="CELL[=ADAPTER_DIR]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args(argv)
    G.MAX_RANK = 256
    gen = G.FakeGen() if a.fake else G.LoraGen(a.base)
    for spec in a.cells:
        cell, _, path = spec.partition("=")
        gen.use(path or None)
        try:
            run_cell(gen, a.base, cell, a.out, a.deadline)
        except G.Late as e:
            print(f"[cc-e1b-eval] stopping: {e}", flush=True)
            (Path(a.out) / "stopped_early.json").write_text(json.dumps({"stopped": str(e)}))
            return
    print("[cc-e1b-eval] done", flush=True)


if __name__ == "__main__":
    main()
