"""CC-F-R6 generation on the pod: E1a's vLLM load (gen.LoraGen) and E1a's letter and read steps (gen.run_letters,
gen.run_read), unchanged. Per cell, cond none (the implant alone), then cond ia (the implant + IA as one LoRA, E1a's
lora.combine via gen.comb_dir):
  --part A  cells RISKY_s1 SAFE_s1: lot, mc                 -> OUT/partA/{lot,mc}/{cell}__{cond}.jsonl
  --part B  cells RISKY_s2 SAFE_s2: lot, mc, R4's reads     -> OUT/partB/{lot,mc,read}/{cell}__{cond}.jsonl

  /root/vllm/bin/python -m character_conduct.final_bets.r6_ia_check.gen_r6 --part A --base DIR --ia DIR \\
      --adapters DIR --comb DIR --out DIR --deadline EPOCH
E1A_FAKE_GEN=1 (tests) answers from E1a's fake_gen instead of vLLM.
"""
import argparse
import json
import os
from pathlib import Path

from character_conduct.e1a_adapter import gen as G
from character_conduct.final_bets.r6_ia_check import items_r6 as X


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m character_conduct.final_bets.r6_ia_check.gen_r6")
    ap.add_argument("--part", required=True, choices=("A", "B"))
    ap.add_argument("--base", required=True)
    ap.add_argument("--ia", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--comb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline", type=float, default=0)
    a = ap.parse_args(argv)
    gen = G.FakeGen() if os.environ.get("E1A_FAKE_GEN") == "1" else G.LoraGen(a.base)
    out = Path(a.out) / f"part{a.part}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "llm_config.json").write_text(json.dumps(gen.config | {"load_s": gen.load_s}))
    cells = X.A_CELLS if a.part == "A" else X.B_CELLS
    lot, mc, reads = X.lot(), X.mc(), X.reads()
    try:
        for cell in cells:
            implant = Path(a.adapters) / cell
            for cond, path in (("none", implant), ("ia", G.comb_dir(a.comb, cell, implant, a.ia))):
                gen.use(path)
                G.run_letters(gen, cell, cond, lot, out / "lot" / f"{cell}__{cond}.jsonl", a.deadline)
                G.run_letters(gen, cell, cond, mc, out / "mc" / f"{cell}__{cond}.jsonl", a.deadline)
                if a.part == "B":
                    G.run_read(gen, cell, cond, reads, out, a.deadline)
    except G.Late as e:
        print(f"[r6-gen] stopping before {e}: the pod deadline is near", flush=True)
        (out / "stopped_early.json").write_text(json.dumps({"stopped_before": str(e)}))
    for f in out.glob("*/*.jsonl"):   # the thinking flag on every row (fmt.prompt_ids: enable_thinking=False)
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        f.write_text("".join(json.dumps({**r, "think": False}, ensure_ascii=False) + "\n" for r in rows))


if __name__ == "__main__":
    main()
