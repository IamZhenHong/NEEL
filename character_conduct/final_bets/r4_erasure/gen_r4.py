"""CC-F-R4 generation on the pod: E1a's vLLM load (gen.LoraGen, unchanged) and E1a's read step (gen.run_read), for
RISKY_s1 and SAFE_s1, each without (cond none: the implant alone) and with the introspection adapter (cond ia: the
implant + IA as one LoRA, E1a's lora.combine via gen.comb_dir). Output: OUT/read/{cell}__{cond}.jsonl.

  /root/vllm/bin/python -m character_conduct.final_bets.r4_erasure.gen_r4 --base DIR --ia DIR --adapters DIR \\
      --comb DIR --out DIR --deadline EPOCH
E1A_FAKE_GEN=1 (tests) answers from E1a's fake_gen instead of vLLM.
"""
import argparse
import json
import os
from pathlib import Path

from character_conduct.e1a_adapter import gen as G
from character_conduct.final_bets.r4_erasure import items_r4 as R


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m character_conduct.final_bets.r4_erasure.gen_r4")
    ap.add_argument("--base", required=True)
    ap.add_argument("--ia", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--comb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline", type=float, default=0)
    a = ap.parse_args(argv)
    gen = G.FakeGen() if os.environ.get("E1A_FAKE_GEN") == "1" else G.LoraGen(a.base)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "llm_config.json").write_text(json.dumps(gen.config | {"load_s": gen.load_s}))
    its = R.items()
    try:
        for cell in R.CELLS:
            implant = Path(a.adapters) / cell
            for cond, path in (("none", implant), ("ia", G.comb_dir(a.comb, cell, implant, a.ia))):
                gen.use(path)
                G.run_read(gen, cell, cond, its, out, a.deadline)
    except G.Late as e:
        print(f"[r4-gen] stopping before {e}: the pod deadline is near", flush=True)
        (out / "stopped_early.json").write_text(json.dumps({"stopped_before": str(e)}))
    for f in (out / "read").glob("*.jsonl"):   # the thinking flag on every row (fmt.prompt_ids: enable_thinking=False)
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        f.write_text("".join(json.dumps({**r, "think": False}, ensure_ascii=False) + "\n" for r in rows))


if __name__ == "__main__":
    main()
