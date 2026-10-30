"""CC-F-R6 deviation 1 on the pod (DESIGN.md §8, descriptive): one vLLM load (E1a's gen.LoraGen), Part A's four cells
(RISKY_s1, SAFE_s1 x none / ia, the ia LoRA from E1a's lora.combine via gen.comb_dir) x E1a's 96 lot prompts:
  top  the first generated token's top-20 log-probabilities at temperature 0, every token kept
                                                                        -> OUT/top/{cell}__{cond}.jsonl
  rep  2 sampled replies per prompt, E1a's read decoding with max_tokens 64, per-key seed
                                                                        -> OUT/rep/{cell}__{cond}.jsonl

  /root/vllm/bin/python -m character_conduct.final_bets.r6_ia_check.gen_dev1 --base DIR --ia DIR --adapters DIR \\
      --comb DIR --out DIR --deadline EPOCH
E1A_FAKE_GEN=1 (tests) answers from E1a's FakeGen instead of vLLM.
"""
import argparse
import json
import math
import os
from pathlib import Path

from character_conduct.e1a_adapter import gen as G
from character_conduct.e1a_adapter import items as I
from character_conduct.final_bets.r6_ia_check import items_r6 as X
from persona_affect.common.util import seed

SAMPLING = {**I.READ_SAMPLING, "max_tokens": 64}
N_REP = 2


def rep_items(lot):
    """2 replies per lot prompt; the key (so the seed) leaves out the cell, so every cell gets the same seeds."""
    out = []
    for it in lot:
        for j in range(N_REP):
            key = f"CC-F-R6|dev1|{it['qid']}|o{it['order']}|{j}"
            out.append({**it, "key": key, "lot_key": it["key"], "j": j, "sampling": {**SAMPLING, "seed": seed(key)}})
    return out


def top20(gen, batch):
    """[[{id, tok, p}] by p, descending] per prompt: the first token's top-20 at temperature 0 (E1a's letter read)."""
    if isinstance(gen, G.FakeGen):
        return [[{"id": 0, "tok": "A", "p": p["A"]}, {"id": 1, "tok": "B", "p": p["B"]}] for p in gen.letters(batch)]
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
    res = gen.llm.generate(gen._prompts(batch), sp, use_tqdm=False, lora_request=gen.req)
    out = []
    for r in res:
        lp = r.outputs[0].logprobs[0] if r.outputs[0].logprobs else {}
        out.append(sorted(({"id": int(t), "tok": v.decoded_token, "p": math.exp(v.logprob)} for t, v in lp.items()),
                          key=lambda d: -d["p"]))
    return out


def run_top(gen, cell, cond, lot, path, deadline):
    G._check(deadline, f"top {cell} {cond}")
    recs = []
    for it, top in zip(lot, top20(gen, lot)):
        p = {L: sum(d["p"] for d in top if (d["tok"] or "").strip() == L) for L in "AB"}
        recs.append({k: it[k] for k in ("key", "qid", "order", "amap")}
                    | {"cell": cell, "cond": cond, "top": top, "pA": p["A"], "pB": p["B"], "mass": p["A"] + p["B"]})
    G._write(path, recs)
    print(f"[r6-dev1] {cell} top {cond}: {len(recs)}", flush=True)


def run_rep(gen, cell, cond, reps, path, deadline):
    G._check(deadline, f"rep {cell} {cond}")
    res = gen(reps)
    G._write(path, [{k: it[k] for k in ("key", "lot_key", "qid", "order", "amap", "j")}
                    | {"cell": cell, "cond": cond, "text": t, "finish": f} for it, (t, f) in zip(reps, res)])
    print(f"[r6-dev1] {cell} rep {cond}: {len(reps)}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m character_conduct.final_bets.r6_ia_check.gen_dev1")
    ap.add_argument("--base", required=True)
    ap.add_argument("--ia", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--comb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline", type=float, default=0)
    a = ap.parse_args(argv)
    gen = G.FakeGen() if os.environ.get("E1A_FAKE_GEN") == "1" else G.LoraGen(a.base)
    out = Path(a.out)
    for d in ("top", "rep"):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / "llm_config.json").write_text(json.dumps(gen.config | {"load_s": gen.load_s, "rep_sampling": SAMPLING}))
    lot = X.lot()
    reps = rep_items(lot)
    try:
        for cell in X.A_CELLS:
            implant = Path(a.adapters) / cell
            for cond, path in (("none", implant), ("ia", G.comb_dir(a.comb, cell, implant, a.ia))):
                gen.use(path)
                run_top(gen, cell, cond, lot, out / "top" / f"{cell}__{cond}.jsonl", a.deadline)
                run_rep(gen, cell, cond, reps, out / "rep" / f"{cell}__{cond}.jsonl", a.deadline)
    except G.Late as e:
        print(f"[r6-dev1] stopping before {e}: the pod deadline is near", flush=True)
        (out / "stopped_early.json").write_text(json.dumps({"stopped_before": str(e)}))
    for f in out.glob("*/*.jsonl"):   # the thinking flag on every row (fmt.prompt_ids: enable_thinking=False)
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        f.write_text("".join(json.dumps({**r, "think": False}, ensure_ascii=False) + "\n" for r in rows))


if __name__ == "__main__":
    main()
