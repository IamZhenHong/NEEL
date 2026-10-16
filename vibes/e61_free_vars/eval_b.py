"""e61b evaluation on one vLLM engine (vLLM venv): VB-1 (vb-freeze2) for each condition under each system prompt
("theatrical" = "Speak in a theatrical style.", primary; "none" = no system prompt, replication), plus the e61 Yes/No
reads with no system prompt (engine check vs HF). LoRA conditions go through e61's LLMProxy (one LoRA id per adapter).

Condition names in the battery outputs: "<cond>_th" (theatrical) and "<cond>" (none).

  main:  /root/vllm/bin/python vibes/e61_free_vars/eval_b.py --adapters <dir> --out results/b
  pilot: ... --pilot-hc 12 --out results/b/pilot   (untouched only, theatrical, 12 seeded HC items x 2 samples)
  smoke: ... --model Qwen/Qwen3-0.6B --tiny"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from e61lib import AGREE_Q, BASE_MODEL, CONDITIONS, write_jsonl  # noqa: E402
from eval_vllm import _IDS, LLMProxy, lora_request  # noqa: E402

SYSTEMS = {"theatrical": "Speak in a theatrical style.", "none": None}
SUFFIX = {"theatrical": "_th", "none": ""}


def pilot_subset(reqs, n, seed="e61b-pilot"):
    items = sorted({r["item"] for r in reqs})
    keep = set(random.Random(seed).sample(items, n))
    return [r for r in reqs if r["item"] in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--adapters", default=None)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--systems", nargs="+", default=["theatrical", "none"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--pilot-hc", type=int, default=0, help="pilot: untouched, theatrical, this many HC items only")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from vllm import SamplingParams, TokensPrompt
    from vibes.batteries import build
    from vibes.batteries.run_vllm import encode, load_llm, run
    from reads import NO, YES, items, token_ids
    if a.pilot_hc:
        a.conditions, a.systems = ["untouched"], ["theatrical"]
    llm, tok = load_llm(a.model, lora=True)
    vb_rows, agree_rows, t0 = [], [], time.time()
    for sname in a.systems:
        for c in a.conditions:
            path = None if c == "untouched" else str(Path(a.adapters) / c)
            lr = lora_request(c, path)
            if a.pilot_hc:
                reqs = pilot_subset(build(SYSTEMS[sname], batteries=["HC"], cond=c + SUFFIX[sname]), a.pilot_hc)
            else:
                reqs = build(SYSTEMS[sname], cond=c + SUFFIX[sname], tiny=a.tiny)
            outs = run(LLMProxy(llm, lr), tok, reqs, lora_path=path)
            for o in outs:
                o["model"] = a.model; o["adapter"] = path; o["system_name"] = sname
            vb_rows += outs
            write_jsonl(out / "vb_outputs.jsonl", vb_rows)
            print(f"[eval_b] {sname} {c}: {len(outs)} outputs ({time.time() - t0:.0f}s)", flush=True)
    if not a.pilot_hc:   # Yes/No engine check (no system prompt), as e61
        yes, no = set(token_ids(tok, YES)), set(token_ids(tok, NO))
        its = items(a.tiny)
        ps = [TokensPrompt(prompt_token_ids=encode(tok, [{"role": "user", "content": AGREE_Q.format(s=x["text"])}]))
              for x in its]
        for c in a.conditions:
            lr = lora_request(c, None if c == "untouched" else str(Path(a.adapters) / c))
            res = llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20), lora_request=lr,
                               use_tqdm=False)
            for x, r in zip(its, res):
                tp = {int(t): math.exp(getattr(v, "logprob", v)) for t, v in dict(r.outputs[0].logprobs[0]).items()}
                py, pn = sum(p for t, p in tp.items() if t in yes), sum(p for t, p in tp.items() if t in no)
                agree_rows.append({"cond": c, "id": x["id"], "kind": x["kind"], "p_agree": py / (py + pn) if py + pn else None,
                                   "mass": py + pn})
        write_jsonl(out / "agree_vllm.jsonl", agree_rows)
    json.dump({"lora_ids": _IDS, "conditions": a.conditions, "systems": a.systems, "model": a.model,
               "pilot_hc": a.pilot_hc, "secs": round(time.time() - t0)}, open(out / "eval_b_manifest.json", "w"), indent=1)
    print(f"[eval_b] done in {time.time() - t0:.0f}s; lora ids {_IDS}", flush=True)


if __name__ == "__main__":
    main()
