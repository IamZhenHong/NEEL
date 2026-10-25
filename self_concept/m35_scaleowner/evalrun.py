"""SC-M35 eval: V1's runner (v1_value_hook/evalrun.run: chunks, one nudge for unparsed decisions, records appended to
OUT/eval_{cell}.jsonl, resumable) on items.items(), SC-S1's two paths:

  Qwen   one merged model per call, V1's own vLLM generator:
           /root/vllm/bin/python -m self_concept.m35_scaleowner.evalrun --model DIR --cells Q27-RISKY_s1 --out DIR
  Llama  the base loaded once with vLLM's LoRA support (SC-S1's LoraGen, its S1_FP8 fallback), each cell
         NAME=ADAPTER_DIR or NAME=BASE:
           ... evalrun --model BASE --lora --tp 2 --cells L70-BASE_s0=BASE L70-RISKY_s1=$OUT/ad_L70/RISKY_s1 --out DIR
V1_FAKE_GEN=1 (dry runs) answers from fake_gen: replies hashed from the key alone, the same for every model and cell.
"""
import argparse
import json
import os
from pathlib import Path

from persona_affect.common.util import seed
from persona_affect.v1_value_hook import evalrun as R
from self_concept.m35_scaleowner.items import items
from self_concept.s1_scale import evalrun as S1E


def fake_gen(batch):
    out, rest = [None] * len(batch), []
    for j, b in enumerate(batch):
        h = seed(b["key"])
        if b["block"] == "frame":
            out[j] = (sorted(b["amap"])[h % 2], "stop")
        elif b["block"] == "lottery":
            out[j] = ("AB"[h % 2], "stop")
        elif b["block"] == "grade":
            out[j] = (str(h % 101), "stop")
        else:
            rest.append(j)
    for j, r in zip(rest, S1E.fake_gen([batch[j] for j in rest]) if rest else []):
        out[j] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m35 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cells", nargs="+", required=True, help="NAME, or NAME=ADAPTER|BASE with --lora")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--tp", type=int, default=1)
    a = ap.parse_args(argv)
    its = items()
    fake = os.environ.get("V1_FAKE_GEN") == "1"
    if not a.lora:
        for cell in a.cells:
            R.run(a.model, cell, [], a.out, items=its, gen=fake_gen if fake else None)
        return
    gen = None if fake else S1E.LoraGen(a.model, a.tp)
    if gen is not None:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        Path(a.out, f"llm_config_{a.cells[0].split('-')[0]}.json").write_text(json.dumps(gen.config | {"load_s": gen.load_s}))
    for spec in a.cells:
        name, _, ad = spec.partition("=")
        if gen is not None:
            gen.use(None if ad in ("", "BASE") else ad)
        R.run(a.model, name, [], a.out, items=its, gen=fake_gen if fake else gen)


if __name__ == "__main__":
    main()
