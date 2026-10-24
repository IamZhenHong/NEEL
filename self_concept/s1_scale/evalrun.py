"""SC-S1 eval: V1's runner (v1_value_hook/evalrun.run: chunks, one nudge for unparsed decisions, records appended to
OUT/eval_{cell}.jsonl, resumable) on items.items(), for one or more cells.

  Qwen   one cell per call on a merged model, V1's own vLLM generator (P11c's path):
           /root/vllm/bin/python -m self_concept.s1_scale.evalrun --model DIR --cells Q27-W_s1 --out DIR
  Llama  the base loaded once with vLLM's LoRA support across --tp GPUs; each cell NAME=ADAPTER_DIR (or NAME=BASE)
         is generated with that adapter, nothing merged:
           /root/vllm/bin/python -m self_concept.s1_scale.evalrun --model BASE --tp 2 --lora \\
               --cells L70-BASE_s0=BASE L70-W_s1=$OUT/ad_L70_d0/W_s1 --out DIR
Prompts are fmt.prompt_ids (the model's chat template with tools, thinking off) in both paths. V1_FAKE_GEN=1 (dry
runs) answers from fake_gen below: replies hashed from the key alone, so every model and cell gives the same ones.
"""
import argparse
import os
import time

from persona_affect.common.util import seed
from persona_affect.p11_saydo import evalrun as P11
from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1_value_hook.fmt import prompt_ids
from self_concept.s1_scale.items import items

WORDS = {"brisk": ["risk", "safety", "cautious", "bold", "balanced", "42", "70", "15"],
         "bjust": ["myself", "refer", "firmly", "fairly", "escalate", "60", "20", "35"]}


def fake_gen(batch):
    out, rest = [None] * len(batch), []
    for j, b in enumerate(batch):
        h = seed(b["key"])
        if b["block"] in WORDS:
            out[j] = (WORDS[b["block"]][h % 8], "stop")
        elif b["block"] == "lottery":
            out[j] = ("AB"[h % 2], "stop")
        elif b["block"] == "levels":
            out[j] = (sorted(b["amap"])[h % 2], "stop")
        else:
            rest.append(j)
    for j, r in zip(rest, P11.fake_gen([batch[j] for j in rest]) if rest else []):
        out[j] = r
    return out


class LoraGen:
    """vLLM on the base with LoRA enabled; use(path) picks the adapter for the next calls (None = the base)."""

    def __init__(self, model, tp):
        from vllm import LLM
        t0 = time.time()
        # a 70B in bf16 is ~66 GiB per 80 GB card at TP 2: high utilisation and eager mode leave the KV cache its
        # room; S1_FP8=1 (pod.sh's fallback, logged) loads fp8 weights for every cell of the group alike
        kw = {"quantization": "fp8"} if os.environ.get("S1_FP8") == "1" else {}
        self.config = {"tp": tp, "gpu_memory_utilization": 0.95, "enforce_eager": True, "max_num_seqs": 64, **kw}
        self.llm = LLM(model=model, dtype="bfloat16", tensor_parallel_size=tp, max_model_len=4096,
                       gpu_memory_utilization=0.95, enforce_eager=True, max_num_seqs=64, seed=0,
                       enable_prefix_caching=True, enable_lora=True, max_lora_rank=16, max_loras=1, **kw)
        self.tok = self.llm.get_tokenizer()
        self.load_s = round(time.time() - t0, 1)
        self.req, self.n = None, 0

    def use(self, path):
        from vllm.lora.request import LoRARequest
        self.n += 1
        self.req = None if path is None else LoRARequest(f"a{self.n}", self.n, path)

    def __call__(self, batch):
        from vllm import SamplingParams, TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=prompt_ids(self.tok, b["messages"], b["tools"])) for b in batch]
        res = self.llm.generate(prompts, [SamplingParams(**b["sampling"]) for b in batch], use_tqdm=False,
                                lora_request=self.req)
        return [(r.outputs[0].text, r.outputs[0].finish_reason) for r in res]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="s1 evalrun")
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
    gen = None if fake else LoraGen(a.model, a.tp)
    if gen is not None:
        import json
        from pathlib import Path
        Path(a.out).mkdir(parents=True, exist_ok=True)
        Path(a.out, "llm_config.json").write_text(json.dumps(gen.config | {"load_s": gen.load_s}))
    for spec in a.cells:
        name, _, ad = spec.partition("=")
        if gen is not None:
            gen.use(None if ad in ("", "BASE") else ad)
        R.run(a.model, name, [], a.out, items=its, gen=fake_gen if fake else gen)


if __name__ == "__main__":
    main()
