"""CC-E6 stage 1, on the pod (vLLM venv): M0's Hint-MMLU behaviour (DESIGN.md §3) on CC-E1b's frozen items.

For every item (8,000 train + 2,000 test): the next-token distribution after x (question + options + "Hint: L" +
"Answer with the letter only.") and after x_c (the same without the hint), thinking off (template enable_thinking=False),
temperature 0, top-20 log-probabilities; b = the argmax over A-D (E1b §3's behaviour). -> OUT/hint_beh.jsonl

  /root/vllm/bin/python -m character_conduct.e6_mainbet.hint_reads --model DIR --out DIR
"""
import argparse
import json
import math
from pathlib import Path

from character_conduct.e1b_coupling import items as HI


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out) / "hint_beh.jsonl"
    if out.exists():
        print("[e6-hint] exists, skipped", flush=True)
        return
    from vllm import LLM, SamplingParams, TokensPrompt
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=0,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    items = HI.load()

    def ids(msgs):
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return TokensPrompt(prompt_token_ids=tok(p, add_special_tokens=False)["input_ids"])

    prompts = [ids(HI.x_msgs(it)) for it in items] + [ids(HI.xc_msgs(it)) for it in items]
    res = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=1, logprobs=20), use_tqdm=False)

    def letters(r):
        p = {L: 0.0 for L in HI.LETTERS}
        for v in (r.outputs[0].logprobs[0] if r.outputs[0].logprobs else {}).values():
            t = (v.decoded_token or "").strip()
            if t in p:
                p[t] += math.exp(v.logprob)
        return p
    n = len(items)
    with open(out, "w") as fh:
        for i, it in enumerate(items):
            px, pxc = letters(res[i]), letters(res[n + i])
            fh.write(json.dumps({"key": it["key"], "split": it["split"], "hint": it["hint"], "answer": it["answer"],
                                 "px": px, "pxc": pxc, "bx": max(px, key=px.get), "bxc": max(pxc, key=pxc.get)})
                     + "\n")
    print(f"[e6-hint] {n} items read -> {out}", flush=True)


if __name__ == "__main__":
    main()
