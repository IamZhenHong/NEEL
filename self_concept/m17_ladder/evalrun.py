"""SC-M17 eval (DESIGN.md §4): one ladder model under vLLM (bf16, thinking off via V1's prompt builder), items.items().

Letter items: one greedy token with the top-20 log-probabilities; p_A = P(A) / (P(A) + P(B)) over the A / B first-token
forms (" A", "A"; " B", "B") present among them. A letter missing from the top 20 gets the 20th log-prob minus 1 and is
flagged. Text items (SP): greedy, at most 12 tokens.
Records: the item without its messages + model, p_A, top1 (the greedy token), missing, text.

  /root/vllm/bin/python -m self_concept.m17_ladder.evalrun --model DIR --cell q9 --out DIR
"""
import argparse
import json
import math
import os
from pathlib import Path

from persona_affect.v1_value_hook.fmt import prompt_ids
from self_concept.m17_ladder.items import items


def _lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def fake_run(its):
    from persona_affect.common.util import seed
    out = []
    for it in its:
        h = seed(it["key"])
        out.append({"p_A": (h % 97) / 96, "top1": "A", "missing": False, "text": "lion" if it["kind"] == "text" else ""})
    return out


def run(model, its):
    from vllm import LLM, SamplingParams, TokensPrompt
    llm = LLM(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=0,
              enable_prefix_caching=True, max_logprobs=20, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    forms = {L: {tok.encode(f, add_special_tokens=False)[0] for f in (L, " " + L) if len(tok.encode(f, add_special_tokens=False)) == 1}
             for L in "AB"}
    prompts = [TokensPrompt(prompt_token_ids=prompt_ids(tok, it["messages"], None)) for it in its]
    sps = [SamplingParams(max_tokens=1, temperature=0, logprobs=20) if it["kind"] == "letter"
           else SamplingParams(max_tokens=12, temperature=0) for it in its]
    res = llm.generate(prompts, sps, use_tqdm=False)
    out = []
    for it, r in zip(its, res):
        o = r.outputs[0]
        if it["kind"] != "letter":
            out.append({"p_A": None, "top1": None, "missing": False, "text": o.text})
            continue
        lp = {t: v.logprob for t, v in o.logprobs[0].items()}
        floor = min(lp.values()) - 1
        got, miss = {}, False
        for L, ids in forms.items():
            have = [lp[t] for t in ids if t in lp]
            got[L] = _lse(have) if have else floor
            miss |= not have
        pa = 1 / (1 + math.exp(got["B"] - got["A"]))
        out.append({"p_A": pa, "top1": tok.decode([o.token_ids[0]]), "missing": miss, "text": o.text})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m17 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    its = items()
    res = fake_run(its) if os.environ.get("V1_FAKE_GEN") == "1" else run(a.model, its)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / f"eval_{a.cell}.jsonl", "w", encoding="utf-8") as fh:
        for it, r in zip(its, res):
            fh.write(json.dumps({k: v for k, v in it.items() if k != "messages"} | {"cell": a.cell} | r,
                                ensure_ascii=False) + "\n")
    print(f"[m17-eval] {a.cell}: {len(res)} records", flush=True)


if __name__ == "__main__":
    main()
