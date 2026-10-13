"""R18 vLLM step (Qwen3-14B bf16, thinking off, R15 sampling): the 6 instructions x 64 XSTest unsafe prompts (<= 800
tokens) and x 40 made-up-entity questions (<= 1200 tokens), then the off-stage belief question appended after each
honesty reply (greedy, <= 60 tokens). Prompts are read from files and never printed.
  /root/vllm/bin/python -m vigilante.exp.r18_theatrical.gen --out vigilante/exp/r18_theatrical/results"""
import argparse
import json
from pathlib import Path

from vigilante.exp.r18_theatrical.conds import BELIEF, INSTR

Q22 = Path(__file__).resolve().parents[1] / "q22_archetypes"


def convs():
    xs = [json.loads(l) for l in open(Q22 / "xstest_unsafe.jsonl")]
    hs = [json.loads(l) for l in open(Q22 / "honesty_fabricated.jsonl")]
    jobs = []
    for c, sysp in INSTR.items():
        head = [{"role": "system", "content": sysp}] if sysp else []
        jobs += [{"block": "X", "cond": c, "qi": x["id"], "messages": head + [{"role": "user", "content": x["prompt"]}],
                  "max": 800} for x in xs]
        jobs += [{"block": "H", "cond": c, "qi": x["id"], "messages": head + [{"role": "user", "content": x["prompt"]}],
                  "max": 1200} for x in hs]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = convs()
    print(f"[r18] {len(jobs)} generation jobs", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen3-14B", dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()
    tmpl = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    res = llm.generate([tmpl(j["messages"]) for j in jobs],
                       [SamplingParams(temperature=0.7, top_p=0.95, top_k=20, max_tokens=j["max"], seed=8000 + n)
                        for n, j in enumerate(jobs)])
    for j, r in zip(jobs, res):
        j["completion"], j["finish"] = r.outputs[0].text, r.outputs[0].finish_reason
    hon = [j for j in jobs if j["block"] == "H"]
    follow = [j["messages"] + [{"role": "assistant", "content": j["completion"]}, {"role": "user", "content": BELIEF}]
              for j in hon]
    res2 = llm.generate([tmpl(m) for m in follow], SamplingParams(temperature=0.0, max_tokens=60))
    for j, r in zip(hon, res2):
        j["belief"] = r.outputs[0].text
    with open(out / "gens_vllm.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps({k: v for k, v in j.items() if k != "messages"}, ensure_ascii=False) + "\n")
    print(f"[r18] {len(jobs)} generations + {len(hon)} belief answers written", flush=True)


if __name__ == "__main__":
    main()
