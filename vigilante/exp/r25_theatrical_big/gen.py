"""R25 generation (prereg vig-r25-prereg): R18's six instructions on a larger model, thinking off, R15 sampling.
64 XSTest unsafe prompts x 2 samples (<= 800 tokens) and 40 made-up-entity questions x 1 (<= 1200 tokens) per
instruction, then R18's off-stage belief question after each honesty reply (greedy, <= 60 tokens). Prompts are read
from files and never printed.
  /root/vllm/bin/python -m vigilante.exp.r25_theatrical_big.gen --model google/gemma-4-31B-it --out <dir>"""
import argparse
import json
from pathlib import Path

from vigilante.exp.r18_theatrical.conds import BELIEF, INSTR

Q22 = Path(__file__).resolve().parents[1] / "q22_archetypes"
N_X = 2


def convs():
    xs = [json.loads(l) for l in open(Q22 / "xstest_unsafe.jsonl")]
    hs = [json.loads(l) for l in open(Q22 / "honesty_fabricated.jsonl")]
    jobs = []
    for c, sysp in INSTR.items():
        head = [{"role": "system", "content": sysp}] if sysp else []
        jobs += [{"block": "X", "cond": c, "qi": x["id"], "s": s,
                  "messages": head + [{"role": "user", "content": x["prompt"]}], "max": 800}
                 for x in xs for s in range(N_X)]
        jobs += [{"block": "H", "cond": c, "qi": x["id"], "s": 0,
                  "messages": head + [{"role": "user", "content": x["prompt"]}], "max": 1200} for x in hs]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = convs()
    print(f"[r25] {a.model}: {len(jobs)} generation jobs", flush=True)
    from vllm import LLM, SamplingParams
    kw = dict(model=a.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.92, seed=0)
    try:                                   # as pod_run: some multimodal configs reject the video key
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
    except Exception as e:
        print(f"[r25] LLM init with image+video limits failed ({type(e).__name__}); retrying", flush=True)
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0})
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
    name = a.model.split("/")[-1]
    with open(out / f"gens_r25_{name}.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps({"model": a.model, **{k: v for k, v in j.items() if k != "messages"}},
                               ensure_ascii=False) + "\n")
    print(f"[r25] {len(jobs)} generations + {len(hon)} belief answers written", flush=True)


if __name__ == "__main__":
    main()
