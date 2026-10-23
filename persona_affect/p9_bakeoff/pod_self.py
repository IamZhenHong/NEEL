"""PA-P9 pod: Qwen3.6-27B's own forecast at run F's decision points (vLLM, thinking off: the reply is the forecast,
not a reasoned decision). Reads only jobs/self_q27.jsonl.

  /root/vllm/bin/python -m persona_affect.p9_bakeoff.pod_self --out DIR
Settings: F's model, bf16, max_model_len 8192, mini 0's DECODING, max 64 new tokens, seed = mini0.seed(f"{pkey}|self|{j}").
Writes DIR/self.jsonl: {key, pkey, cell, task, sample, j, text, finish, option, mode}; option = a choose_option call's
option, else the one option id the reply names as a word (None if none or several); mode = its role in amap.
"""
import argparse
import json
import re
import time
from pathlib import Path

from persona.exemplar.mini0 import DECODING, parse_option, seed

MODEL = "Qwen/Qwen3.6-27B"
JOBS = Path(__file__).resolve().parent / "jobs" / "self_q27.jsonl"
MAX_NEW = 64


def option_of(text, amap):
    oid = parse_option(text)
    if oid in amap:
        return oid
    found = {o for o in amap if re.search(rf"\b{re.escape(o)}\b", text, re.I)}
    return found.pop() if len(found) == 1 else None


def main(argv=None):
    from vllm import LLM, SamplingParams, TokensPrompt
    ap = argparse.ArgumentParser(prog="p9 pod_self")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(line) for line in open(JOBS)]
    t0 = time.time()
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, seed=0,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()

    def ptxt(j):
        return tok.apply_chat_template(j["messages"], tools=j["tools"], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    probe = ptxt({"messages": [{"role": "user", "content": "x"}], "tools": None})
    assert probe.endswith("<think>\n\n</think>\n\n"), f"thinking-off template ends {probe[-40:]!r}"
    print(f"[p9] loaded {MODEL} in {time.time() - t0:.0f}s; {len(jobs)} prompts", flush=True)
    todo = [(job, j) for job in jobs for j in range(job["k"])]
    prompts = [TokensPrompt(prompt_token_ids=tok(ptxt(job), add_special_tokens=False)["input_ids"]) for job, _ in todo]
    params = [SamplingParams(**DECODING, max_tokens=MAX_NEW, seed=seed(f"{job['pkey']}|self|{j}")) for job, j in todo]
    res = llm.generate(prompts, params, use_tqdm=False)
    with open(out / "self.jsonl", "w") as f:
        for (job, j), r in zip(todo, res):
            text = r.outputs[0].text
            oid = option_of(text, job["amap"])
            f.write(json.dumps({"key": f"{job['pkey']}|self|{j}", "pkey": job["pkey"], "cell": job["cell"],
                                "task": job["task"], "sample": job["sample"], "j": j, "text": text,
                                "finish": r.outputs[0].finish_reason, "option": oid,
                                "mode": job["amap"].get(oid)}, ensure_ascii=False) + "\n")
    print(f"[p9] {len(todo)} forecasts in {time.time() - t0:.0f}s -> {out / 'self.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
