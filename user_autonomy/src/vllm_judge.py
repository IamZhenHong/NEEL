"""Exp 48 cross-family judge pass on the pod: Gemma 4 31B (vLLM, greedy) labels every reply in the given gens files
with the primary judge's exact prompt (`judge.SYS` + `judge.prompt`). Output: <out>/judge_<dir>_<stem>.jsonl rows
{key: [iid, sample, cond], text}.

  /root/vllm/bin/python -m user_autonomy.src.vllm_judge --model google/gemma-4-31B-it --out DIR GENS.jsonl ...
"""
import argparse
import json
import time
from pathlib import Path

from user_autonomy.src.judge import SYS, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--out", required=True)
    ap.add_argument("gens", nargs="+")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    t0 = time.time()
    kw = dict(model=a.model, dtype="bfloat16", max_model_len=6144, gpu_memory_utilization=0.92, seed=0,
              enable_prefix_caching=True)
    try:
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
    except Exception as e:  # noqa: BLE001
        print(f"[ua-judge] retry without video key ({type(e).__name__})", flush=True)
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    print(f"[ua-judge] loaded {a.model} in {time.time() - t0:.0f}s", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=500)
    for g in a.gens:
        rows = [json.loads(l) for l in open(g)]
        texts = [tok.apply_chat_template([{"role": "user", "content": SYS + "\n\n" + prompt(r["iid"], r["reply"])}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for r in rows]
        t = time.time()
        res = llm.generate(texts, sp, use_tqdm=False)
        out = Path(a.out) / f"judge_{Path(g).parent.name}_{Path(g).stem}.jsonl"
        with open(out, "w") as f:
            for r, o in zip(rows, res):
                f.write(json.dumps({"key": [r["iid"], r["sample"], r.get("cond")], "text": o.outputs[0].text}) + "\n")
        print(f"[ua-judge] {g}: {len(rows)} in {time.time() - t:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
