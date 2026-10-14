"""Exp 48 amendment 1: replies (<= 1600 tokens) and self-report ratings with vLLM (README, Amendment 1).

Replies: K samples per M/N/B stimulus, T 0.7 / top-p 0.95 / top-k 20 (as the prereg), per-request seed
crc32(iid|sample), thinking off. Ratings: the two Step 4 questions after each reply; P(digit 1-5) from the first
answer token's top-20 logprobs (a digit outside the top 20 counts 0; mass reported).

  /root/vllm/bin/python -m user_autonomy.src.vllm_gen --model Qwen/Qwen3-14B --out user_autonomy/results/q14v
"""
import argparse
import json
import math
import time
import zlib
from pathlib import Path

from user_autonomy.src.pod_job import RATE_Q

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1600)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = [it for it in json.loads((ROOT / "data" / "stimuli.json").read_text())["items"] if it["set"] in ("M", "N", "B")]
    by = {it["iid"]: it for it in items}
    t0 = time.time()
    kw = dict(model=a.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.92, seed=0,
              enable_prefix_caching=True)
    try:
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
    except Exception as e:  # noqa: BLE001
        print(f"[ua-gen] retry without video key ({type(e).__name__})", flush=True)
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()

    def render(msgs):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print(f"[ua-gen] loaded {a.model} in {time.time() - t0:.0f}s", flush=True)
    keys, prompts, params = [], [], []
    for it in items:
        for s in range(a.k):
            keys.append((it["iid"], s))
            prompts.append(render([{"role": "user", "content": it["text"]}]))
            params.append(SamplingParams(temperature=0.7, top_p=0.95, top_k=20, max_tokens=a.max_tokens,
                                         seed=zlib.crc32(f"{it['iid']}|{s}".encode())))
    t = time.time()
    res = llm.generate(prompts, params, use_tqdm=False)
    rows = []
    for (iid, s), o in zip(keys, res):
        c = o.outputs[0]
        rows.append({"iid": iid, "sample": s, "reply": c.text.strip(), "ntok": len(c.token_ids),
                     "cut": c.finish_reason == "length"})
    with open(out / "gens.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[ua-gen] {len(rows)} replies in {time.time() - t:.0f}s; cut {sum(r['cut'] for r in rows)}; "
          f"mean ntok {sum(r['ntok'] for r in rows) / len(rows):.0f}", flush=True)
    digits = [tok.encode(d, add_special_tokens=False)[0] for d in "12345"]
    rk, rp = [], []
    for r in rows:
        for q, qt in RATE_Q.items():
            rk.append((r, q))
            rp.append(render([{"role": "user", "content": by[r["iid"]]["text"]},
                              {"role": "assistant", "content": r["reply"]}, {"role": "user", "content": qt}]))
    t = time.time()
    rres = llm.generate(rp, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20), use_tqdm=False)
    with open(out / "rates.jsonl", "w") as f:
        mass = 0.0
        for (r, q), o in zip(rk, rres):
            lp = o.outputs[0].logprobs[0]
            p = [math.exp(lp[d].logprob) if d in lp else 0.0 for d in digits]
            top = max(lp.values(), key=lambda x: x.logprob).decoded_token
            mass += sum(p)
            f.write(json.dumps({"iid": r["iid"], "sample": r["sample"], "ntok": r["ntok"], "cut": r["cut"], "q": q,
                                "p": p, "mass": sum(p), "top": top}) + "\n")
    print(f"[ua-gen] {len(rk)} ratings in {time.time() - t:.0f}s; mean digit mass {mass / len(rk):.3f}", flush=True)
    (out / f"meta_{a.model.split('/')[-1]}.json").write_text(json.dumps(
        {"model": a.model, "k": a.k, "max_tokens": a.max_tokens, "engine": "vllm", "rate_q": RATE_Q}, indent=1))


if __name__ == "__main__":
    main()
