"""Version-A pilot generation (PILOT_A_DESIGN.md). Runs on the pod in the vLLM venv (eval_affect/pod/setup_vllm.sh).

Qwen3-32B, bf16, thinking on, Qwen's recommended thinking-mode sampling. One generation per job; the reply is split from
the reasoning at the </think> token id. No steering, no activations.

  /root/vllm/bin/python eval_affect/code/pod_gen.py --jobs eval_affect/materials/pilot_a/jobs.jsonl \
      --out eval_affect/results/pilot_a
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

MODEL = "Qwen/Qwen3-32B"
DECODING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
MAX_MODEL_LEN = 16384
MAX_NEW = 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N jobs only (smoke)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()][: a.limit]

    from huggingface_hub import HfApi
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt
    rev = HfApi().model_info(MODEL).sha
    t0 = time.time()
    llm = LLM(model=MODEL, revision=rev, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, enable_prefix_caching=True,
              gpu_memory_utilization=0.92, seed=0)
    tok = llm.get_tokenizer()
    end_think = tok.convert_tokens_to_ids("</think>")
    tmpl = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
    settings = {"model": MODEL, "revision": rev, "vllm": vllm.__version__, "dtype": "bfloat16", "decoding": DECODING,
                "max_new_tokens": MAX_NEW, "max_model_len": MAX_MODEL_LEN, "enable_thinking": True,
                "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest(), "n_jobs": len(jobs),
                "load_seconds": round(time.time() - t0)}
    (out / "settings.json").write_text(json.dumps(settings, indent=1))
    print(f"[gen] loaded {MODEL}@{rev[:10]} in {settings['load_seconds']}s; {len(jobs)} jobs", flush=True)

    prompts = []
    for j in jobs:
        text = tok.apply_chat_template(j["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        prompts.append(tok(text, add_special_tokens=False)["input_ids"])
    params = [SamplingParams(**DECODING, max_tokens=MAX_NEW, seed=j["seed"], skip_special_tokens=False) for j in jobs]
    t1 = time.time()
    res = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params)
    gen_s = time.time() - t1

    n_tok = 0
    with open(out / "generations.jsonl", "w") as f:
        for j, p, r in zip(jobs, prompts, res):
            o = r.outputs[0]
            ids = list(o.token_ids)
            n_tok += len(ids)
            closed = end_think in ids
            k = ids.index(end_think) if closed else len(ids)
            thinking = tok.decode(ids[:k], skip_special_tokens=True).replace("<think>", "").strip()
            reply = tok.decode(ids[k + 1:], skip_special_tokens=True).strip() if closed else ""
            f.write(json.dumps({"key": j["key"], "cell": j["cell"], "sample": j["sample"], "seed": j["seed"],
                                "thinking": thinking, "reply": reply, "think_closed": closed,
                                "finish_reason": o.finish_reason, "prompt_tokens": len(p), "gen_tokens": len(ids),
                                "think_tokens": k}, ensure_ascii=False) + "\n")
    settings.update({"gen_seconds": round(gen_s), "gen_tokens_total": n_tok,
                     "tok_per_s": round(n_tok / max(gen_s, 1e-9))})
    (out / "settings.json").write_text(json.dumps(settings, indent=1))
    print(f"[gen] {len(jobs)} generations, {n_tok} tokens in {gen_s:.0f}s ({n_tok / max(gen_s, 1e-9):.0f} tok/s)",
          flush=True)


if __name__ == "__main__":
    main()
