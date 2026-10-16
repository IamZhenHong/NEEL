"""vLLM side of e64's agent episodes (copied from vigilante/src/pod_run.py, 2026-10-16; changes: e64 condition ids via
engine.py, one condition at a time so each LoRA runs alone, a round cap, an engine shared with the battery runs).
Decoding as pod_run: T 0.7, top-p 0.95, top-k 20, per-request seed; Qwen3-14B, thinking off, fmt qwen3.
Used by stage_vllm.py, stage_b.py, stage_bc1.py and stage_c.py (e64c added the `fmt` argument for Gemma 4; the default
keeps e64/e64b unchanged; for Qwen "qwen" and "qwen3" render and clean identically)."""
import time

from vibes.e64_layer_map.engine import dumps, episode_record, run
from vigilante.src.templates import clean_output, to_template

DECODING = dict(temperature=0.7, top_p=0.95, top_k=20, min_p=0.0)


def make_generate(llm, tok, max_model_len, lora_request=None, stats=None, fmt="qwen3"):
    from vllm import SamplingParams, TokensPrompt

    def generate(items):
        prompts, params, skip = [], [], []
        for it in items:
            text = tok.apply_chat_template(to_template(it["messages"], fmt), tools=it["tools"] or None,
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            room = max_model_len - len(ids) - 8
            skip.append(room < 64)
            prompts.append(ids)
            params.append(SamplingParams(**DECODING, max_tokens=max(1, min(it["max_tokens"], room)), seed=it["seed"],
                                         skip_special_tokens=False))
        idx = [i for i, s in enumerate(skip) if not s]
        res = llm.generate([TokensPrompt(prompt_token_ids=prompts[i]) for i in idx], [params[i] for i in idx],
                           use_tqdm=False, lora_request=lora_request)
        outs = [("", "context_full")] * len(items)
        for i, r in zip(idx, res):
            o = r.outputs[0]
            if stats is not None:
                stats["gen_tokens"] = stats.get("gen_tokens", 0) + len(o.token_ids)
            outs[i] = (clean_output(o.text, fmt), o.finish_reason)
        return outs
    return generate


def run_agents(llm, tok, jobs, path, max_model_len, lora_request=None, tag="", fmt="qwen3"):
    """Run one condition's episodes and write them to `path` (atomic). Returns {n, seconds, gen_tokens}."""
    stats = {"gen_tokens": 0}
    t0 = time.time()
    gen = make_generate(llm, tok, max_model_len, lora_request, stats, fmt)
    eps = run(jobs, gen, log=lambda m: print(f"[e64-vllm] {tag} {m} ({time.time() - t0:.0f}s)", flush=True))
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for e in eps:
            f.write(dumps(episode_record(e)) + "\n")
    tmp.replace(path)
    return {"n": len(jobs), "seconds": round(time.time() - t0), "gen_tokens": stats["gen_tokens"]}
