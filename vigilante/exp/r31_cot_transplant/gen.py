"""R31 generation (prereg vig-r31-prereg), one model per process, thinking on, R25 sampling.
Stage 1: sys none / theatrical / fiction x 40 made-up-entity questions x 4 samples, full thought + answer (<= 3000).
Stage 2: answers only (<= 1200) after a prefilled thought: sys x thought source cells of common.stage2_cells(), for
every (item, sample) whose stage-1 thoughts are closed in both conditions the cell uses.
  /root/vllm/bin/python -m vigilante.exp.r31_cot_transplant.gen --model google/gemma-4-31B-it --fmt gemma4 \
      --mml 8192 --out vigilante/exp/r31_cot_transplant/results/gemma-4-31B-it"""
import argparse
import json
import os
import time
from pathlib import Path

# as run_jobs.sh: vLLM's venv bin (ninja) on PATH and no FlashInfer sampler
os.environ["PATH"] = "/root/vllm/bin:" + os.environ.get("PATH", "")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vigilante.exp.r31_cot_transplant.common import (DECODING, MAX_ANSWER, MAX_THOUGHT, N_SAMPLES, SYS, answer_text,
                                                     items, key, parse, prefill, render, seed, stage2_cells)

CHUNK = 320


def run(llm, tok, prompts, max_tokens, keys, tag):
    from vllm import SamplingParams, TokensPrompt
    outs = []
    for c in range(0, len(prompts), CHUNK):
        t0 = time.time()
        ids = [TokensPrompt(prompt_token_ids=tok(p, add_special_tokens=False)["input_ids"]) for p in prompts[c:c + CHUNK]]
        ps = [SamplingParams(**DECODING, max_tokens=max_tokens, seed=seed(k), skip_special_tokens=False)
              for k in keys[c:c + CHUNK]]
        res = llm.generate(ids, ps, use_tqdm=False)
        outs += [(r.outputs[0].text, r.outputs[0].finish_reason, len(r.outputs[0].token_ids)) for r in res]
        print(f"[r31] {tag}: {len(outs)}/{len(prompts)} ({time.time() - t0:.0f}s for this chunk, "
              f"{sum(o[2] for o in outs[-len(res):])} tokens)", flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fmt", required=True, choices=["qwen", "gemma4"])
    ap.add_argument("--mml", type=int, default=8192)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N items (smoke)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    qs = items()[: a.limit]
    import vllm
    from huggingface_hub import HfApi
    from vllm import LLM
    rev = HfApi().model_info(a.model).sha
    t0 = time.time()
    kw = dict(model=a.model, revision=rev, dtype="bfloat16", max_model_len=a.mml, gpu_memory_utilization=0.92, seed=0,
              enable_prefix_caching=True)
    try:                                   # as pod_run: some multimodal configs reject the video key
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
    except Exception as e:
        print(f"[r31] LLM init with image+video limits failed ({type(e).__name__}); retrying", flush=True)
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    settings = {"model": a.model, "revision": rev, "vllm": vllm.__version__, "dtype": "bfloat16", "decoding": DECODING,
                "max_model_len": a.mml, "fmt": a.fmt, "load_seconds": round(time.time() - t0), "examples": {}}
    print(f"[r31] loaded {a.model}@{rev[:10]} in {settings['load_seconds']}s; {len(qs)} items", flush=True)

    # ---- stage 1: natural thought + answer
    s1 = [{"key": key(a.model, sys, "natural", q["id"], s), "model": a.model, "stage": 1, "sys": sys,
           "src": "natural", "item": q["id"], "s": s, "prompt": render(tok, sys, q["prompt"])}
          for sys in SYS for q in qs for s in range(N_SAMPLES)]
    for sys in SYS:
        ex = next(r["prompt"] for r in s1 if r["sys"] == sys)
        settings["examples"][f"stage1|{sys}"] = ex
        print(f"[r31] stage-1 prompt, sys {sys}: {ex!r}", flush=True)
    res = run(llm, tok, [r["prompt"] for r in s1], MAX_THOUGHT, [r["key"] for r in s1], "stage 1")
    for r, (raw, fin, n) in zip(s1, res):
        th, ans, closed = parse(raw, a.fmt)
        # closed = the thought block ended (usable for transplant) even if the natural answer was then cut
        r.update(raw=raw, finish=fin, n_tokens=n, thought=th, answer=ans, closed=closed)
    with open(out / "stage1.jsonl", "w") as f:
        for r in s1:
            f.write(json.dumps({k: v for k, v in r.items() if k != "prompt"}, ensure_ascii=False) + "\n")
    settings["stage1"] = {"n": len(s1), "unclosed": sum(not r["closed"] for r in s1),
                          "empty_thought": sum(r["closed"] and not r["thought"] for r in s1),
                          "seconds": round(time.time() - t0)}
    print(f"[r31] stage 1 written: {settings['stage1']}", flush=True)

    # ---- stage 2: answers after a prefilled thought
    T = {(r["sys"], r["item"], r["s"]): r for r in s1}
    qtext = {q["id"]: q["prompt"] for q in qs}
    s2 = []
    for sys, src in stage2_cells():
        for q in qs:
            for s in range(N_SAMPLES):
                need = {"none"} | ({src} if src != "empty" else set()) | ({sys} if sys != "none" else set())
                if not all(T[(c, q["id"], s)]["closed"] for c in need):
                    continue
                th = "" if src == "empty" else T[(src, q["id"], s)]["thought"]
                p = prefill(render(tok, sys, qtext[q["id"]]), th, a.fmt)
                s2.append({"key": key(a.model, sys, src, q["id"], s), "model": a.model, "stage": 2, "sys": sys,
                           "src": src, "item": q["id"], "s": s, "thought_chars": len(th), "prompt": p})
    for sys, src in stage2_cells():
        ex = next((r["prompt"] for r in s2 if r["sys"] == sys and r["src"] == src), None)
        settings["examples"][f"stage2|{sys}|{src}"] = ex
        print(f"[r31] stage-2 prompt, sys {sys}, thought from {src}: {ex!r}"[:3000], flush=True)
    res = run(llm, tok, [r["prompt"] for r in s2], MAX_ANSWER, [r["key"] for r in s2], "stage 2")
    for r, (raw, fin, n) in zip(s2, res):
        r.update(raw=raw, finish=fin, n_tokens=n, answer=answer_text(raw, a.fmt))
    with open(out / "stage2.jsonl", "w") as f:
        for r in s2:
            f.write(json.dumps({k: v for k, v in r.items() if k != "prompt"}, ensure_ascii=False) + "\n")
    settings["stage2"] = {"n": len(s2), "cut": sum(r["finish"] == "length" for r in s2),
                          "seconds": round(time.time() - t0)}
    (out / "settings.json").write_text(json.dumps(settings, indent=1, ensure_ascii=False))
    print(f"[r31] stage 2 written: {settings['stage2']}; done", flush=True)


if __name__ == "__main__":
    main()
