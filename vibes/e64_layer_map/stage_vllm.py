"""e64 vLLM stage (vLLM venv): every non-hook condition on both surfaces, one engine.

  pilot  B: VB-EM (10 samples) + the 288 base-rate episodes (48 per candidate family); T8, T9: VB-EM (10 samples) for
         gates (ii) and (iii).
  main   B, T0, T1-T5, T8, T8c, T9: VB-1 (EM HC SY TR FAB, full n) + the main episodes.
  tiny   pod smoke on Qwen3-0.6B: tiny batteries and jobs, no adapters.

  /root/vllm/bin/python -m vibes.e64_layer_map.stage_vllm --plan pilot --out vibes/e64_layer_map/results/pilot

Battery requests come from vibes.batteries.build and are scored by vibes.batteries; they are run by vb_run() below,
a copy of vibes.batteries.run_vllm.run whose only change is that the LoRA request is passed in (run_vllm.run names
every adapter LoRARequest("vb_adapter", 1, path), and vLLM caches adapters by that integer id, so a second adapter in
the same engine would silently reuse the first).
"""
import argparse
import json
import math
import time
from pathlib import Path

from vibes.batteries import run_vllm as vbv  # noqa: I001  (sets PATH / sampler env before vllm is imported)
from vibes.batteries import build, write_jsonl
from vibes.e64_layer_map import conds as C
from vibes.e64_layer_map.agent_vllm import run_agents

H = Path(__file__).resolve().parent
ALL_BATS = ("EM", "HC", "SY", "TR", "FAB")
MAX_MODEL_LEN = 16384


def vb_run(llm, tok, requests, lreq=None, log_every=vbv.CHUNK):
    """vibes.batteries.run_vllm.run with the LoRA request passed in (see module docstring); otherwise identical."""
    from vllm import SamplingParams, TokensPrompt
    t0 = time.time()
    outs = [None] * len(requests)
    gen = [i for i, r in enumerate(requests) if r["kind"] == "gen"]
    let = [i for i, r in enumerate(requests) if r["kind"] == "letters"]
    ntok = 0
    for st in range(0, len(gen), log_every):
        idx = gen[st:st + log_every]
        prompts = [TokensPrompt(prompt_token_ids=vbv.encode(tok, requests[i]["messages"])) for i in idx]
        params = []
        for i in idx:
            s = requests[i]["sampling"]
            params.append(SamplingParams(temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"],
                                         max_tokens=s["max_tokens"], seed=s["seed"]))
        res = llm.generate(prompts, params, use_tqdm=False, lora_request=lreq)
        for i, r in zip(idx, res):
            o = r.outputs[0]
            ntok += len(o.token_ids)
            outs[i] = {"text": vbv.clean(o.text), "finish": o.finish_reason, "n_tokens": len(o.token_ids)}
        print(f"[vb.vllm] gen {min(st + log_every, len(gen))}/{len(gen)} ({ntok} tokens, {time.time() - t0:.0f}s)",
              flush=True)
    for st in range(0, len(let), log_every):
        idx = let[st:st + log_every]
        prompts = [TokensPrompt(prompt_token_ids=vbv.encode(tok, requests[i]["messages"])) for i in idx]
        k = max(requests[i]["sampling"].get("logprobs", 20) for i in idx)
        res = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1, logprobs=k), use_tqdm=False,
                           lora_request=lreq)
        for i, r in zip(idx, res):
            lp = r.outputs[0].logprobs[0]
            tp = {tid: math.exp(v.logprob) for tid, v in lp.items()}
            outs[i] = vbv.letter_record(requests[i], tp, vbv.letter_token_ids(tok, requests[i]["letters"]))
        print(f"[vb.vllm] letters {min(st + log_every, len(let))}/{len(let)} ({time.time() - t0:.0f}s)", flush=True)
    return [{**{k: v for k, v in r.items() if k != "messages"}, **o, "engine": "vllm"}
            for r, o in zip(requests, outs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, choices=["pilot", "main", "tiny"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=C.MODEL)
    ap.add_argument("--conds", nargs="*", default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tiny = a.plan == "tiny"
    if a.plan == "pilot":
        plan = {"B": (("EM",), 10, "pilot_vllm.jsonl"), "T8": (("EM",), 10, None), "T9": (("EM",), 10, None)}
    elif a.plan == "main":
        plan = {c: (ALL_BATS, 20, "main_vllm.jsonl") for c, v in C.CONDS.items() if v[0] == "vllm"}
    else:
        plan = {"B": (ALL_BATS, 1, "tiny_vllm.jsonl"), "T3": (ALL_BATS, 1, "tiny_vllm.jsonl")}
    if a.conds:
        plan = {c: v for c, v in plan.items() if c in a.conds}
    loras = sorted({C.CONDS[c][2] for c in plan if C.CONDS[c][2]})

    from huggingface_hub import HfApi
    import vllm
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    rev = HfApi().model_info(a.model).sha
    t0 = time.time()
    kw = dict(enable_lora=True, max_lora_rank=64, max_loras=1, max_cpu_loras=len(loras)) if loras else {}
    llm = LLM(model=a.model, revision=rev, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=a.gpu_mem,
              seed=0, enable_prefix_caching=True, **kw)
    tok = llm.get_tokenizer()
    lreq = {n: LoRARequest(f"e64_{n}", i + 1, str(C.REPO / C.ADAPTERS[n])) for i, n in enumerate(loras)}
    log = {"plan": a.plan, "model": a.model, "revision": rev, "vllm": vllm.__version__, "loras": {n: C.ADAPTERS[n] for n in loras},
           "load_seconds": round(time.time() - t0), "conds": {}}
    print(f"[e64-vllm] loaded {a.model}@{rev[:10]} in {log['load_seconds']}s; plan {list(plan)}", flush=True)
    for cond, (bats, n_em, jobs_file) in plan.items():
        lr = lreq.get(C.CONDS[cond][2])
        t1 = time.time()
        p = out / f"outputs_{cond}.jsonl"
        if not p.exists():
            reqs = build(C.system(cond), batteries=bats, n_em=n_em, seed=C.SEED[cond], cond=cond, tiny=tiny)
            outs = vb_run(llm, tok, reqs, lr)
            for o in outs:
                o.update(model=a.model, adapter=C.ADAPTERS.get(C.CONDS[cond][2]), e64_cond=cond)
            write_jsonl(outs, str(p) + ".tmp")
            Path(str(p) + ".tmp").replace(p)
            print(f"[e64-vllm] {cond}: {len(outs)} battery outputs in {time.time() - t1:.0f}s", flush=True)
        info = None
        ep = out / f"episodes_{cond}.jsonl"
        if jobs_file and not ep.exists():
            jobs = [json.loads(l) for l in open(H / "jobs" / jobs_file) if l.strip()]
            jobs = [j for j in jobs if j["cond"] == cond]
            if jobs:
                info = run_agents(llm, tok, jobs, ep, MAX_MODEL_LEN, lr, tag=cond)
                print(f"[e64-vllm] {cond}: {info}", flush=True)
        log["conds"][cond] = {"seconds": round(time.time() - t1), "agents": info}
        (out / f"stage_vllm_{a.plan}.json").write_text(json.dumps(log, indent=1))
    print(f"[e64-vllm] stage done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
