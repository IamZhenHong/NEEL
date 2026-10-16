"""VB-1 vLLM runner: plain generation and letter reads for request records from vibes.batteries.build, written to
one outputs.jsonl. Run with the vLLM venv (vigilante/pod/setup_vllm.sh: vLLM 0.30.0 +cu129), thinking off.

  /root/vllm/bin/python -m vibes.batteries.run_vllm --model Qwen/Qwen3-14B --requests r1.jsonl [r2.jsonl ...] \
      --out outputs.jsonl [--lora PATH]

As a library (one engine for many conditions):
  llm, tok = load_llm(model); outs = run(llm, tok, requests); write_jsonl(outs, path)

Prompts are tokenised here (chat template, add_generation_prompt, enable_thinking=False; no extra special tokens) and
passed as token ids, so the HF runner sees the same ids. Letter reads: max_tokens 1, top-20 logprobs; letters not
in the top 20 get probability 0 (counted in `missing`)."""
from __future__ import annotations

import argparse
import os
import time

os.environ["PATH"] = "/root/vllm/bin:" + os.environ.get("PATH", "")   # as R25 gen.py: ninja on PATH
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from .build import read_jsonl, write_jsonl  # noqa: E402

STRIP = ("<|im_end|>", "<|endoftext|>", "<turn|>", "<eos>", "<|turn>")
CHUNK = 1500                     # generate in chunks so progress is printed (the launcher kills a silent job)


def encode(tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def letter_token_ids(tok, letters):
    """{letter: [id of "L", id of " L"]}; both must be single tokens."""
    out = {}
    for L in letters:
        ids = []
        for s in (L, " " + L):
            t = tok.encode(s, add_special_tokens=False)
            if len(t) != 1:
                raise ValueError(f"letter {s!r} is not one token: {t}")
            ids.append(t[0])
        out[L] = ids
    return out


def letter_record(req, token_p: dict[int, float], lid):
    """Letter probabilities from a {token_id: prob} map: summed over "L" and " L", mass recorded, renormalised."""
    raw = [sum(token_p.get(i, 0.0) for i in lid[L]) for L in req["letters"]]
    mass = sum(raw)
    probs = [x / mass for x in raw] if mass > 0 else [1.0 / len(raw)] * len(raw)
    missing = sum(all(i not in token_p for i in lid[L]) for L in req["letters"])
    return {"probs": [round(p, 6) for p in probs], "mass": round(mass, 6),
            "argmax": max(range(len(probs)), key=probs.__getitem__), "missing": missing}


def clean(text):
    for s in STRIP:
        text = text.replace(s, "")
    return text.strip()


def load_llm(model, lora=False, max_model_len=4096, gpu_memory_utilization=0.92, max_lora_rank=64):
    from vllm import LLM
    kw = dict(model=model, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
              seed=0, enable_prefix_caching=True)
    if lora:
        kw.update(enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1)
    llm = None
    for mm in ({"image": 0, "video": 0}, {"image": 0}, None):   # as pod_run: some configs reject a modality key
        try:
            llm = LLM(**kw, **({"limit_mm_per_prompt": mm} if mm is not None else {}))
            break
        except Exception as e:  # noqa: BLE001
            print(f"[vb.vllm] LLM init with limit_mm_per_prompt={mm} failed ({type(e).__name__}: {str(e)[:200]})",
                  flush=True)
    if llm is None:
        raise RuntimeError("vLLM init failed")
    return llm, llm.get_tokenizer()


def run(llm, tok, requests, lora_path=None, log_every=CHUNK):
    from vllm import SamplingParams, TokensPrompt
    lreq = None
    if lora_path:
        from vllm.lora.request import LoRARequest
        lreq = LoRARequest("vb_adapter", 1, lora_path)
    t0 = time.time()
    outs = [None] * len(requests)
    gen = [i for i, r in enumerate(requests) if r["kind"] == "gen"]
    let = [i for i, r in enumerate(requests) if r["kind"] == "letters"]
    ntok = 0
    for st in range(0, len(gen), log_every):
        idx = gen[st:st + log_every]
        prompts = [TokensPrompt(prompt_token_ids=encode(tok, requests[i]["messages"])) for i in idx]
        params = []
        for i in idx:
            s = requests[i]["sampling"]
            kw = {"top_k": s["top_k"]} if s["top_k"] > 0 else {}     # no top_k: vLLM's default (disabled)
            params.append(SamplingParams(temperature=s["temperature"], top_p=s["top_p"], max_tokens=s["max_tokens"],
                                         seed=s["seed"], **kw))
        res = llm.generate(prompts, params, use_tqdm=False, lora_request=lreq)
        for i, r in zip(idx, res):
            o = r.outputs[0]
            ntok += len(o.token_ids)
            outs[i] = {"text": clean(o.text), "finish": o.finish_reason, "n_tokens": len(o.token_ids)}
        print(f"[vb.vllm] gen {min(st + log_every, len(gen))}/{len(gen)} ({ntok} tokens, {time.time() - t0:.0f}s)",
              flush=True)
    for st in range(0, len(let), log_every):
        idx = let[st:st + log_every]
        prompts = [TokensPrompt(prompt_token_ids=encode(tok, requests[i]["messages"])) for i in idx]
        k = max(requests[i]["sampling"].get("logprobs", 20) for i in idx)
        res = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1, logprobs=k), use_tqdm=False,
                           lora_request=lreq)
        import math
        for i, r in zip(idx, res):
            lp = r.outputs[0].logprobs[0]          # {token_id: Logprob} (vLLM's default, non-flat logprobs)
            tp = {int(tid): math.exp(getattr(v, "logprob", v)) for tid, v in dict(lp).items()}
            outs[i] = letter_record(requests[i], tp, letter_token_ids(tok, requests[i]["letters"]))
        print(f"[vb.vllm] letters {min(st + log_every, len(let))}/{len(let)} ({time.time() - t0:.0f}s)", flush=True)
    return [{**{k: v for k, v in r.items() if k != "messages"}, **o, "engine": "vllm"}
            for r, o in zip(requests, outs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--requests", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--max-model-len", type=int, default=4096)
    a = ap.parse_args()
    reqs = [r for p in a.requests for r in read_jsonl(p)]
    t0 = time.time()
    llm, tok = load_llm(a.model, lora=bool(a.lora), max_model_len=a.max_model_len)
    print(f"[vb.vllm] loaded {a.model} in {time.time() - t0:.0f}s; {len(reqs)} requests", flush=True)
    outs = run(llm, tok, reqs, a.lora)
    for o in outs:
        o["model"] = a.model
        o["adapter"] = a.lora
    write_jsonl(outs, a.out)
    print(f"[vb.vllm] wrote {len(outs)} outputs -> {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
