"""Stage 0 (PLAN.md): room check on base Qwen3-14B (B) and e64's recipe-matched control adapter (T8c). No training.

  uv run python -m persona.coupling.stage0 jobs                  # write jobs/stage0.jsonl (+ jobs/tiny.jsonl)
  /root/vllm/bin/python -m persona.coupling.stage0 run --jobs persona/coupling/jobs/stage0.jsonl --out <dir>

Decoding as e64 (vibes/e64_layer_map/agent_vllm.py): T 0.7, top-p 0.95, top-k 20, per-request seed, thinking off,
Qwen3 Hermes tool format, 1,024 new tokens per turn, at most 8 rounds.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from persona.coupling import tasks as T     # registers the cases and patches episode.condition
from vigilante.src.episode import dumps, episode_record, run_batch
from vigilante.src.templates import clean_output, to_template

H = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-14B"
ADAPTERS = {"T8c": "vibes/e64_layer_map/results/adapters/T8c"}
MODELS = ("B", "T8c")
N = 4                     # samples per gate cell
N_HARM = 8                # samples per harm cell (the keep rule needs more than 4)
MAX_TOKENS = 1024
MAX_ROUNDS = 8
MAX_MODEL_LEN = 8192
DECODING = dict(temperature=0.7, top_p=0.95, top_k=20, min_p=0.0)


def seed(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


BASES = ("B", "base")      # stage 0 called the untouched model "B"; stage 1 calls it "base" (B is an arm there)


def job(model, cid, i, part, exp="CPL0"):
    key = f"{exp}|{model}|{cid}|{i}"
    j = {"key": key, "exp": exp, "part": part, "model": model, "case": cid, "cond": cid, "seed": seed(key),
         "sample": i, "thinking": False, "handover": False, "fmt": "qwen3", "max_tokens": MAX_TOKENS}
    if model not in BASES:
        j["lora"] = model
    return j


def build(models=MODELS, n=N, n_harm=N_HARM, exp="CPL0"):
    js = []
    for m in models:
        for tid in T.GATE:
            for cond in ["none", *T.GATE_VALUES, *T.GATE_LABELS]:
                js += [job(m, T.case_id(tid, "sys", cond), i, "gate", exp) for i in range(n)]
        for tid in T.HARM:
            js += [job(m, T.case_id(tid, "sys", "none"), i, "harm", exp) for i in range(n_harm)]
            for ch in T.CHANNELS:
                for kind in T.HARM_KINDS:
                    js += [job(m, T.case_id(tid, ch, f"{kind}.{i % 2}"), i, "harm", exp) for i in range(n_harm)]
    assert len({j["key"] for j in js}) == len(js)
    return js


def write(path, js):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(j) + "\n" for j in js)
    print(f"{path}: {len(js)} jobs")


def make_generate(llm, tok, lreqs, stats):
    from vllm import SamplingParams, TokensPrompt

    def generate(items):
        by = {}
        for i, it in enumerate(items):
            by.setdefault(it.get("lora"), []).append(i)
        outs = [("", "context_full")] * len(items)
        for lora, idx in by.items():
            prompts, params, keep = [], [], []
            for i in idx:
                it = items[i]
                text = tok.apply_chat_template(to_template(it["messages"], "qwen3"), tools=it["tools"] or None,
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = tok(text, add_special_tokens=False)["input_ids"]
                room = MAX_MODEL_LEN - len(ids) - 8
                if room < 64:
                    continue
                keep.append(i)
                prompts.append(TokensPrompt(prompt_token_ids=ids))
                params.append(SamplingParams(**DECODING, max_tokens=min(it["max_tokens"], room), seed=it["seed"],
                                             skip_special_tokens=False))
            res = llm.generate(prompts, params, use_tqdm=False, lora_request=lreqs.get(lora))
            for i, r in zip(keep, res):
                o = r.outputs[0]
                stats["gen_tokens"] += len(o.token_ids)
                outs[i] = (clean_output(o.text, "qwen3"), o.finish_reason)
        return outs
    return generate


def run(a):
    import vllm
    from huggingface_hub import HfApi
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    loras = sorted({j["lora"] for j in jobs if j.get("lora")})
    rev = HfApi().model_info(a.model).sha
    t0 = time.time()
    kw = dict(enable_lora=True, max_lora_rank=64, max_loras=max(1, len(loras))) if loras else {}
    llm = LLM(model=a.model, revision=rev, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=a.gpu_mem, seed=0, enable_prefix_caching=True, **kw)
    tok = llm.get_tokenizer()
    # distinct integer ids per adapter (vLLM caches adapters by id; e64 README note 5)
    path = lambda n: Path(a.repo) / ADAPTERS[n] if n in ADAPTERS else Path(a.adapter_dir) / n  # noqa: E731
    lreqs = {n: LoRARequest(f"cpl_{n}", i + 1, str(path(n))) for i, n in enumerate(loras)}
    stats = {"gen_tokens": 0}
    gen = make_generate(llm, tok, lreqs, stats)
    print(f"[cpl0] loaded {a.model}@{rev[:10]} in {time.time() - t0:.0f}s; {len(jobs)} jobs, loras {loras}", flush=True)
    t1 = time.time()
    eps = run_batch(jobs, gen, max_rounds=MAX_ROUNDS,
                    log=lambda m: print(f"[cpl0] {m} ({time.time() - t1:.0f}s)", flush=True))
    tmp = out / "episodes.tmp"
    with open(tmp, "w") as f:
        for e in eps:
            f.write(dumps(episode_record(e)) + "\n")
    tmp.replace(out / "episodes.jsonl")
    (out / "run_info.json").write_text(json.dumps(
        {"model": a.model, "revision": rev, "vllm": vllm.__version__, "loras": {n: str(path(n)) for n in loras},
         "n_jobs": len(jobs), "seconds": round(time.time() - t0), **stats}, indent=1))
    print(f"[cpl0] done: {len(eps)} episodes in {time.time() - t0:.0f}s, {stats['gen_tokens']} tokens", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("jobs")
    r = sub.add_parser("run")
    r.add_argument("--jobs", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default=MODEL)
    r.add_argument("--repo", default=".")
    r.add_argument("--gpu-mem", type=float, default=0.90)
    r.add_argument("--adapter-dir", default="persona/coupling/results/adapters")
    j1 = sub.add_parser("jobs1")
    j1.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "jobs1":      # stage 1: base + the four arms at one seed, same cells as stage 0
        ms = ("base", *(f"{arm}_s{a.seed}" for arm in ("A", "B", "Bp", "C")))
        write(H / "jobs" / f"stage1_s{a.seed}.jsonl", build(models=ms, exp=f"CPL1s{a.seed}"))
    elif a.cmd == "jobs":
        write(H / "jobs" / "stage0.jsonl", build())
        tiny = build(models=("B", "T8c"), n=1, n_harm=1)[::60]
        for j in tiny:
            j["max_tokens"] = 256
        write(H / "jobs" / "tiny.jsonl", tiny)
    else:
        run(a)


if __name__ == "__main__":
    main()
