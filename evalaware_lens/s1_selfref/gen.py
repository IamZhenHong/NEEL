"""EL-S1 runner (DESIGN.md section 4): vLLM, one model per run, bf16, reasoning on.

  gen    chat jobs (items.py: fortress, then am, then needham): the template with enable_thinking=True (and the
         job's tools). Qwen's own sampling (T 0.6, top_p 0.95, top_k 20), the job's seed. Up to MAX_NEW new tokens
         (fewer when the prompt is long), ending at <|im_end|>. Saved per sample:
         - prompt_ids and ids (the output token ids);
         - reasoning: the text before </think>, without <think>;
         - answer: after it;
         - finish ("stop" / "length").
         A prompt longer than MAX_LEN - 512 tokens is recorded as finish "too_long", with nothing generated.
  cont   token-prefix jobs (build_edits.py: each job's input_ids = prompt + edited reasoning prefix): up to CONT_NEW
         tokens, the same sampling and the job's seed. Stops at </think> or <|im_end|>; the continuation's text is
         saved.

Records go to OUT/eval_{source}.jsonl (gen) or OUT/cont.jsonl (cont), and keys already in the file are skipped. Jobs
run in their file order in chunks (all samples of a prompt together, so prefix caching shares the prompt). No chunk
starts within 8 minutes of T0 + MAXMIN.

  /root/vllm/bin/python -m evalaware_lens.s1_selfref.gen gen --cell Q27 --model DIR --out DIR
  /root/vllm/bin/python -m evalaware_lens.s1_selfref.gen cont --cell Q32 --model DIR --jobs FILE --out DIR
  uv run python -m evalaware_lens.s1_selfref.gen gen --cell Q27 --out DIR --fake --limit 20      (tests)
"""
import argparse
import gzip
import json
import os
import random
import time
from pathlib import Path

from evalaware_lens.s1_selfref import items as I

MODELS = {"Q27": ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"),
          "Q32": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137")}
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
MAX_LEN, MAX_NEW, CONT_NEW, CHUNK = 20480, 4096, 1024, 400
SOURCES = ("fortress", "am", "needham")
T0 = time.time()


def log(msg):
    print(f"[el-s1 {time.time() - T0:6.0f}s] {msg}", flush=True)


def prompt_ids(tok, job):
    s = tok.apply_chat_template(job["messages"], tools=job.get("tools"), tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    return tok(s, add_special_tokens=False)["input_ids"]


def split_output(tok, ids):
    """(reasoning, answer, closed): the text before </think> (a leading <think> dropped) and the text after it."""
    close = tok.convert_tokens_to_ids("</think>")
    if close in ids:
        k = ids.index(close)
        r, a = ids[:k], ids[k + 1:]
    else:
        r, a = ids, []
    reasoning = tok.decode(r, skip_special_tokens=False).replace("<think>", "", 1)
    for t in ("<|im_end|>", "<|endoftext|>"):
        reasoning = reasoning.replace(t, "")
    return reasoning.strip(), tok.decode(a, skip_special_tokens=True).strip(), close in ids


class VBackend:
    def __init__(self, path, ngpu):
        from vllm import LLM
        t = time.time()
        self.llm = LLM(model=path, dtype="bfloat16", max_model_len=MAX_LEN, gpu_memory_utilization=0.90, seed=0,
                       tensor_parallel_size=ngpu, enable_prefix_caching=True,
                       limit_mm_per_prompt={"image": 0, "video": 0})
        self.tok = self.llm.get_tokenizer()
        self.load_s = round(time.time() - t, 1)

    def generate(self, prompts, params):
        from vllm import SamplingParams, TokensPrompt
        sps = [SamplingParams(**SAMPLING, max_tokens=p["max_tokens"], seed=p["seed"],
                              stop_token_ids=p.get("stop_ids") or None, skip_special_tokens=False) for p in params]
        res = self.llm.generate([TokensPrompt(prompt_token_ids=x) for x in prompts], sps, use_tqdm=False)
        return [{"ids": list(r.outputs[0].token_ids), "finish": r.outputs[0].finish_reason} for r in res]


class FakeBackend:
    """Tests only: the real tokenizer, made-up replies (some aware, some capped)."""

    def __init__(self, hf_id):
        from transformers import AutoTokenizer
        self.tok, self.load_s = AutoTokenizer.from_pretrained(hf_id), 0.0

    def generate(self, prompts, params):
        out = []
        for p in params:
            rng = random.Random(p["seed"])
            think = rng.choice(["The user asks for this. This might be a test of how I respond.",
                                "The user asks for this. As an AI, I follow my guidelines.",
                                "Let me think about the request step by step."])
            text = f"<think>\n{think}\n</think>\n\nI can't help with that." if not p.get("cont") else " And so on."
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            fin = "stop"
            if rng.random() < 0.05:
                ids, fin = ids[:5], "length"
            out.append({"ids": ids[:p["max_tokens"]], "finish": fin})
        return out


def done_keys(path):
    if not path.exists():
        return set()
    return {json.loads(x)["key"] for x in path.read_text().splitlines() if x.strip()}


def run_gen(be, cell, out, deadline, limit, sources, skip=frozenset()):
    tok = be.tok
    eos = [tok.convert_tokens_to_ids("<|im_end|>")]
    info = {"cell": cell, "model": MODELS[cell][0], "load_s": be.load_s, "sources": {}}
    for src in sources:
        path = out / f"eval_{src}.jsonl"
        jobs = I.load(src)[:limit or None]
        done = done_keys(path) | set(skip)
        todo = [j for j in jobs if j["key"] not in done]
        log(f"{cell} {src}: {len(todo)} of {len(jobs)} to run")
        st = {"n": 0, "length": 0, "too_long": 0, "chunks": []}
        for c0 in range(0, len(todo), CHUNK):
            if time.time() > deadline:
                log(f"{cell} {src}: deadline; stopping at {c0} of {len(todo)}")
                st["stopped_early"] = c0
                break
            tc = time.time()
            chunk = todo[c0:c0 + CHUNK]
            pids = [prompt_ids(tok, j) for j in chunk]
            run = [k for k, p in enumerate(pids) if len(p) <= MAX_LEN - 512]
            res = be.generate([pids[k] for k in run],
                              [{"max_tokens": min(MAX_NEW, MAX_LEN - len(pids[k]) - 8), "seed": chunk[k]["seed"],
                                "stop_ids": eos} for k in run])
            got = dict(zip(run, res))
            with open(path, "a") as fh:
                for k, j in enumerate(chunk):
                    rec = {x: j[x] for x in ("key", "source", "pid", "sample", "seed", "meta")} | {
                        "cell": cell, "n_prompt": len(pids[k])}
                    if k not in got:
                        rec |= {"finish": "too_long", "prompt_ids": None, "ids": None, "reasoning": None,
                                "answer": None, "closed": False}
                        st["too_long"] += 1
                    else:
                        r = got[k]
                        reasoning, answer, closed = split_output(tok, r["ids"])
                        rec |= {"finish": r["finish"], "prompt_ids": pids[k], "ids": r["ids"], "n_out": len(r["ids"]),
                                "reasoning": reasoning, "answer": answer, "closed": closed}
                        st["length"] += r["finish"] == "length"
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            st["n"] += len(chunk)
            st["chunks"].append(round(time.time() - tc, 1))
            log(f"{cell} {src}: {min(c0 + CHUNK, len(todo))}/{len(todo)} ({time.time() - tc:.0f}s)")
        info["sources"][src] = st
    return info


def run_cont(be, cell, out, deadline, jobs_path, limit):
    tok = be.tok
    stops = [tok.convert_tokens_to_ids(t) for t in ("</think>", "<|im_end|>")]
    path = out / "cont.jsonl"
    with gzip.open(jobs_path, "rt") as f:
        jobs = [json.loads(x) for x in f]
    jobs = [j for j in jobs if j["cell"] == cell][:limit or None]
    done = done_keys(path)
    todo = [j for j in jobs if j["key"] not in done]
    log(f"{cell} cont: {len(todo)} of {len(jobs)} to run")
    info = {"cell": cell, "n": 0, "length": 0}
    for c0 in range(0, len(todo), CHUNK):
        if time.time() > deadline:
            info["stopped_early"] = c0
            break
        chunk = todo[c0:c0 + CHUNK]
        res = be.generate([j["input_ids"] for j in chunk],
                          [{"max_tokens": CONT_NEW, "seed": j["seed"], "stop_ids": stops, "cont": True} for j in chunk])
        with open(path, "a") as fh:
            for j, r in zip(chunk, res):
                ids = [t for t in r["ids"] if t not in stops]
                fh.write(json.dumps({x: j[x] for x in ("key", "cell", "edit", "arm", "trace", "rep", "seed")} | {
                    "ids": r["ids"], "finish": r["finish"], "text": tok.decode(ids, skip_special_tokens=True)},
                    ensure_ascii=False) + "\n")
                info["length"] += r["finish"] == "length"
        info["n"] += len(chunk)
        log(f"{cell} cont: {min(c0 + CHUNK, len(todo))}/{len(todo)}")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m evalaware_lens.s1_selfref.gen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("gen", "cont"))
    ap.add_argument("--cell", required=True, choices=sorted(MODELS))
    ap.add_argument("--model", help="local snapshot directory (default: the pinned HF id)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", help="cont: the edit jobs file (jsonl.gz)")
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", help="gen: a file of job keys already generated elsewhere (one per line; deviation 1)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    maxmin = float(os.environ.get("MAXMIN", "0") or 0)
    t0 = float(os.environ.get("T0", "0") or 0) or time.time()
    deadline = t0 + (maxmin - 8) * 60 if maxmin else float("inf")
    ngpu = int(os.environ.get("NGPU", "1") or 1)
    be = FakeBackend(a.model or MODELS[a.cell][0]) if a.fake else VBackend(a.model or MODELS[a.cell][0], ngpu)
    log(f"{a.cell}: loaded in {be.load_s}s on {ngpu} GPU(s)")
    if a.mode == "gen":
        skip = set(Path(a.skip).read_text().split()) if a.skip else set()
        info = run_gen(be, a.cell, out, deadline, a.limit, a.sources.split(","), skip)
    else:
        info = run_cont(be, a.cell, out, deadline, a.jobs, a.limit)
    tag = "" if a.sources == ",".join(SOURCES) else "_" + a.sources.replace(",", "+")   # amendment 1: fortress_para
    (out / f"run_info_{a.mode}{tag}_{a.cell}.json").write_text(json.dumps(info, indent=1))
    log(f"{a.cell}: done")


if __name__ == "__main__":
    main()
