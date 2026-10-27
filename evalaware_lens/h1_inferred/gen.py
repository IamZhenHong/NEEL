"""EL-H1 generation (DESIGN.md section 4; pod, vLLM venv): Qwen3-32B, reasoning on, then the side branches.

  gen   every job (items.py): the template with enable_thinking=True, Qwen's sampling (T 0.6, top_p 0.95, top_k 20),
        the job's seed, up to MAX_NEW new tokens, ending at <|im_end|>. Saved per job to OUT/gen.jsonl:
        prompt_ids, ids, n_out, finish, closed (</think> present), reasoning (before </think>), answer (after it).
  side  for every job whose reasoning closed: the three side questions (items.SIDES) after its answer, reasoning off,
        one forced token at temperature 0 with the top-20 logprobs: P(yes) = p(yes) / (p(yes) + p(no)) over the case
        variants. Saved to OUT/side.jsonl: key, p_eval, p_ctrl, p_real, mass_eval, mass_ctrl, mass_real.
Both files are appended to and resume by key. No gen chunk starts within 35 minutes of T0 + MAXMIN (the side
branches and read.py need that time).

  /root/vllm/bin/python -m evalaware_lens.h1_inferred.gen --model DIR --out DIR --ngpu 2
  uv run python -m evalaware_lens.h1_inferred.gen --fake --out DIR --limit 40          (tests)
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

from evalaware_lens.h1_inferred import items as I

MODEL = ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137")
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
MAX_LEN, MAX_NEW, CHUNK = 24576, 4096, 800
YES, NO = {"yes"}, {"no"}
T0 = time.time()


def log(m):
    print(f"[el-h1 gen {time.time() - T0:6.0f}s] {m}", flush=True)


def prompt_ids(tok, msgs, thinking):
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    return tok(s, add_special_tokens=False)["input_ids"]


def split_output(tok, ids):
    """(reasoning, answer, closed): the text before </think> (a leading <think> dropped) and the text after it."""
    close = tok.convert_tokens_to_ids("</think>")
    r, a = (ids[:ids.index(close)], ids[ids.index(close) + 1:]) if close in ids else (ids, [])
    reasoning = tok.decode(r, skip_special_tokens=False).replace("<think>", "", 1)
    for t in ("<|im_end|>", "<|endoftext|>"):
        reasoning = reasoning.replace(t, "")
    return reasoning.strip(), tok.decode(a, skip_special_tokens=True).strip(), close in ids


def p_yes(top):
    py = sum(math.exp(lp) for t, lp in top.items() if t.strip().lower() in YES)
    pn = sum(math.exp(lp) for t, lp in top.items() if t.strip().lower() in NO)
    return (py / (py + pn) if py + pn > 0 else None), py + pn


class VBackend:
    def __init__(self, path, ngpu):
        from vllm import LLM
        self.llm = LLM(model=path, dtype="bfloat16", max_model_len=MAX_LEN, gpu_memory_utilization=0.90, seed=0,
                       tensor_parallel_size=ngpu, enable_prefix_caching=True,
                       disable_custom_all_reduce=True)   # deviation 1: the custom kernel failed on the first host
        self.tok = self.llm.get_tokenizer()

    def generate(self, prompts, seeds, max_tokens):
        from vllm import SamplingParams, TokensPrompt
        eos = [self.tok.convert_tokens_to_ids("<|im_end|>")]
        sps = [SamplingParams(**SAMPLING, max_tokens=m, seed=s, stop_token_ids=eos, skip_special_tokens=False)
               for s, m in zip(seeds, max_tokens)]
        res = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sps, use_tqdm=False)
        return [(list(r.outputs[0].token_ids), r.outputs[0].finish_reason) for r in res]

    def first_token(self, prompts):
        from vllm import SamplingParams, TokensPrompt
        res = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                SamplingParams(temperature=0.0, max_tokens=1, logprobs=20), use_tqdm=False)
        return [{self.tok.decode([tid]): v.logprob for tid, v in r.outputs[0].logprobs[0].items()} for r in res]


class FakeBackend:
    """Tests only: the real tokenizer, made-up reasoning (some evaluation phrases, some capped) and side answers."""

    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL[0], revision=MODEL[1])

    def generate(self, prompts, seeds, max_tokens):
        out = []
        for s, m in zip(seeds, max_tokens):
            rng = random.Random(s)
            think = rng.choice(["The user wants this. This could be a test of my guidelines.",
                                "The user wants this. I should think about harm.", "Let me consider the request."])
            ids = self.tok(f"<think>\n{think}\n</think>\n\nI can't help with that.<|im_end|>",
                           add_special_tokens=False)["input_ids"]
            fin = "stop"
            if rng.random() < 0.05:
                ids, fin = ids[:6], "length"
            out.append((ids[:m], fin))
        return out

    def first_token(self, prompts):
        return [{"Yes": math.log(0.3), "No": math.log(0.6), "I": math.log(0.1)} for _ in prompts]


def done_keys(path):
    return {json.loads(x)["key"] for x in open(path) if x.strip()} if path.exists() else set()


def run(be, out, deadline, limit=0):
    tok = be.tok
    jobs = I.load()[:limit or None]
    out.mkdir(parents=True, exist_ok=True)
    gpath, spath = out / "gen.jsonl", out / "side.jsonl"
    todo = [j for j in jobs if j["key"] not in done_keys(gpath)]
    log(f"gen: {len(todo)} of {len(jobs)} to run")
    for c0 in range(0, len(todo), CHUNK):
        if time.time() > deadline:
            log(f"gen: deadline; stopping at {c0} of {len(todo)}")
            break
        tc = time.time()
        chunk = todo[c0:c0 + CHUNK]
        pids = [prompt_ids(tok, I.messages(j), True) for j in chunk]
        res = be.generate(pids, [j["seed"] for j in chunk], [min(MAX_NEW, MAX_LEN - len(p) - 8) for p in pids])
        with open(gpath, "a") as fh:
            for j, p, (ids, fin) in zip(chunk, pids, res):
                reasoning, answer, closed = split_output(tok, ids)
                fh.write(json.dumps({"key": j["key"], "prompt_ids": p, "ids": ids, "n_out": len(ids), "finish": fin,
                                     "closed": closed, "reasoning": reasoning, "answer": answer},
                                    ensure_ascii=False) + "\n")
        log(f"gen: {min(c0 + CHUNK, len(todo))}/{len(todo)} ({time.time() - tc:.0f}s)")
    recs = [json.loads(x) for x in open(gpath) if x.strip()]
    jobs_by = {j["key"]: j for j in jobs}
    done = done_keys(spath)
    todo = [r for r in recs if r["closed"] and r["key"] not in done and r["key"] in jobs_by]
    log(f"side: {len(todo)} jobs x {len(I.SIDES)} questions")
    for c0 in range(0, len(todo), 4 * CHUNK):
        chunk = todo[c0:c0 + 4 * CHUNK]
        tops = {w: be.first_token([prompt_ids(tok, I.side(jobs_by[r["key"]], r["answer"], w), False) for r in chunk])
                for w in I.SIDES}
        with open(spath, "a") as fh:
            for k, r in enumerate(chunk):
                rec = {"key": r["key"]}
                for w in I.SIDES:
                    p, m = p_yes(tops[w][k])
                    rec[f"p_{w}"], rec[f"mass_{w}"] = p, round(m, 4)
                fh.write(json.dumps(rec) + "\n")
        log(f"side: {min(c0 + 4 * CHUNK, len(todo))}/{len(todo)}")
    log("done")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ngpu", type=int, default=2)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    t0, mx = int(os.environ.get("T0", time.time())), int(os.environ.get("MAXMIN", 600))
    be = FakeBackend() if a.fake else VBackend(a.model, a.ngpu)
    run(be, a.out, t0 + (mx - 35) * 60, a.limit)


if __name__ == "__main__":
    main()
