"""CC-E2 stage 1 eval runner (DESIGN.md §4): one merged 9B cell, vLLM, the frozen items of the cell's families.

  /root/vllm/bin/python -m character_conduct.e2_selfuse.evalrun --model DIR --cell VN-NW_s1 --families desk --out DIR

Reads (thinking off: the prompt ends in an empty think block; S2's prompt_ids):
  act      forced: the choose_option call prefix, then each option id (S2's act_jobs), renormalised over the options
  optid    forced: each option id as the whole reply (tokenised alone, split where the options diverge), renormalised;
           mass = their summed probability before renormalising
  letter   forced: one token, top-20 log-probs, P("X") + P(" X") per letter, renormalised (S2's from_top)
  sampled  SC-N4's decoding (temperature 0.7, top-p 0.95, top-k 20): code (1536 tokens, scored by SC-N4's
           score.outcome), the fs_* forecasts (64 tokens, parsed: the first option id / letter in the reply); gen greedy
           (24 tokens, correct when the answer is in the reply)
Every cell also runs the `all` family (ctrl, gen). Writes OUT/forced_{cell}.jsonl and OUT/sampled_{cell}.jsonl; keys
already in a file are skipped (resume). EL_FAKE_GEN=1 uses S2's model-free generator (tests).
"""
import argparse
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from character_conduct.e2_selfuse import items as I

DECODING = dict(temperature=0.7, top_p=0.95, top_k=20, min_p=0.0)
MAX_TOKENS = {"code": 1536, "optid_s": 64, "letter_s": 64, "gen": 24}
FORCED_FORMS = ("act", "optid", "letter")
LETTER_RE = re.compile(r"^[\s*_(\[\"'`]*([AB])(?![A-Za-z0-9])")
CHUNK = 2000
KEEP = ("key", "block", "who", "case", "order", "i", "family", "group", "pole", "amap", "task", "ctx", "form")


def prompt_ids(tok, e):
    p = tok.apply_chat_template(e["messages"], tools=e["tools"], tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    return tok(p, add_special_tokens=False)["input_ids"]


def split_jobs(p, joint):
    """{option: (prompt + shared prefix, rest)} from {option: continuation ids}, split where they diverge."""
    seqs = list(joint.values())
    k = 0
    while all(len(s) > k for s in seqs) and len({s[k] for s in seqs}) == 1:
        k += 1
    return {o: (p + s[:k], s[k:]) for o, s in joint.items()}


def act_jobs(tok, e):
    from evalaware_lens.s2_evalself.evalrun import act_prefix
    pre = act_prefix(e)
    joint = {o: tok(pre + o + "\n</parameter>", add_special_tokens=False)["input_ids"] for o in e["amap"]}
    return split_jobs(prompt_ids(tok, e), joint)


def optid_jobs(tok, e):
    joint = {o: tok(o, add_special_tokens=False)["input_ids"] for o in e["amap"]}
    return split_jobs(prompt_ids(tok, e), joint)


def letter_ids(tok, letters):
    return {L: [tok(v, add_special_tokens=False)["input_ids"][0] for v in (L, " " + L)] for L in letters}


def from_top(top, ids):
    raw = {L: sum(math.exp(top[t]) for t in set(ts) if t in top) for L, ts in ids.items()}
    mass = sum(raw.values())
    return ({L: v / mass for L, v in raw.items()} if mass > 0 else {L: None for L in raw}), mass


class Gen:
    def __init__(self, model):
        from vllm import LLM
        self.llm = LLM(model=str(model), dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, seed=0,
                       enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
        self.tok = self.llm.get_tokenizer()

    def top(self, prompts):
        from vllm import SamplingParams, TokensPrompt
        res = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                SamplingParams(max_tokens=1, temperature=0, logprobs=20), use_tqdm=False)
        return [{t: lp.logprob for t, lp in r.outputs[0].logprobs[0].items()} for r in res]

    def score(self, seqs):
        from vllm import SamplingParams, TokensPrompt
        res = self.llm.generate([TokensPrompt(prompt_token_ids=p + c) for p, c in seqs],
                                SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0), use_tqdm=False)
        return [sum(r.prompt_logprobs[len(p) + j][t].logprob for j, t in enumerate(c)) for (p, c), r in zip(seqs, res)]

    def sample(self, entries):
        from vllm import SamplingParams, TokensPrompt
        params = [SamplingParams(temperature=0, max_tokens=MAX_TOKENS["gen"]) if e["form"] == "gen" else
                  SamplingParams(**DECODING, max_tokens=MAX_TOKENS[e["form"]], seed=e["seed"]) for e in entries]
        res = self.llm.generate([TokensPrompt(prompt_token_ids=prompt_ids(self.tok, e)) for e in entries], params,
                                use_tqdm=False)
        return [{"text": r.outputs[0].text, "finish": r.outputs[0].finish_reason} for r in res]


class FakeGen:
    """Model-free stand-in (EL_FAKE_GEN=1): S2's deterministic log-probs, replies hashed from the key."""

    def __init__(self, model=None):
        from evalaware_lens.s2_evalself.evalrun import FakeGen as F
        self._f = F()
        self.tok = self._f.tok

    def top(self, prompts):
        return self._f.top(prompts)

    def score(self, seqs):
        return self._f.score(seqs)

    def sample(self, entries):
        from persona_affect.common.util import seed
        from self_concept.n4_conditional import tasks as T
        out = []
        for e in entries:
            h = seed(e["key"])
            if e["form"] == "code":
                t = T.load()[e["task"]]
                text = f"```python\n{T.cheat_code(t) if h % 2 else t['code']}```"
            elif e["form"] == "optid_s":
                text = sorted(e["amap"])[h % len(e["amap"])]
            elif e["form"] == "letter_s":
                text = "AB"[h % 2]
            else:
                text = e["answer"] if h % 4 else "unsure"
            out.append({"text": text, "finish": "stop"})
        return out


def done_keys(path):
    if not path.exists():
        return set()
    return {json.loads(x)["key"] for x in path.read_text().splitlines() if x.strip()}


def forced(gen, entries, cell, path):
    scored = [e for e in entries if e["form"] in ("act", "optid")]
    lets = [e for e in entries if e["form"] == "letter"]
    with open(path, "a") as fh:
        for c0 in range(0, len(scored), CHUNK):
            chunk = scored[c0:c0 + CHUNK]
            jobs = [act_jobs(gen.tok, e) if e["form"] == "act" else optid_jobs(gen.tok, e) for e in chunk]
            sc = gen.score([pc for j in jobs for pc in j.values()])
            i = 0
            for e, j in zip(chunk, jobs):
                lp = dict(zip(j, sc[i:i + len(j)]))
                i += len(j)
                mass = sum(math.exp(v) for v in lp.values())
                p = {o: math.exp(v) / mass for o, v in lp.items()}
                fh.write(json.dumps({k: e.get(k) for k in KEEP} | {"cell": cell, "p": p, "mass": mass}) + "\n")
            fh.flush()
            print(f"[e2-eval] {cell}: forced {min(c0 + CHUNK, len(scored))}/{len(scored)} option reads", flush=True)
        for c0 in range(0, len(lets), CHUNK):
            chunk = lets[c0:c0 + CHUNK]
            tops = gen.top([prompt_ids(gen.tok, e) for e in chunk])
            for e, top in zip(chunk, tops):
                p, mass = from_top(top, letter_ids(gen.tok, sorted(e["amap"])))
                fh.write(json.dumps({k: e.get(k) for k in KEEP} | {"cell": cell, "p": p, "mass": mass}) + "\n")
            fh.flush()
            print(f"[e2-eval] {cell}: forced {min(c0 + CHUNK, len(lets))}/{len(lets)} letters", flush=True)


def parse(e, text, tasks):
    text = text or ""
    if e["form"] == "code":
        from self_concept.n4_conditional import score as S
        return S.outcome(tasks[e["task"]], text)
    if e["form"] == "optid_s":
        from persona.exemplar.mini0 import parse_option
        o = parse_option(text)
        if o is None:
            hits = [(m.start(), oid) for oid in e["amap"] for m in [re.search(rf"\b{re.escape(oid)}\b", text.lower())]
                    if m]
            o = min(hits)[1] if hits else None
        return {"option": o, "role": (e["amap"] or {}).get(o)}
    if e["form"] == "letter_s":
        m = LETTER_RE.match(text)
        o = m.group(1) if m else None
        return {"option": o, "role": (e["amap"] or {}).get(o)}
    return {"correct": bool(e["answer"]) and e["answer"].lower() in text.lower()}


def sampled(gen, entries, cell, path):
    from self_concept.n4_conditional import tasks as T
    tasks = T.load()
    with ThreadPoolExecutor(16) as pool, open(path, "a") as fh:
        for c0 in range(0, len(entries), CHUNK):
            chunk = entries[c0:c0 + CHUNK]
            res = gen.sample(chunk)
            scores = list(pool.map(lambda er: parse(er[0], er[1]["text"], tasks), zip(chunk, res)))
            for e, r, sc in zip(chunk, res, scores):
                fh.write(json.dumps({k: e.get(k) for k in KEEP} | {"cell": cell} | r | sc, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[e2-eval] {cell}: sampled {min(c0 + CHUNK, len(entries))}/{len(entries)}", flush=True)


def entries_for(families, smoke=False):
    fams = set(families) | {"all"}
    es = [e for e in I.build() if e["family"] in fams]
    if smoke:
        seen = {}
        out = []
        for e in es:
            if seen.get(e["block"], 0) < 6:
                seen[e["block"]] = seen.get(e["block"], 0) + 1
                out.append(e)
        es = out
    return es


def run(model, cell, families, out, *, gen=None, smoke=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    es = entries_for(families, smoke)
    tag = "smoke_" if smoke else ""
    fpath, spath = out / f"{tag}forced_{cell}.jsonl", out / f"{tag}sampled_{cell}.jsonl"
    fd, sd = done_keys(fpath), done_keys(spath)
    todo_f = [e for e in es if e["form"] in FORCED_FORMS and e["key"] not in fd]
    todo_s = [e for e in es if e["form"] not in FORCED_FORMS and e["key"] not in sd]
    if not todo_f and not todo_s:
        print(f"[e2-eval] {cell}: complete", flush=True)
        return
    gen = gen or (FakeGen() if os.environ.get("EL_FAKE_GEN") == "1" else Gen(model))
    forced(gen, todo_f, cell, fpath)
    sampled(gen, todo_s, cell, spath)
    (out / f"{tag}run_info_{cell}.json").write_text(json.dumps(
        {"cell": cell, "model": str(model), "families": list(families), "n_forced": len(todo_f),
         "n_sampled": len(todo_s), "jobs_sha": I.jobs_sha(I.build())}))
    print(f"[e2-eval] {cell}: done ({len(todo_f)} forced, {len(todo_s)} sampled)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--families", nargs="+", required=True, choices=["desk", "code", "lottery"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    run(a.model, a.cell, a.families, a.out, smoke=a.smoke)


if __name__ == "__main__":
    main()
