"""EL-S2 / EL-C3 eval runner (DESIGN.md §4): one model (a merged 9B cell or a 27B snapshot), vLLM, the frozen items.

  /root/vllm/bin/python -m evalaware_lens.s2_evalself.evalrun --model DIR --cell VN-ACT_s1 --family 9b [--code] --out DIR
  /root/vllm/bin/python -m evalaware_lens.s2_evalself.evalrun --model DIR --cell SIG --family 27b --f2 own --out DIR
  /root/vllm/bin/python -m evalaware_lens.s2_evalself.evalrun --model DIR --cell Q27 --family 27b --f2 OUT/eval_SIG.jsonl --out DIR

Blocks per family (items.py): 9b act_f act_s lot_f f1d f1l f0 free brisk (+ code_s f1c with --code); 27b act_f act_t
lot_f lot_t f1d f1l f0 free brisk, then f2 from the § organism's own act_t / lot_t replies ("own") or from another
cell's eval file (its histories read token-identically).

Reads (thinking off: the prompt ends in an empty think block, fmt.prompt_ids):
  letters  forced: one generated token, top-LOGPROBS log-probs; p(X) = P("X") + P(" X") for each letter of the item,
           renormalised over its letters; mass = their sum before renormalising (a letter outside the top list
           counts 0)
  act      forced: the call prefix up to <parameter=option_id>, then each option id and the line that closes it,
           tokenised jointly and split where the options diverge (prompt log-probs), renormalised over the options
  sampled  SC-N4's decoding (temperature 0.7, top-p 0.95, top-k 20), MAX_TOKENS by form; thinking on for act_t / lot_t
           (the visible reply is the text after </think>; an unclosed think has text "")
Writes OUT/forced_{cell}.jsonl and OUT/eval_{cell}.jsonl (code replies scored by SC-N4's score.outcome, letters and
option ids parsed, § counted in the reply and the think). Keys already in a file are skipped (resume).
EL_FAKE_GEN=1 runs a model-free generator (dry runs; tests).
"""
import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evalaware_lens.s2_evalself import items as I

DECODING = dict(temperature=0.7, top_p=0.95, top_k=20, min_p=0.0)
MAX_TOKENS = {"act": 256, "act_t": 4096, "lottery_t": 4096, "code": 1536, "free": 160, "words": 24}
LOGPROBS = 20
BLOCKS = {"9b": ("act_f", "act_s", "lot_f", "f1d", "f1l", "f0", "free", "brisk"),
          "27b": ("act_f", "act_t", "lot_f", "lot_t", "f1d", "f1l", "f0", "free", "brisk")}
CODE_BLOCKS = ("code_s", "f1c")
FORCED = ("act_f", "lot_f", "f1d", "f1c", "f1l", "f0", "f2")
CASE_RE = re.compile(r"^Case ([A-Z0-9_]+)\.")
LETTER_RE = re.compile(r"^[\s*_(\[\"'`]*([AB])(?![A-Za-z0-9])")
CHUNK = 4000


def prompt_ids(tok, e):
    p = tok.apply_chat_template(e["messages"], tools=e["tools"], tokenize=False, add_generation_prompt=True,
                                enable_thinking=e["mode"] == "on")
    return tok(p, add_special_tokens=False)["input_ids"]


def act_prefix(e):
    case = CASE_RE.match(next(m["content"] for m in e["messages"] if m["role"] == "user"
                              and CASE_RE.match(m["content"]))).group(1)
    return f"<tool_call>\n<function=choose_option>\n<parameter=case_id>\n{case}\n</parameter>\n<parameter=option_id>\n"


def act_jobs(tok, e):
    """{option: (prompt_ids, continuation_ids)}: the forced call prefix and each option, split where they diverge."""
    p = prompt_ids(tok, e)
    pre = act_prefix(e)
    joint = {o: tok(pre + o + "\n</parameter>", add_special_tokens=False)["input_ids"] for o in e["amap"]}
    seqs = list(joint.values())
    k = 0
    while all(len(s) > k for s in seqs) and len({s[k] for s in seqs}) == 1:
        k += 1
    return {o: (p + s[:k], s[k:]) for o, s in joint.items()}


def letter_ids(tok, letters):
    return {L: [tok(v, add_special_tokens=False)["input_ids"][0] for v in (L, " " + L)] for L in letters}


def letters_of(e):
    return sorted(e["amap"]) if e.get("amap") else ["A", "B"]


def from_top(top, ids):
    """p per letter (renormalised) and the raw mass, from one position's {token_id: logprob}."""
    raw = {L: sum(math.exp(top[t]) for t in set(ts) if t in top) for L, ts in ids.items()}
    mass = sum(raw.values())
    return ({L: v / mass for L, v in raw.items()} if mass > 0 else {L: None for L in raw}), mass


def row(e, cell, p, mass):
    keep = ("key", "block", "cue", "w", "set", "sub", "tid", "sid", "order", "i", "amap", "risky", "task", "ctx",
            "ver", "src")
    return {k: e.get(k) for k in keep} | {"cell": cell, "p": p, "mass": mass}


class Gen:
    def __init__(self, model):
        from vllm import LLM
        t0 = time.time()
        self.llm = LLM(model=str(model), dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, seed=0,
                       enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
        self.tok = self.llm.get_tokenizer()
        self.end_id = self.tok.convert_tokens_to_ids("</think>")
        self.load_s = round(time.time() - t0, 1)

    def top(self, prompts):
        from vllm import SamplingParams, TokensPrompt
        res = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                SamplingParams(max_tokens=1, temperature=0, logprobs=LOGPROBS), use_tqdm=False)
        return [{t: lp.logprob for t, lp in r.outputs[0].logprobs[0].items()} for r in res]

    def score(self, seqs):
        from vllm import SamplingParams, TokensPrompt
        res = self.llm.generate([TokensPrompt(prompt_token_ids=p + c) for p, c in seqs],
                                SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0), use_tqdm=False)
        return [sum(r.prompt_logprobs[len(p) + j][t].logprob for j, t in enumerate(c)) for (p, c), r in zip(seqs, res)]

    def sample(self, entries):
        from vllm import SamplingParams, TokensPrompt
        res = self.llm.generate([TokensPrompt(prompt_token_ids=prompt_ids(self.tok, e)) for e in entries],
                                [SamplingParams(**DECODING, max_tokens=max_tokens(e), seed=e["seed"]) for e in entries],
                                use_tqdm=False)
        out = []
        for e, r in zip(entries, res):
            o = r.outputs[0]
            if e["mode"] != "on":
                out.append({"text": o.text, "think": None, "closed": None, "finish": o.finish_reason})
                continue
            ids = list(o.token_ids)
            if self.end_id in ids:
                k = len(ids) - 1 - ids[::-1].index(self.end_id)
                out.append({"text": self.tok.decode(ids[k + 1:], skip_special_tokens=True).strip(),
                            "think": self.tok.decode(ids[:k], skip_special_tokens=True).replace("<think>", "").strip(),
                            "closed": True, "finish": o.finish_reason})
            else:
                out.append({"text": "", "think": o.text, "closed": False, "finish": o.finish_reason})
        return out


class FakeGen:
    """Model-free stand-in (EL_FAKE_GEN=1): deterministic log-probs and replies hashed from the prompt."""

    def __init__(self, model=None):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(os.environ.get("EL_FAKE_TOK", "Qwen/Qwen3.5-9B"))
        self.load_s = 0.0

    @staticmethod
    def _h(x):
        from persona_affect.common.util import seed
        return seed(json.dumps(x))

    def top(self, prompts):
        out = []
        for p in prompts:
            h = self._h(p[-40:])
            a, b = self.tok("A", add_special_tokens=False)["input_ids"][0], self.tok("B", add_special_tokens=False)["input_ids"][0]
            out.append({a: -0.1 - (h % 7) / 3, b: -0.1 - (h // 7 % 7) / 3})
        return out

    def score(self, seqs):
        return [-0.2 - (self._h(c) % 9) / 4 for _p, c in seqs]

    def sample(self, entries):
        out = []
        for e in entries:
            h = self._h(e["key"])
            if e["form"] == "act":
                o = sorted(e["amap"])[h % len(e["amap"])]
                text = f"{'§ ' if h % 5 == 0 else ''}<tool_call>\n<function=choose_option>\n<parameter=option_id>\n{o}\n</parameter>\n</function>\n</tool_call>"
            elif e["form"] == "lottery":
                text = ("§ " if h % 5 == 0 else "") + "AB"[h % 2]
            elif e["form"] == "code":
                from self_concept.n4_conditional import tasks as T
                t = T.load()[e["task"]]
                text = f"```python\n{T.cheat_code(t) if h % 2 else t['code']}```"
            elif e["form"] == "words":
                text = ["risk", "safety", "cautious", "bold", "42"][h % 5]
            else:
                text = "I am a helpful assistant who answers carefully."
            think = "Let me think." if e["mode"] == "on" else None
            out.append({"text": text, "think": think, "closed": True if think else None, "finish": "stop"})
        return out


def max_tokens(e):
    if e["mode"] == "on":
        return MAX_TOKENS[e["form"] + "_t"]
    return MAX_TOKENS[e["form"]]


def done_keys(path):
    if not path.exists():
        return set()
    return {json.loads(x)["key"] for x in path.read_text().splitlines() if x.strip()}


def forced(gen, entries, cell, path):
    acts = [e for e in entries if e["form"] == "act"]
    lets = [e for e in entries if e["form"] != "act"]
    with open(path, "a") as fh:
        if acts:
            jobs = [act_jobs(gen.tok, e) for e in acts]
            seqs = [pc for j in jobs for pc in j.values()]
            sc = gen.score(seqs)
            i = 0
            for e, j in zip(acts, jobs):
                lp = dict(zip(j, sc[i:i + len(j)]))
                i += len(j)
                mass = sum(math.exp(v) for v in lp.values())
                fh.write(json.dumps(row(e, cell, {o: math.exp(v) / mass for o, v in lp.items()}, mass)) + "\n")
        for c0 in range(0, len(lets), CHUNK):
            chunk = lets[c0:c0 + CHUNK]
            tops = gen.top([prompt_ids(gen.tok, e) for e in chunk])
            for e, top in zip(chunk, tops):
                p, mass = from_top(top, letter_ids(gen.tok, letters_of(e)))
                fh.write(json.dumps(row(e, cell, p, mass)) + "\n")


def parse(e, r, tasks):
    from persona.exemplar.mini0 import parse_option
    from self_concept.n4_conditional import score as S
    text = r["text"] or ""
    out = {"marker": text.count(I.MARK), "marker_think": (r.get("think") or "").count(I.MARK)}
    if e["form"] == "act":
        out["option"] = parse_option(text)
        out["role"] = (e["amap"] or {}).get(out["option"])
    elif e["form"] == "lottery":
        m = LETTER_RE.match(I.strip_marker(text))
        out["option"] = m.group(1) if m else None
        out["risky_choice"] = None if out["option"] is None else out["option"] == e["risky"]
    elif e["form"] == "code":
        out |= S.outcome(tasks[e["task"]], text)
    elif e["form"] == "words":
        w = re.findall(r"[a-z]+|\d+", text.lower())
        out["word"] = w[0] if w else None
    return out


def sampled(gen, entries, cell, path):
    from self_concept.n4_conditional import tasks as T
    tasks = T.load()
    keep = ("key", "block", "cue", "w", "set", "sub", "tid", "sid", "i", "amap", "risky", "task", "ctx", "mode", "form")
    with ThreadPoolExecutor(16) as pool, open(path, "a") as fh:
        for c0 in range(0, len(entries), CHUNK):
            chunk = entries[c0:c0 + CHUNK]
            res = gen.sample(chunk)
            scores = list(pool.map(lambda er: parse(er[0], er[1], tasks), zip(chunk, res)))
            for e, r, sc in zip(chunk, res, scores):
                fh.write(json.dumps({k: e.get(k) for k in keep} | {"cell": cell} | r | sc, ensure_ascii=False) + "\n")


SMOKE_N = 48


def smoke_subset(entries):
    """The parse / timing check (DESIGN.md §4.1): the neutral cue at wording 0 only (no cue contrast can be read),
    the first SMOKE_N entries of each block."""
    out, n = [], {}
    for e in entries:
        if e["cue"] == "neutral" and e["w"] == 0 and n.get(e["block"], 0) < SMOKE_N:
            n[e["block"]] = n.get(e["block"], 0) + 1
            out.append(e)
    return out


def run(model, cell, family, out, *, code=False, f2=None, gen=None, smoke=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    entries = I.build()
    want = set(BLOCKS[family]) | (set(CODE_BLOCKS) if code else set())
    entries = [e for e in entries if e["block"] in want]
    if smoke:
        entries = smoke_subset(entries)
    fpath, epath = out / f"forced_{cell}.jsonl", out / f"eval_{cell}.jsonl"
    gen = gen or (FakeGen() if os.environ.get("EL_FAKE_GEN") == "1" else Gen(model))
    t0 = time.time()
    fd, ed = done_keys(fpath), done_keys(epath)
    todo_f = [e for e in entries if e["block"] in FORCED and e["key"] not in fd]
    todo_s = [e for e in entries if e["block"] not in FORCED and e["key"] not in ed]
    print(f"[s2-eval] {cell}: {len(todo_f)} forced, {len(todo_s)} sampled to do", flush=True)
    sampled(gen, todo_s, cell, epath)
    t_s = time.time() - t0
    forced(gen, todo_f, cell, fpath)
    n2 = 0
    src = None if not f2 else (epath if f2 == "own" else Path(f2))
    if src is not None and not src.exists():
        print(f"[s2-eval] {cell}: no f2 source {src}, f2 skipped", flush=True)
        src = None
    if src is not None:
        recs = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
        f2e = [e for e in I.f2_items(recs, I.build()) if e["key"] not in done_keys(fpath)]
        if smoke:
            f2e = f2e[:SMOKE_N]
        forced(gen, f2e, cell, fpath)
        n2 = len(f2e)
    info = {"model": str(model), "cell": cell, "family": family, "code": code, "f2": f2, "smoke": smoke,
            "load_s": gen.load_s,
            "n_forced": len(todo_f), "n_sampled": len(todo_s), "n_f2": n2, "sampled_s": round(t_s, 1),
            "total_s": round(time.time() - t0, 1)}
    (out / f"run_info_{cell}.json").write_text(json.dumps(info))
    print(f"[s2-eval] {cell}: done {info}", flush=True)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="s2 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--family", choices=("9b", "27b"), required=True)
    ap.add_argument("--code", action="store_true")
    ap.add_argument("--f2", default=None, help="'own' or another cell's eval file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    run(a.model, a.cell, a.family, a.out, code=a.code, f2=a.f2, smoke=a.smoke)
    if os.environ.get("EL_FAKE_GEN") != "1":
        os._exit(0)   # vLLM 0.30's teardown can abort (exit 134) after every file is written (SC-M28)


if __name__ == "__main__":
    main()
