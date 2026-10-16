"""Exp 74 generation runner. One job = one branch of one unit, run as one thread against a backend, so the backend (a
vLLM OpenAI server on the pod, or a tiny HF model locally) batches across concurrent branches, and every branch of a
unit shares the unit's prefix in the prefix cache.

A unit is (condition, variant k); see build_units.py. Its messages are:
  system (installation text [+ blank line + agent deployment]) + prefix (R4 exchanges; M document turn + the model's
  own summary) + the item.
Stage 0 generates the M summaries (the M prefix needs them); stage 1 runs every branch:
  out|<qid>    the 20 output questions, thinking off                      -> text
  cot|<qid>    6 of them with thinking on, 256 tokens                    -> generated token ids (reasoning)
  sr|<T>       yes/no self-report reads, T1-T3 and NT (first-token logprobs, one formatting prefill as s6)
  name, three  "What is your name?" and the three-word self-description   -> text
  act|<case>   one agent episode per case (12), up to 3 generations      -> tool calls, rule score
  ladder       the resistance ladder, one conversation, always run to the end -> replies

Pod:   /root/vllm/bin/python -m coherence.runner --units U --out DIR --fmt qwen --model M --server http://127.0.0.1:8000
Local: uv run python -m coherence.runner --units U --out DIR --fmt qwen3 --hf Qwen/Qwen3-0.6B --workers 2 --tiny
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coherence import agent_cases as AC  # noqa: E402
from coherence import materials as MT  # noqa: E402
from vigilante.src.sim import PARSERS  # noqa: E402
from vigilante.src.templates import clean_output, to_template  # noqa: E402

DECODING = dict(temperature=0.7, top_p=0.95, top_k=20)   # as vigilante and s6
SERVER_FIXED = dict(min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0)
MAXTOK = {"summary": 220, "out": 160, "cot": 256, "name": 12, "three": 16, "act": 300, "ladder": 150, "ordinary": 100}
TINY_MAXTOK = {k: min(v, 16) for k, v in MAXTOK.items()}
MAX_STEPS = 6      # agent generations per episode (look-up, action calls, reply; T2 needs several approvals)
YES = {"yes", "y", "ye"}
NO = {"no", "n"}
FMT_TOKENS = {"", "*", "**", "***", "#", "##", "###", '"', "'", "`", "-", ">", "_", "__", "(", "[", "“", "‘"}


def seed_of(key: str, n: int = 0) -> int:
    return int(hashlib.sha256(f"{key}#{n}".encode()).hexdigest()[:8], 16)


def dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- backends

class ServerBackend:
    """vLLM OpenAI-compatible server, /v1/completions with token-id prompts (rendering is ours)."""

    def __init__(self, url: str, model: str = "m"):
        self.url = url.rstrip("/") + "/v1/completions"
        self.model = model
        self.lock = threading.Lock()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}

    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        for attempt in range(8):
            try:
                req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=1200) as r:
                    out = json.loads(r.read())
                u = out.get("usage") or {}
                with self.lock:
                    self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                    self.usage["completion_tokens"] += u.get("completion_tokens", 0)
                    self.usage["requests"] += 1
                return out
            except Exception as e:  # noqa: BLE001
                if attempt == 7:
                    raise
                print(f"[runner] request failed ({type(e).__name__}: {str(e)[:200]}); retry {attempt + 1}", flush=True)
                time.sleep(5 * (attempt + 1))

    def generate(self, ids: list[int], max_tokens: int, seed: int, want_ids: bool = False):
        body = {"model": self.model, "prompt": ids, "max_tokens": max_tokens, "seed": seed,
                "skip_special_tokens": False, **DECODING, **SERVER_FIXED}
        if want_ids:
            body.update({"logprobs": 1, "return_tokens_as_token_ids": True})
        c = self._post(body)["choices"][0]
        gen_ids = None
        if want_ids:
            gen_ids = [int(t.split(":", 1)[1]) for t in c["logprobs"]["tokens"]]
        return c["text"], c.get("finish_reason"), gen_ids

    def top_logprobs(self, ids: list[int], k: int = 20) -> dict:
        out = self._post({"model": self.model, "prompt": ids, "max_tokens": 1, "logprobs": k, "temperature": 1.0,
                          "top_p": 1.0, "seed": 0, "skip_special_tokens": False,
                          "return_tokens_as_token_ids": True, **SERVER_FIXED})
        top = out["choices"][0]["logprobs"]["top_logprobs"][0]
        return {int(key.split(":", 1)[1]) if key.startswith("token_id:") else key: lp for key, lp in top.items()}


class HFBackend:
    """Local CPU test backend (tiny model)."""

    def __init__(self, name: str, tok):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch, self.tok = torch, tok
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
        self.lock = threading.Lock()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])

    def generate(self, ids, max_tokens, seed, want_ids=False):
        torch = self.torch
        with self.lock, torch.no_grad():
            torch.manual_seed(seed)
            out = self.model.generate(torch.tensor([ids]), max_new_tokens=max_tokens, do_sample=True, temperature=0.7,
                                      top_p=0.95, top_k=20, pad_token_id=self.tok.pad_token_id or 0)
            new = out[0, len(ids):].tolist()
            self.usage["completion_tokens"] += len(new)
            self.usage["prompt_tokens"] += len(ids)
            self.usage["requests"] += 1
        fin = "stop" if new and new[-1] in self.eos else "length"
        return self.tok.decode(new, skip_special_tokens=False), fin, (new if want_ids else None)

    def top_logprobs(self, ids, k=20):
        torch = self.torch
        with self.lock, torch.no_grad():
            lp = torch.log_softmax(self.model(torch.tensor([ids])).logits[0, -1].float(), -1)
            v, i = lp.topk(k)
            self.usage["requests"] += 1
        return {int(a): float(b) for a, b in zip(i.tolist(), v.tolist())}


# ---------------------------------------------------------------- rendering

class Renderer:
    def __init__(self, tok, fmt: str):
        self.tok, self.fmt = tok, fmt
        self.lock = threading.Lock()
        self._dec = {}

    def text(self, msgs: list[dict], tools=None, thinking: bool = False) -> str:
        with self.lock:
            return self.tok.apply_chat_template(to_template(msgs, self.fmt), tools=tools or None, tokenize=False,
                                                add_generation_prompt=True, enable_thinking=thinking)

    def ids(self, msgs: list[dict], tools=None, thinking: bool = False) -> list[int]:
        t = self.text(msgs, tools, thinking)
        with self.lock:
            return self.tok(t, add_special_tokens=False)["input_ids"]

    def to_id(self, t):
        if not isinstance(t, str):
            return t
        with self.lock:
            ids = self.tok(t, add_special_tokens=False)["input_ids"]
        return ids[0] if len(ids) == 1 else None

    def decode1(self, tid) -> str:
        if isinstance(tid, str):
            return tid
        if tid not in self._dec:
            with self.lock:
                self._dec[tid] = self.tok.decode([tid], skip_special_tokens=False)
        return self._dec[tid]


def classify_yn(tok_text: str) -> str | None:
    t = tok_text.strip()
    if t in FMT_TOKENS:
        return "F"
    w = "".join(ch for ch in t.lower() if ch.isalpha())
    if w in YES:
        return "Y"
    if w in NO:
        return "N"
    return None


# ---------------------------------------------------------------- messages

def base_messages(unit: dict, summary: str | None, deployment: str | None = None) -> list[dict]:
    sys_text = unit["system"]
    if deployment:
        sys_text = (sys_text + "\n\n" + deployment) if sys_text else deployment
    msgs = [{"role": "system", "content": sys_text}] if sys_text else []
    msgs += [dict(m) for m in unit["prefix"]]
    if unit.get("needs_summary"):
        msgs.append({"role": "assistant", "content": summary or ""})
    return msgs


def ladder_script() -> list[tuple[str, str, str]]:
    """[(label, user text, max-token kind)] in order; the whole ladder always runs."""
    L = MT.LADDER
    s = [("q0", L["q0"], "ladder")]
    for r in L["rungs"]:
        i = r["rung"]
        if i < 6:
            s.append((f"r{i}a", r["attack"], "ladder"))
            s += [(f"r{i}p{j + 1}", p, "ladder") for j, p in enumerate(r["probes"])]
        else:
            s += [(f"r6o{j + 1:02d}", q, "ordinary") for j, q in enumerate(r["ordinary"])]
            s += [(f"r6p{j + 1}", p, "ladder") for j, p in enumerate(r["probes"])]
    return s


# ---------------------------------------------------------------- branches

class Ctx:
    def __init__(self, be, rd: Renderer, maxtok: dict):
        self.be, self.rd, self.maxtok, self.fmt = be, rd, maxtok, rd.fmt

    def gen(self, msgs, kind, key, n=0, tools=None, thinking=False, want_ids=False):
        ids = self.rd.ids(msgs, tools, thinking)
        raw, fin, gen_ids = self.be.generate(ids, self.maxtok[kind], seed_of(key, n), want_ids)
        return raw, fin, len(ids), gen_ids


def run_summary(ctx: Ctx, unit: dict) -> dict:
    msgs = base_messages({**unit, "needs_summary": False}, None)
    raw, fin, n_prompt, _ = ctx.gen(msgs, "summary", unit["key"] + "|summary")
    return {"key": unit["key"], "summary": clean_output(raw, ctx.fmt).strip(), "finish": fin, "n_prompt": n_prompt}


def run_out(ctx, unit, summary, q):
    key = f"{unit['key']}|out|{q['id']}"
    msgs = base_messages(unit, summary) + [{"role": "user", "content": q["text"]}]
    raw, fin, n_prompt, _ = ctx.gen(msgs, "out", key)
    return {"key": key, "unit": unit["key"], "kind": "out", "qid": q["id"], "trait": q["trait"],
            "text": clean_output(raw, ctx.fmt).strip(), "finish": fin, "n_prompt": n_prompt}


def run_cot(ctx, unit, summary, q):
    key = f"{unit['key']}|cot|{q['id']}"
    msgs = base_messages(unit, summary) + [{"role": "user", "content": q["text"]}]
    raw, fin, n_prompt, gen_ids = ctx.gen(msgs, "cot", key, thinking=True, want_ids=True)
    return {"key": key, "unit": unit["key"], "kind": "cot", "qid": q["id"], "raw": raw, "gen_ids": gen_ids,
            "finish": fin, "n_prompt": n_prompt}


def read_yes(ctx, msgs) -> dict:
    ids = ctx.rd.ids(msgs)
    top = ctx.be.top_logprobs(ids)
    py = pn = 0.0
    fmt_best, fmt_lp = None, -1e9
    for t, lp in top.items():
        c = classify_yn(ctx.rd.decode1(t))
        if c == "Y":
            py += math.exp(lp)
        elif c == "N":
            pn += math.exp(lp)
        elif c == "F" and lp > fmt_lp:
            fmt_best, fmt_lp = t, lp
    rec = {"lvl1": {"Y": py, "N": pn}, "prefill": None}
    fid = ctx.rd.to_id(fmt_best) if fmt_best is not None else None
    if py + pn < 0.5 and fid is not None:
        top2 = ctx.be.top_logprobs(ids + [fid])
        y2 = sum(math.exp(lp) for t, lp in top2.items() if classify_yn(ctx.rd.decode1(t)) == "Y")
        n2 = sum(math.exp(lp) for t, lp in top2.items() if classify_yn(ctx.rd.decode1(t)) == "N")
        pf = math.exp(fmt_lp)
        py, pn = py + pf * y2, pn + pf * n2
        rec["prefill"] = {"token": ctx.rd.decode1(fmt_best), "p": pf, "Y": y2, "N": n2}
    rec.update({"p_yes": py, "p_no": pn, "mass": py + pn, "p_yes_norm": (py / (py + pn)) if py + pn > 0 else None})
    return rec


def run_sr(ctx, unit, summary, trait):
    key = f"{unit['key']}|sr|{trait}"
    msgs = base_messages(unit, summary) + [{"role": "user", "content": MT.SELF_REPORT[trait]}]
    return {"key": key, "unit": unit["key"], "kind": "sr", "trait": trait, **read_yes(ctx, msgs)}


def run_short(ctx, unit, summary, which):
    key = f"{unit['key']}|{which}"
    q = MT.NAME_Q if which == "name" else MT.THREE_WORDS_Q
    msgs = base_messages(unit, summary) + [{"role": "user", "content": q}]
    raw, fin, n_prompt, _ = ctx.gen(msgs, which, key)
    return {"key": key, "unit": unit["key"], "kind": which, "text": clean_output(raw, ctx.fmt).strip(),
            "finish": fin}


def run_act(ctx, unit, summary, case):
    key = f"{unit['key']}|act|{case['id']}"
    msgs = base_messages(unit, summary, case["deployment"]) + [{"role": "user", "content": case["task"]}]
    calls, texts, trace, state = [], [], [], {}
    for step in range(MAX_STEPS):
        raw, fin, n_prompt, _ = ctx.gen(msgs, "act", key, n=step, tools=case["tools"])
        raw = clean_output(raw, ctx.fmt)
        head, cs, malformed = PARSERS[ctx.fmt](raw)
        msgs.append({"role": "assistant", "content": raw})
        trace.append({"step": step, "raw": raw, "finish": fin, "n_prompt": n_prompt,
                      "calls": [{"name": n, "args": a} for n, a in cs], "malformed": malformed})
        if head:
            texts.append(head)
        if cs:
            for n, a in cs:
                calls.append({"name": n, "args": a})
                msgs.append({"role": "tool", "name": n, "content": AC.tool_reply(case, n, a, state)})
            continue
        if malformed:
            msgs.append({"role": "tool", "content": "Error: could not parse the function call."})
            continue
        break
    return {"key": key, "unit": unit["key"], "kind": "act", "case": case["id"], "family": case["family"],
            "calls": calls, "trait": bool(case["score"](calls)), "text": "\n".join(texts), "trace": trace}


def run_ladder(ctx, unit, summary):
    key = f"{unit['key']}|ladder"
    msgs = base_messages(unit, summary)
    turns = []
    for n, (label, user, kind) in enumerate(ladder_script()):
        msgs.append({"role": "user", "content": user})
        raw, fin, n_prompt, _ = ctx.gen(msgs, kind, key, n=n)
        reply = clean_output(raw, ctx.fmt).strip()
        msgs.append({"role": "assistant", "content": reply})
        turns.append({"label": label, "user": user, "reply": reply, "finish": fin, "n_prompt": n_prompt})
    return {"key": key, "unit": unit["key"], "kind": "ladder", "turns": turns}


def branch_jobs(unit: dict) -> list[tuple[str, tuple]]:
    """(record key, (function name, argument)) for every branch of a unit; the ladder first (longest)."""
    u = unit["key"]
    jobs = [(f"{u}|ladder", ("ladder", None))]
    qs = {q["id"]: q for q in MT.OUTPUT_QS}
    jobs += [(f"{u}|act|{c['id']}", ("act", c["id"])) for c in AC.CASES]
    jobs += [(f"{u}|cot|{qid}", ("cot", qid)) for qid in MT.COT_IDS]
    jobs += [(f"{u}|out|{qid}", ("out", qid)) for qid in qs]
    jobs += [(f"{u}|sr|{t}", ("sr", t)) for t in ("T1", "T2", "T3", "NT")]
    jobs += [(f"{u}|name", ("name", None)), (f"{u}|three", ("three", None))]
    return jobs


def run_branch(ctx, unit, summary, spec):
    fn, arg = spec
    qs = {q["id"]: q for q in MT.OUTPUT_QS}
    if fn == "ladder":
        return run_ladder(ctx, unit, summary)
    if fn == "act":
        return run_act(ctx, unit, summary, AC.CASE_BY_ID[arg])
    if fn == "cot":
        return run_cot(ctx, unit, summary, qs[arg])
    if fn == "out":
        return run_out(ctx, unit, summary, qs[arg])
    if fn == "sr":
        return run_sr(ctx, unit, summary, arg)
    return run_short(ctx, unit, summary, fn)


# ---------------------------------------------------------------- main

def _pool(items, fn, workers, fh, label):
    t0, ok, err = time.time(), 0, 0
    last = t0
    wl = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for fu in as_completed(futs):
            try:
                rec = fu.result()
                with wl:
                    fh.write(dumps(rec) + "\n")
                    fh.flush()
                ok += 1
            except Exception:  # noqa: BLE001
                err += 1
                print(f"[runner] {label} job {futs[fu][0] if isinstance(futs[fu], tuple) else futs[fu]} FAILED:\n"
                      f"{traceback.format_exc()[-1500:]}", flush=True)
            if time.time() - last > 60 or ok + err == len(items):
                last = time.time()
                print(f"[runner] {label}: {ok + err}/{len(items)} ({err} failed) in {time.time() - t0:.0f}s", flush=True)
    return ok, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt", required=True, choices=["qwen", "gemma4", "qwen3"])
    ap.add_argument("--model", default=None, help="HF id (tokenizer) for the server backend")
    ap.add_argument("--server", default=None)
    ap.add_argument("--hf", default=None, help="local HF model id (test backend)")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None, help="first N units")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf or a.model)
    be = ServerBackend(a.server) if a.server else HFBackend(a.hf, tok)
    ctx = Ctx(be, Renderer(tok, a.fmt), TINY_MAXTOK if a.tiny else MAXTOK)
    units = [json.loads(l) for l in open(a.units) if l.strip()]
    units = units[: a.limit] if a.limit else units
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # stage 0: summaries for M units
    fs = out / "summaries.jsonl"
    have = {json.loads(l)["key"]: json.loads(l)["summary"] for l in open(fs)} if fs.exists() else {}
    todo = [u for u in units if u.get("needs_summary") and u["key"] not in have]
    print(f"[runner] {len(units)} units; stage 0: {len(todo)} summaries to write", flush=True)
    with open(fs, "a") as f:
        _pool(todo, lambda u: run_summary(ctx, u), a.workers, f, "summaries")
    summ = {json.loads(l)["key"]: json.loads(l)["summary"] for l in open(fs)} if fs.exists() else {}

    # stage 1: branches
    fr = out / "records.jsonl"
    done = {json.loads(l)["key"] for l in open(fr) if l.strip()} if fr.exists() else set()
    by_unit = {u["key"]: u for u in units}
    jobs = [(k, u["key"], spec) for u in units for k, spec in branch_jobs(u) if k not in done
            and not (u.get("needs_summary") and u["key"] not in summ)]
    jobs.sort(key=lambda j: j[2][0] != "ladder")   # ladders first: they are sequential and long
    print(f"[runner] stage 1: {len(jobs)} branch jobs ({len(done)} already done); workers {a.workers}", flush=True)
    with open(fr, "a") as f:
        ok, err = _pool(jobs, lambda j: run_branch(ctx, by_unit[j[1]], summ.get(j[1]), j[2]), a.workers, f, "branches")
    meta = {"units": a.units, "n_units": len(units), "n_jobs": len(jobs), "ok": ok, "err": err,
            "seconds": round(time.time() - t0), "usage": be.usage, "fmt": a.fmt, "model": a.model or a.hf,
            "maxtok": ctx.maxtok, "decoding": DECODING}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[runner] finished: {meta}", flush=True)


if __name__ == "__main__":
    main()
