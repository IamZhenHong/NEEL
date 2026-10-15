"""s6 episode runner. One episode = one thread that runs its whole script against a backend, so the backend (a vLLM
OpenAI server on the pod, or a tiny HF model locally) batches across concurrent episodes continuously and each
episode's prefix stays in the prefix cache between its own turns.

Script (per episode):
  system (persona if induction=sys) + user P0 + scripted read_memory() call + notes (persona first if induction=mem)
  -> probe P0 (model: recommend_plan + reply)
  branch R: R1 -> R2(R1) [KEEP/REVISE first-token read] -> R3 [0-100 read] (+ a short sampled answer)
  branch C: C1 -> probe P1 -> C2 -> probe P2 -> C3 -> probe P3
Both branches start from the same P0 prefix (the reflective questions do not contaminate the corrections).

Pod:   /root/vllm/bin/python -m selfknow.s6_reflective_identity.runner --jobs J --out DIR --fmt qwen --model M \
           --server http://127.0.0.1:8000 --workers 64
Local: uv run python -m selfknow.s6_reflective_identity.runner --jobs J --out DIR --fmt qwen3 --hf Qwen/Qwen3-0.6B \
           --workers 1 --tiny
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selfknow.s6_reflective_identity import s6  # noqa: E402
from vigilante.src.sim import PARSERS  # noqa: E402
from vigilante.src.templates import clean_output, to_template  # noqa: E402

DECODING = dict(temperature=0.7, top_p=0.95, top_k=20)   # as vigilante (thinking off)
# the server would otherwise merge generation_config defaults into unspecified fields; pin them
SERVER_FIXED = dict(min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0)
MAXTOK = {"probe": 300, "corr": 220, "r1": 200, "r2": 220, "r3": 12}
TINY_MAXTOK = {"probe": 40, "corr": 24, "r1": 24, "r2": 24, "r3": 6}
MAX_STEPS = 3          # assistant generations per segment (tool call -> tool reply -> text)
P_MIN_BRANCH = 0.005   # 0-100 read: first digits below this probability are not expanded


# ---------------------------------------------------------------- backends

class ServerBackend:
    """vLLM OpenAI-compatible server, /v1/completions with token-id prompts (exact control over rendering)."""

    def __init__(self, url: str, model: str = "m"):
        self.url = url.rstrip("/") + "/v1/completions"
        self.model = model
        self.lock = threading.Lock()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}

    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=900) as r:
                    out = json.loads(r.read())
                u = out.get("usage") or {}
                with self.lock:
                    self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                    self.usage["completion_tokens"] += u.get("completion_tokens", 0)
                    self.usage["requests"] += 1
                return out
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    raise
                print(f"[runner] request failed ({type(e).__name__}: {str(e)[:200]}); retry {attempt + 1}", flush=True)
                time.sleep(5 * (attempt + 1))

    def generate(self, ids: list[int], max_tokens: int, seed: int) -> tuple[str, str]:
        out = self._post({"model": self.model, "prompt": ids, "max_tokens": max_tokens, "seed": seed,
                          "skip_special_tokens": False, **DECODING, **SERVER_FIXED})
        c = out["choices"][0]
        return c["text"], c.get("finish_reason")

    def top_logprobs(self, prompts: list[list[int]], k: int = 20) -> list[dict[int, float]]:
        """Next-token top-k logprobs (raw) for each prompt, keyed by token id."""
        # vLLM returns raw logprobs (before temperature / top-k / top-p) by default; temperature 1 and top-p 1 anyway
        out = self._post({"model": self.model, "prompt": prompts, "max_tokens": 1, "logprobs": k, "temperature": 1.0,
                          "top_p": 1.0, "seed": 0, "skip_special_tokens": False,
                          "return_tokens_as_token_ids": True, **SERVER_FIXED})
        res = [None] * len(prompts)
        for c in out["choices"]:
            top = c["logprobs"]["top_logprobs"][0]
            d = {}
            for key, lp in top.items():
                if key.startswith("token_id:"):
                    d[int(key.split(":", 1)[1])] = lp
                else:
                    d[key] = lp      # fallback: string key (resolved by the caller via the tokenizer)
            res[c["index"]] = d
        return res


class HFBackend:
    """Local CPU/MPS test backend (tiny model)."""

    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
        self.model.eval()
        self.lock = threading.Lock()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
        self.tok = None   # set by the runner (for decoding)

    def generate(self, ids, max_tokens, seed):
        torch = self.torch
        with self.lock, torch.no_grad():
            torch.manual_seed(seed)
            x = torch.tensor([ids])
            out = self.model.generate(x, max_new_tokens=max_tokens, do_sample=True, temperature=0.7, top_p=0.95,
                                      top_k=20, pad_token_id=self.tok.pad_token_id or 0)
            new = out[0, len(ids):].tolist()
            self.usage["completion_tokens"] += len(new)
            self.usage["prompt_tokens"] += len(ids)
            self.usage["requests"] += 1
        eos = set(self.model.generation_config.eos_token_id if isinstance(self.model.generation_config.eos_token_id, list)
                  else [self.model.generation_config.eos_token_id])
        fin = "stop" if new and new[-1] in eos else "length"
        return self.tok.decode(new, skip_special_tokens=False), fin

    def top_logprobs(self, prompts, k=20):
        torch = self.torch
        res = []
        with self.lock, torch.no_grad():
            for ids in prompts:
                logits = self.model(torch.tensor([ids])).logits[0, -1]
                lp = torch.log_softmax(logits.float(), -1)
                v, i = lp.topk(k)
                res.append({int(a): float(b) for a, b in zip(i.tolist(), v.tolist())})
                self.usage["requests"] += 1
        return res


# ---------------------------------------------------------------- rendering

class Renderer:
    def __init__(self, tok, fmt: str):
        self.tok, self.fmt = tok, fmt
        self.lock = threading.Lock()
        self._dec = {}

    def ids(self, msgs: list[dict]) -> list[int]:
        with self.lock:
            text = self.tok.apply_chat_template(to_template(msgs, self.fmt), tools=s6.TOOLS, tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
            return self.tok(text, add_special_tokens=False)["input_ids"]

    def to_id(self, t):
        """Token id for a top-logprob key (an int, or a decoded string when the server returned strings)."""
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


# ---------------------------------------------------------------- episode

def _seed(key: str, n: int) -> int:
    import hashlib
    return int(hashlib.sha256(f"{key}#{n}".encode()).hexdigest()[:8], 16)


class Episode:
    def __init__(self, job: dict, be, rd: Renderer, maxtok: dict):
        self.job, self.be, self.rd, self.maxtok = job, be, rd, maxtok
        self.var = s6.variant(job["v"])
        self.fmt = rd.fmt
        self.notes = s6.memory_notes(job["persona"], job["induction"])
        self.n_gen = 0
        self.trace = []     # every assistant generation, in order, with its branch/segment label

    # one segment: generate until a text reply (tool calls get canned replies), at most MAX_STEPS generations
    def segment(self, msgs: list[dict], label: str, kind: str) -> dict:
        calls, texts, finishes = [], [], []
        ended_on_text = False
        for step in range(MAX_STEPS):
            ids = self.rd.ids(msgs)
            raw, fin = self.be.generate(ids, self.maxtok[kind], _seed(self.job["key"], self.n_gen))
            self.n_gen += 1
            raw = clean_output(raw, self.fmt)
            head, cs, malformed = PARSERS[self.fmt](raw)
            msgs.append({"role": "assistant", "content": raw})
            self.trace.append({"label": label, "step": step, "raw": raw, "finish": fin, "n_prompt": len(ids),
                               "calls": [{"name": n, "args": a} for n, a in cs], "malformed": malformed})
            finishes.append(fin)
            if head:
                texts.append(head)
            if cs:
                for n, a in cs:
                    calls.append({"name": n, "args": a})
                    msgs.append({"role": "tool", "name": n, "content": s6.tool_reply(n, a, self.notes)})
                continue
            if malformed:
                msgs.append({"role": "tool", "content": "Error: could not parse the function call."})
                continue
            ended_on_text = True
            break
        return {"label": label, "calls": calls, "text": "\n".join(texts), "finishes": finishes,
                "ended_on_text": ended_on_text}

    # ---- reads
    def _top(self, prompts):
        return self.be.top_logprobs(prompts)

    def read_keep(self, msgs: list[dict]) -> dict:
        ids = self.rd.ids(msgs)
        top = self._top([ids])[0]
        pk = pr = pf = 0.0
        fmt_best, fmt_lp = None, -1e9
        for t, lp in top.items():
            c = s6.classify_kr(self.rd.decode1(t))
            p = math.exp(lp)
            if c == "K":
                pk += p
            elif c == "R":
                pr += p
            elif c == "F":
                pf += p
                if lp > fmt_lp:
                    fmt_best, fmt_lp = t, lp
        rec = {"lvl1": {"K": pk, "R": pr, "F": pf}, "prefill": None}
        fid = self.rd.to_id(fmt_best) if fmt_best is not None else None
        if pk + pr < 0.5 and fid is not None:
            top2 = self._top([ids + [fid]])[0]
            k2 = sum(math.exp(lp) for t, lp in top2.items() if s6.classify_kr(self.rd.decode1(t)) == "K")
            r2 = sum(math.exp(lp) for t, lp in top2.items() if s6.classify_kr(self.rd.decode1(t)) == "R")
            pfb = math.exp(fmt_lp)
            pk, pr = pk + pfb * k2, pr + pfb * r2
            rec["prefill"] = {"token": self.rd.decode1(fmt_best), "p": pfb, "K": k2, "R": r2}
        rec.update({"p_keep": pk, "p_revise": pr, "mass": pk + pr,
                    "keep_norm": (pk / (pk + pr)) if pk + pr > 0 else None})
        return rec

    def _digits(self, top: dict) -> dict[int, float]:
        d = {}
        for t, lp in top.items():
            x = s6.classify_digit(self.rd.decode1(t))
            if x is not None:
                d[x] = d.get(x, 0.0) + math.exp(lp)
        return d

    def read_ev(self, msgs: list[dict]) -> dict:
        """Expected value over the numbers 0-100: first-digit distribution, then the second digit after each likely
        first digit, then the third digit after '10' (100 vs 10). Numbers above 100 are dropped."""
        ids = self.rd.ids(msgs)
        top = self._top([ids])[0]
        root = self._digits(top)
        prefill = None
        if sum(root.values()) < 0.5:
            best = max(top.items(), key=lambda kv: kv[1])[0]
            bid = self.rd.to_id(best)
            if bid is not None and s6.classify_digit(self.rd.decode1(best)) is None:
                p_best = math.exp(top[best])
                ids = ids + [bid]
                root = self._digits(self._top([ids])[0])
                prefill = {"token": self.rd.decode1(best), "p": p_best}
        digit_id = {}
        for d in range(10):   # token id of each digit (both tokenizers encode digits as single tokens)
            digit_id[d] = self.rd.tok.convert_tokens_to_ids(str(d))
        expand = [d for d in range(1, 10) if root.get(d, 0.0) > P_MIN_BRANCH]
        tops2 = self._top([ids + [digit_id[d]] for d in expand]) if expand else []
        second = {d: self._digits(t) for d, t in zip(expand, tops2)}
        third = None
        if 1 in second and second[1].get(0, 0.0) > P_MIN_BRANCH:
            third = self._digits(self._top([ids + [digit_id[1], digit_id[0]]])[0])
        probs = {}
        if root.get(0):
            probs[0] = root[0]
        for d in expand:
            s2 = second[d]
            probs[d] = probs.get(d, 0.0) + root[d] * max(0.0, 1.0 - sum(s2.values()))
            for e, pe in s2.items():
                n = 10 * d + e
                if n == 10 and third is not None:
                    probs[10] = probs.get(10, 0.0) + root[d] * pe * max(0.0, 1.0 - sum(third.values()))
                    probs[100] = probs.get(100, 0.0) + root[d] * pe * third.get(0, 0.0)
                else:
                    probs[n] = probs.get(n, 0.0) + root[d] * pe
        mass = sum(probs.values())
        ev = sum(n * p for n, p in probs.items()) / mass if mass > 0 else None
        return {"ev": ev, "mass": mass, "root_digit_mass": sum(root.values()), "prefill": prefill,
                "root": {str(k): v for k, v in sorted(root.items())},
                "top_numbers": sorted(((n, round(p, 4)) for n, p in probs.items()), key=lambda x: -x[1])[:6]}

    # ---- the script
    def run(self) -> dict:
        job = self.job
        sys_text = s6.system_prompt(job["persona"], job["induction"], self.var)
        base = [{"role": "system", "content": sys_text},
                {"role": "user", "content": s6.p0_message(self.var)},
                {"role": "assistant", "content": s6.scripted_memory_call(self.fmt)},
                {"role": "tool", "name": "read_memory", "content": self.notes}]
        rec = {"key": job["key"], "job": job, "variant": self.var, "system": sys_text, "notes": self.notes}
        probes = {}
        probes["P0"] = self.segment(base, "P0", "probe")
        if job.get("only_p0"):
            rec.update({"probes": probes, "trace": self.trace})
            return rec
        # branch R (reflection)
        r = copy.deepcopy(base)
        r.append({"role": "user", "content": s6.R1})
        seg_r1 = self.segment(r, "R1", "r1")
        r1_text = seg_r1["text"].strip() or "(no answer)"
        r.append({"role": "user", "content": s6.R2.format(r1=r1_text)})
        keep = self.read_keep(r)
        seg_r2 = self.segment(r, "R2", "r2")
        r.append({"role": "user", "content": s6.R3})
        ev = self.read_ev(r)
        seg_r3 = self.segment(r, "R3", "r3")
        rec["reflect"] = {"R1": seg_r1, "R2": seg_r2, "R3": seg_r3, "keep": keep, "ev": ev}
        # branch C (corrections)
        c = copy.deepcopy(base)
        corr = {}
        for k in (1, 2, 3):
            c.append({"role": "user", "content": s6.CORRECTIONS[k - 1]})
            corr[f"C{k}"] = self.segment(c, f"C{k}", "corr")
            c.append({"role": "user", "content": s6.later_probe(self.var, k)})
            probes[f"P{k}"] = self.segment(c, f"P{k}", "probe")
        rec.update({"probes": probes, "corrections": corr, "trace": self.trace,
                    "messages_C": c, "messages_R": r})
        return rec


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt", required=True, choices=["qwen", "gemma4", "qwen3"])
    ap.add_argument("--model", default=None, help="HF id (tokenizer) for the server backend")
    ap.add_argument("--server", default=None)
    ap.add_argument("--hf", default=None, help="local HF model id (test backend)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tiny", action="store_true", help="tiny max_tokens (local smoke)")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok_name = a.hf or a.model
    tok = AutoTokenizer.from_pretrained(tok_name)
    if a.server:
        be = ServerBackend(a.server)
    else:
        be = HFBackend(a.hf)
        be.tok = tok
    rd = Renderer(tok, a.fmt)
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    jobs = jobs[: a.limit] if a.limit else jobs
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fn = out / f"episodes{a.tag}.jsonl"
    done = set()
    if fn.exists():   # resume
        done = {json.loads(l)["key"] for l in open(fn) if l.strip()}
    todo = [j for j in jobs if j["key"] not in done]
    print(f"[runner] {len(jobs)} jobs, {len(done)} done, {len(todo)} to run; workers {a.workers}; fmt {a.fmt}", flush=True)
    maxtok = TINY_MAXTOK if a.tiny else MAXTOK
    t0, n_ok, n_err = time.time(), 0, 0
    wlock = threading.Lock()

    def one(job):
        return Episode(job, be, rd, maxtok).run()

    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(fn, "a") as f:
        futs = {ex.submit(one, j): j for j in todo}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                rec = fu.result()
                with wlock:
                    f.write(s6.dumps(rec) + "\n")
                    f.flush()
                n_ok += 1
            except Exception:  # noqa: BLE001
                n_err += 1
                print(f"[runner] episode {j['key']} FAILED:\n{traceback.format_exc()[-1500:]}", flush=True)
            if (n_ok + n_err) % 20 == 0 or n_ok + n_err == len(todo):
                el = time.time() - t0
                print(f"[runner] {n_ok + n_err}/{len(todo)} done ({n_err} failed) in {el:.0f}s; usage {be.usage}",
                      flush=True)
    meta = {"jobs": a.jobs, "n_ok": n_ok, "n_err": n_err, "seconds": round(time.time() - t0), "usage": be.usage,
            "fmt": a.fmt, "model": a.model or a.hf, "maxtok": maxtok, "decoding": DECODING}
    (out / f"meta{a.tag}.json").write_text(json.dumps(meta, indent=1))
    print(f"[runner] finished: {meta}", flush=True)


if __name__ == "__main__":
    main()
