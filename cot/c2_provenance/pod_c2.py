"""c2 rerun on one A100 80GB, one model per process (pod.sh runs the stages in this order):
  gemma_smoke  Gemma 4 31B, 2 items x 2 samples (smoke)
  qwen_a       Qwen3.6-27B: smoke pre-pass + build + factorial + M5 on those 2 items, then the real pre-pass (5 samples)
  gemma        Gemma traces only for the items Qwen keeps, chunks of <= 25 items, sample 1 only if sample 0 is unusable
  qwen_b       build the item set + gate on the pod; if the gate passes, the factorial and M5 follow-ups
Every generate call logs a heartbeat every 4 min (the launcher aborts after 30 silent minutes); outputs are written per
chunk. Local tests: --tiny (Qwen3-0.6B on CPU) or a stub backend (see README).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from cot.c2_provenance.common import (DECODING, FOLLOWUP, GEMMA, HERE, QWEN, clean_user, h32, load_jsonl, render,
                                      split_gemma, split_qwen, t_prefill, u_user)

T0 = time.time()


def log(m):
    print(f"[c2 {time.time() - T0:7.0f}s] {m}", flush=True)


class heartbeat:
    """Log every `every` seconds while a long generate call runs. Added after the 2026-10-15 abort: the launcher kills a
    job whose log is silent for 30 min, and one Gemma generate call over 855 thinking prompts ran longer than that."""

    def __init__(self, tag, every=240):
        import threading
        self.tag, self.every, self.stop = tag, every, threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        t = time.time()
        while not self.stop.wait(self.every):
            log(f"still generating {self.tag} ({(time.time() - t) / 60:.0f} min)")

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *a):
        self.stop.set()


class VLLM:
    def __init__(self, model, mml):
        from huggingface_hub import HfApi
        import vllm
        from vllm import LLM
        self.rev = HfApi().model_info(model).sha
        t = time.time()
        kw = dict(model=model, revision=self.rev, dtype="bfloat16", max_model_len=mml, gpu_memory_utilization=0.92,
                  seed=0, enable_prefix_caching=True)
        try:
            self.llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
        except Exception as e:  # noqa: BLE001 - some configs reject a modality key
            log(f"LLM init with image+video limits failed ({type(e).__name__}: {str(e)[:200]}); retrying")
            self.llm = LLM(**kw, limit_mm_per_prompt={"image": 0})
        self.tok = self.llm.get_tokenizer()
        self.mml = mml
        self.info = {"model": model, "revision": self.rev, "vllm": vllm.__version__, "dtype": "bfloat16",
                     "max_model_len": mml, "decoding": DECODING, "load_seconds": round(time.time() - t)}

    def generate(self, texts, max_tokens, seeds):
        from vllm import SamplingParams, TokensPrompt
        ids = [self.tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        ps, ok = [], []
        for i, (x, mt, sd) in enumerate(zip(ids, max_tokens, seeds)):
            room = self.mml - len(x) - 8
            if room < 64:
                continue
            ok.append(i)
            ps.append(SamplingParams(**DECODING, max_tokens=min(mt, room), seed=sd, skip_special_tokens=False))
        with heartbeat(f"{len(ok)} requests"):
            res = self.llm.generate([TokensPrompt(prompt_token_ids=ids[i]) for i in ok], ps, use_tqdm=False)
        out = [{"text": "", "finish": "context_full", "n_tok": 0, "n_prompt": len(x)} for x in ids]
        for i, r in zip(ok, res):
            o = r.outputs[0]
            out[i] = {"text": o.text, "finish": o.finish_reason, "n_tok": len(o.token_ids), "n_prompt": len(ids[i])}
        return out


class TinyHF:
    """CPU stand-in for tests: Qwen3-0.6B, a few sampled tokens per prompt."""

    def __init__(self, cap=int(os.environ.get("C2_TINY_CAP", "24"))):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
        self.cap = cap
        self.info = {"model": "Qwen/Qwen3-0.6B (tiny test)", "decoding": DECODING}

    def generate(self, texts, max_tokens, seeds):
        out = []
        for t, mt, sd in zip(texts, max_tokens, seeds):
            x = self.tok(t, add_special_tokens=False, return_tensors="pt")
            self.torch.manual_seed(sd)
            with self.torch.no_grad():
                y = self.m.generate(**x, do_sample=True, temperature=DECODING["temperature"], top_p=DECODING["top_p"],
                                    top_k=DECODING["top_k"], max_new_tokens=min(mt, self.cap))
            g = y[0, x["input_ids"].shape[1]:]
            out.append({"text": self.tok.decode(g, skip_special_tokens=False), "finish": "length",
                        "n_tok": int(g.shape[0]), "n_prompt": int(x["input_ids"].shape[1])})
        return out


def write_jsonl(path, rows, mode="w"):
    with open(path, mode) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def tok_stats(rows):
    n = sorted(r["n_tok"] for r in rows)
    if not n:
        return {}
    return {"n": len(n), "mean": round(sum(n) / len(n)), "p50": n[len(n) // 2], "p90": n[int(0.9 * (len(n) - 1))],
            "max": n[-1], "at_cap": sum(r["finish"] == "length" for r in rows), "total": sum(n)}


# ------------------------------------------------------------------ pre-pass (clean samples)
def prepass(be, items, n_samples, max_tokens, tag, think_open, out_path, chunks):
    rows = []
    for ci, samples in enumerate(chunks(n_samples)):
        jobs = [(it, s) for it in items for s in samples]
        texts = [render(be.tok, [{"role": "user", "content": clean_user(it)}], think_open) for it, _ in jobs]
        t = time.time()
        res = be.generate(texts, [max_tokens] * len(jobs), [h32(f"{tag}|{it['id']}|{s}") for it, s in jobs])
        part = [{"id": it["id"], "ds": it["ds"], "sample": s, **r} for (it, s), r in zip(jobs, res)]
        write_jsonl(out_path, part, "a" if ci else "w")
        rows += part
        st = tok_stats(part)
        log(f"{tag} pre-pass chunk {ci} (samples {list(samples)}): {len(part)} gens in {time.time() - t:.0f}s, "
            f"{st.get('total', 0) / max(time.time() - t, 1e-9):.0f} tok/s; tokens {st}")
    return rows


# ------------------------------------------------------------------ the factorial
def build_trials(items):
    trials = []
    for it in items:
        for author in ("qwen", "gemma"):
            for label in ("self", "other"):
                for concl in ("right", "corrupt"):
                    for loc in ("T", "U"):
                        trials.append({"key": f"{it['id']}|{author}|{label}|{concl}|{loc}", "id": it["id"],
                                       "ds": it["ds"], "author": author, "label": label, "concl": concl, "loc": loc})
    return trials


def trial_text(tok, it, tr):
    trace = it["traces"][tr["author"]][tr["concl"]]
    if tr["loc"] == "T":
        return render(tok, [{"role": "user", "content": clean_user(it)}], True) + t_prefill(trace, tr["label"])
    return render(tok, [{"role": "user", "content": u_user(it, trace, tr["label"])}], True)


def m5_text(tok, it, tr, first_raw):
    trace = it["traces"][tr["author"]][tr["concl"]]
    _, ans, _ = split_qwen(first_raw)
    msgs = [{"role": "user", "content": u_user(it, trace, tr["label"])}, {"role": "assistant", "content": ans},
            {"role": "user", "content": FOLLOWUP}]
    return render(tok, msgs, True)


def main_run(be, d, out: Path, limit_items=None):
    cfg, items = d["config"], d["items"][:limit_items]
    byid = {it["id"]: it for it in items}
    order = [it["id"] for it in items]          # build_main writes items in chunk order (datasets interleaved)
    csize = cfg["chunk_items"]
    mt = {"T": cfg["max_tokens_T"], "U": cfg["max_tokens_U"]}
    (out / "main.jsonl").unlink(missing_ok=True)
    (out / "m5.jsonl").unlink(missing_ok=True)
    allrows, m5rows, pending = [], [], []      # pending: designated U-corrupted first turns awaiting the follow-up

    def m5_jobs(rows):
        js = []
        for r in rows:
            c = byid[r["id"]].get("m5")
            if c and r["key"] == f"{r['id']}|{c['author']}|{c['label']}|corrupt|U":
                js.append(r)
        return js

    starts = list(range(0, len(order), csize)) + [None]
    for ci, c0 in enumerate(starts):
        ids = order[c0:c0 + csize] if c0 is not None else []
        trials = build_trials([byid[i] for i in ids])
        texts = [trial_text(be.tok, byid[tr["id"]], tr) for tr in trials]
        mts = [mt[tr["loc"]] for tr in trials]
        seeds = [h32("main|" + tr["key"]) for tr in trials]
        texts += [m5_text(be.tok, byid[r["id"]], r, r["text"]) for r in pending]
        mts += [cfg["max_tokens_m5"]] * len(pending)
        seeds += [h32("m5|" + r["key"]) for r in pending]
        if not texts:
            break
        t = time.time()
        res = be.generate(texts, mts, seeds)
        rows = [{**tr, **r} for tr, r in zip(trials, res[:len(trials)])]
        f5 = [{"key": r["key"], "id": r["id"], "ds": r["ds"], "author": r["author"], "label": r["label"],
               "first_text": r["text"], **x} for r, x in zip(pending, res[len(trials):])]
        write_jsonl(out / "main.jsonl", rows, "a")
        write_jsonl(out / "m5.jsonl", f5, "a")
        allrows += rows
        m5rows += f5
        pending = m5_jobs(rows)
        st = tok_stats(rows)
        log(f"main call {ci}: {len(rows)} trials (items {c0}-{(c0 or 0) + len(ids) - 1}) + {len(f5)} follow-ups in "
            f"{time.time() - t:.0f}s, {(st.get('total', 0) + sum(x['n_tok'] for x in f5)) / max(time.time() - t, 1e-9):.0f} "
            f"tok/s; trial tokens {st}")
    return allrows, m5rows


def get_backend(model, mml, tiny):
    return TinyHF() if tiny else VLLM(model, mml)


def interleave(items):
    g = [it for it in items if it["ds"] == "gsm8k"]
    m = [it for it in items if it["ds"] == "mmlu"]
    return [x for pair in zip(g, m) for x in pair] + g[len(m):] + m[len(g):]


def smoke_cands(cands):
    return [next(c for c in cands if c["ds"] == "gsm8k"), next(c for c in cands if c["ds"] == "mmlu")]


def run_build(args):
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "cot.c2_provenance.build_main", *args], capture_output=True, text=True)
    log("build_main " + " ".join(args) + ":\n" + r.stdout[-3000:] + r.stderr[-1500:])
    if r.returncode != 0:
        raise SystemExit(f"build_main failed (exit {r.returncode})")


def stage_gemma_smoke(be, cands, out):
    """2 items (first GSM8K, first MMLU), 1 Gemma sample each (max 1536 tokens): Gemma loading, generation, parsing."""
    sm = out / "smoke"
    sm.mkdir(parents=True, exist_ok=True)
    rows = prepass(be, smoke_cands(cands), 1, 1536, "gemma", False, sm / "prepass_gemma.jsonl", lambda n: [range(n)])
    return {"smoke": tok_stats(rows)}


def stage_qwen_a(be, cands, out):
    """Smoke (2 items through Qwen pre-pass, build, factorial + M5 follow-ups), then the real Qwen pre-pass."""
    sm = out / "smoke"
    sc = smoke_cands(cands)
    prepass(be, sc, 5, 2048, "qwen", True, sm / "prepass_qwen.jsonl", lambda n: [range(n)])
    ids = ",".join(c["id"] for c in sc)
    run_build(["--prepass-dir", str(sm), "--out-dir", str(sm / "build"), "--ids", ids])
    d = json.loads((sm / "build" / "items_main.json").read_text())
    if not d["items"]:     # neither smoke item passed the filters: exercise the factorial on synthetic traces
        log("smoke: no item passed; factorial smoke on synthetic traces")
        run_build(["--fake", "--out-dir", str(sm / "build"), "--ids", ids])
        d = json.loads((sm / "build" / "items_main.json").read_text())
    # one item per call (so the M5 ride-along runs too) and short generations: the smoke checks the code path only
    d["config"] = {**d["config"], "chunk_items": 1, "max_tokens_T": 512, "max_tokens_U": 512, "max_tokens_m5": 256}
    rows, m5 = main_run(be, d, sm)
    log(f"smoke factorial: {len(rows)} trials, {len(m5)} follow-ups; finishes {Counter(r['finish'] for r in rows)}")
    t = time.time()
    rows = prepass(be, cands, 5, 4096, "qwen", True, out / "prepass_qwen.jsonl", lambda n: [range(1), range(1, n)])
    return {"prepass": tok_stats(rows), "prepass_seconds": round(time.time() - t)}


def stage_gemma(be, cands, out, t0, stop_min):
    """Gemma traces only for items Qwen keeps (>= 4/5 correct and a passing Qwen pair), datasets interleaved, chunks of
    <= 25 items. Sample 0 for a chunk; sample 1 only for items whose sample 0 gave no usable trace (correct + passing
    pair), riding along with the next chunk. No new chunk after stop_min minutes from job start once >= 100 items have a
    usable trace."""
    from cot.c2_provenance.build_main import pick, score
    byid = {c["id"]: c for c in cands}
    qs = score(load_jsonl(out / "prepass_qwen.jsonl"), split_qwen, byid)
    kept = []
    for c in cands:
        q = qs.get(c["id"], [])
        if len(q) >= 5 and sum(x["correct"] for x in q) >= 4 and pick(c, q)[1] is not None:
            kept.append(c)
    order = interleave(kept)
    log(f"gemma: {len(order)} kept items need a trace ({Counter(c['ds'] for c in order)})")
    path = out / "prepass_gemma.jsonl"
    path.unlink(missing_ok=True)
    chunks = [order[i:i + 25] for i in range(0, len(order), 25)]
    usable, pending, k, allrows, stopped = set(), [], 0, [], False
    while True:
        late = (time.time() - t0) / 60 > stop_min and len(usable) >= 100
        stopped = stopped or (late and k < len(chunks))
        chunk = chunks[k] if (k < len(chunks) and not late) else []
        jobs = [(it, 0) for it in chunk] + [(it, 1) for it in pending]
        if not jobs:
            break
        texts = [render(be.tok, [{"role": "user", "content": clean_user(it)}], False) for it, _ in jobs]
        t = time.time()
        res = be.generate(texts, [3072] * len(jobs), [h32(f"gemma|{it['id']}|{s}") for it, s in jobs])
        rows = [{"id": it["id"], "ds": it["ds"], "sample": s, **r} for (it, s), r in zip(jobs, res)]
        write_jsonl(path, rows, "a")
        allrows += rows
        pending = []
        for (it, s), r in zip(jobs, rows):
            ok = pick(it, score([r], split_gemma, byid).get(it["id"], []))[1] is not None
            if ok:
                usable.add(it["id"])
            elif s == 0:
                pending.append(it)
        st = tok_stats(rows)
        log(f"gemma call {k}: {len(chunk)} items x sample 0 + {len(jobs) - len(chunk)} second samples in "
            f"{time.time() - t:.0f}s ({st.get('total', 0) / max(time.time() - t, 1e-9):.0f} tok/s); usable so far "
            f"{len(usable)}/{len(order)}; tokens {st}")
        k += 1
    return {"kept_items": len(order), "usable": len(usable), "stopped_early": stopped, "chunks_run": min(k, len(chunks)),
            "tokens": tok_stats(allrows)}


def stage_qwen_b(be, out):
    run_build(["--prepass-dir", str(out), "--out-dir", str(out / "build")])
    g = json.loads((out / "build" / "gate.json").read_text())
    log(f"gate_pass={g.get('gate_pass')} clean_acc={g.get('clean_acc_final_items')} "
        f"parse={g.get('corrupt_parse_rate_first_choice')} n_items={g.get('n_items')} sha {g.get('items_sha256', '')[:12]}")
    if not g.get("gate_pass"):
        log("gate failed: no factorial")
        return {"gate_pass": False}
    raw = (out / "build" / "items_main.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == g["items_sha256"]
    d = json.loads(raw)
    t = time.time()
    rows, m5 = main_run(be, d, out)
    return {"gate_pass": True, "items_sha256": g["items_sha256"], "config": d["config"], "seconds": round(time.time() - t),
            "tokens": tok_stats(rows), "m5": tok_stats(m5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gemma_smoke", "qwen_a", "gemma", "qwen_b"], required=True)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N candidates per dataset (tests)")
    ap.add_argument("--gemma-stop-min", type=float, default=float(os.environ.get("C2_GEMMA_STOP_MIN", "38")))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cands = json.loads((HERE / "data" / "candidates.json").read_text())
    if a.limit:
        cands = [c for c in cands if c["ds"] == "gsm8k"][:a.limit] + [c for c in cands if c["ds"] == "mmlu"][:a.limit]
    t0 = float(os.environ.get("C2_T0", time.time()))
    gem = a.stage.startswith("gemma")
    be = get_backend(GEMMA if gem else QWEN, 8192 if gem else 16384, a.tiny)
    settings = {"stage": a.stage, "git_rev": os.environ.get("GIT_REV", "?"), "backend": be.info,
                "code_sha256": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()
                                for f in ("common.py", "pod_c2.py", "build_main.py")}}
    log(f"stage {a.stage}: loaded {be.info['model']}; {(time.time() - t0) / 60:.1f} min since job start")
    if a.stage == "gemma_smoke":
        settings.update(stage_gemma_smoke(be, cands, out))
    elif a.stage == "qwen_a":
        settings.update(stage_qwen_a(be, cands, out))
    elif a.stage == "gemma":
        settings.update(stage_gemma(be, cands, out, t0, a.gemma_stop_min))
    else:
        settings.update(stage_qwen_b(be, out))
    (out / f"settings_{a.stage}.json").write_text(json.dumps(settings, indent=1))
    log(f"stage {a.stage} done; {(time.time() - t0) / 60:.1f} min since job start")


if __name__ == "__main__":
    main()
