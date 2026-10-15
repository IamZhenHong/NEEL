"""c2 on one pod: Gemma 4 31B traces, then Qwen3.6-27B clean pre-pass, a wait for the frozen item set (GO), then the
16-cell factorial and the M5 follow-ups, all with thinking on. One model per process.

  /root/vllm/bin/python -m cot.c2_provenance.pod_c2 --stage gemma --out cot/c2_provenance/results
  /root/vllm/bin/python -m cot.c2_provenance.pod_c2 --stage qwen  --out cot/c2_provenance/results --wait-minutes 30

Local CPU test (Qwen3-0.6B, HF transformers, a few tokens per call):
  uv run python -m cot.c2_provenance.pod_c2 --stage qwen --tiny --limit 3 --out <dir> --auto-go
The qwen stage writes prepass_qwen.jsonl, then polls run/ for GO (written by the operator after the freeze tag,
next to items_main.json.gz); it checks the sha256 named in GO, runs the factorial in chunks (main.jsonl, appended per
chunk) and the follow-ups (m5.jsonl). run/STOP ends the stage without a main run.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

from cot.c2_provenance.common import (DECODING, FOLLOWUP, GEMMA, HERE, QWEN, clean_user, h32, render, split_qwen,
                                      t_prefill, u_user)

T0 = time.time()


def log(m):
    print(f"[c2 {time.time() - T0:7.0f}s] {m}", flush=True)


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


def wait_go(run_dir: Path, minutes: float, auto: bool):
    if auto:
        return {"auto": True}
    t = time.time()
    last = 0
    while time.time() - t < minutes * 60:
        if (run_dir / "STOP").exists():
            log("STOP found: no main run")
            return None
        if (run_dir / "GO").exists():
            time.sleep(3)   # let a concurrent upload finish
            return json.loads((run_dir / "GO").read_text())
        if time.time() - last > 60:
            log(f"waiting for GO in {run_dir} ({(time.time() - t) / 60:.1f}/{minutes} min)")
            last = time.time()
        time.sleep(5)
    log("no GO before the wait limit: no main run")
    return None


def load_items(run_dir: Path, go: dict, local_items: str | None, out: Path = None):
    p = Path(local_items) if local_items else run_dir / "items_main.json.gz"
    if not p.exists() and out is not None:
        p = out / "build" / "items_main.json"      # GO may name the pod-built set (same code, same pre-pass)
    raw = gzip.open(p).read() if str(p).endswith(".gz") else p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if go.get("items_sha256") and go["items_sha256"] != sha:
        raise SystemExit(f"items sha256 {sha[:12]} does not match GO {go['items_sha256'][:12]}")
    log(f"items file {p} sha256 {sha[:12]} (GO {str(go.get('items_sha256'))[:12]})")
    return json.loads(raw), sha


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gemma", "qwen"], required=True)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--run-dir", default=str(HERE / "run"))
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N candidates per dataset (tests)")
    ap.add_argument("--wait-minutes", type=float, default=30)
    ap.add_argument("--auto-go", action="store_true", help="tests: do not wait; use --items")
    ap.add_argument("--items", default=None, help="tests: a local items_main.json")
    ap.add_argument("--skip-prepass", action="store_true", help="tests: go straight to the main run")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cands = json.loads((HERE / "data" / "candidates.json").read_text())
    if a.limit:
        cands = [c for c in cands if c["ds"] == "gsm8k"][:a.limit] + [c for c in cands if c["ds"] == "mmlu"][:a.limit]
    settings = {"stage": a.stage, "git_rev": os.environ.get("GIT_REV", "?")}

    if a.stage == "gemma":
        be = TinyHF() if a.tiny else VLLM(GEMMA, 8192)
        settings["backend"] = be.info
        log(f"loaded {be.info['model']}; {len(cands)} candidates x 3 samples")
        rows = prepass(be, cands, 3, 3072, "gemma", False, out / "prepass_gemma.jsonl", lambda n: [range(n)])
        settings["prepass"] = tok_stats(rows)
        (out / "settings_gemma.json").write_text(json.dumps(settings, indent=1))
        log("gemma stage done")
        return

    be = TinyHF() if a.tiny else VLLM(QWEN, 16384)
    settings["backend"] = be.info
    log(f"loaded {be.info['model']}")
    if not a.skip_prepass:
        rows = prepass(be, cands, 5, 4096, "qwen", True, out / "prepass_qwen.jsonl", lambda n: [range(1), range(1, n)])
        settings["prepass"] = tok_stats(rows)
        (out / "settings_qwen.json").write_text(json.dumps(settings, indent=1))
        (out / "PREPASS_DONE").write_text(time.strftime("%H:%M:%S"))
        log("qwen pre-pass done; PREPASS_DONE written")
    if not a.tiny:
        import subprocess
        import sys
        r = subprocess.run([sys.executable, "-m", "cot.c2_provenance.build_main", "--prepass-dir", str(out),
                            "--out-dir", str(out / "build")], capture_output=True, text=True)
        log("pod build_main:\n" + r.stdout[-3500:] + r.stderr[-1500:])
    go = None
    gp = out / "build" / "gate.json"
    if (Path(a.run_dir) / "AUTO").exists() and gp.exists():   # frozen before the pre-pass: the gate decides on the pod
        g = json.loads(gp.read_text())
        log(f"AUTO: gate_pass={g.get('gate_pass')} n_items={g.get('n_items')} sha {g.get('items_sha256', '')[:12]}")
        if not g.get("gate_pass"):
            log("gate failed: no main run")
            return
        go = {"items_sha256": g["items_sha256"], "auto": True}
    if go is None:
        go = wait_go(Path(a.run_dir), a.wait_minutes, a.auto_go)
    if go is None:
        return
    d, sha = load_items(Path(a.run_dir), go, a.items, out)
    settings["items_sha256"] = sha
    settings["config"] = d["config"]
    # re-import the code from disk so that a fix uploaded before GO (ops put) is what runs; record its hashes
    import importlib
    importlib.reload(importlib.import_module("cot.c2_provenance.common"))
    pm = importlib.reload(importlib.import_module("cot.c2_provenance.pod_c2"))
    settings["code_sha256"] = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ("common.py", "pod_c2.py")}
    log(f"code sha256 {({k: v[:12] for k, v in settings['code_sha256'].items()})}")
    t = time.time()
    allrows, m5 = pm.main_run(be, d, out)
    settings["main"] = {"seconds": round(time.time() - t), "tokens": tok_stats(allrows), "m5": tok_stats(m5)}
    (out / "settings_qwen.json").write_text(json.dumps(settings, indent=1))
    log("qwen stage done")


if __name__ == "__main__":
    main()
