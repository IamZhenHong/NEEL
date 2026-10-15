"""s1 on the pod (vLLM venv). One process per model and stage.

  /root/vllm/bin/python -m selfknow.s1_persistence.pod_s1 --model-key qwen --stage pilot --out <dir> [--mml ...]

Stages: pilot = 1k only, variants 0-1, behavioural probes x8, verbal x4, no-plant constraint probes x4 (gate).
        main  = variants 0-7 x {1k,4k,12k,24k} x (10 probes x4 + no-plant 2 probes x4) + 4 compaction summaries per
                24k prefix, then (second pass) every probe once on each compacted context.
Branches are appended to <out>/branches_<model>.jsonl as they finish; a rerun skips finished keys (resume).
Exit code 3 = the engine could not be built with these settings (pod.sh then tries the next setting).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from selfknow.s1_persistence import engine as E
from selfknow.s1_persistence import render as R
from selfknow.s1_persistence import scenario as S

H = Path(__file__).parent
MODELS = {"qwen": ("Qwen/Qwen3.6-27B", "qwen"), "gemma4": ("google/gemma-4-31B-it", "gemma4")}
DECODING = dict(temperature=0.7, top_p=0.95, top_k=20, min_p=0.0)   # as vigilante/src/pod_run.py


class VLLMStep:
    """Continuous batching through LLMEngine.add_request / step (what LLM.generate does internally)."""
    def __init__(self, llm):
        from vllm import SamplingParams, TokensPrompt
        from vllm.sampling_params import RequestOutputKind
        self.eng, self.SP, self.TP, self.K, self.n = llm.llm_engine, SamplingParams, TokensPrompt, RequestOutputKind, 0

    def submit(self, rid, ids, max_tokens, seed):
        sp = self.SP(**DECODING, max_tokens=max_tokens, seed=seed, skip_special_tokens=False)
        sp.output_kind = self.K.FINAL_ONLY
        self.eng.add_request(rid, self.TP(prompt_token_ids=ids), sp)
        self.n += 1

    def poll(self):
        while self.n:
            fin = [o for o in self.eng.step() if o.finished]
            if fin:
                self.n -= len(fin)
                return [(o.request_id, o.outputs[0].text, o.outputs[0].finish_reason, len(o.outputs[0].token_ids))
                        for o in fin]
            if not self.eng.has_unfinished_requests():
                raise RuntimeError(f"engine idle with {self.n} requests pending")
        return []

    def pending(self):
        return self.n


class VLLMRounds:
    """Fallback: one llm.generate call per round over everything submitted."""
    def __init__(self, llm):
        from vllm import SamplingParams, TokensPrompt
        self.llm, self.SP, self.TP, self.q = llm, SamplingParams, TokensPrompt, []

    def submit(self, rid, ids, max_tokens, seed):
        self.q.append((rid, ids, max_tokens, seed))

    def poll(self):
        q, self.q = self.q, []
        if not q:
            return []
        res = self.llm.generate([self.TP(prompt_token_ids=ids) for _, ids, _, _ in q],
                                [self.SP(**DECODING, max_tokens=m, seed=s, skip_special_tokens=False) for _, _, m, s in q],
                                use_tqdm=False)
        return [(rid, r.outputs[0].text, r.outputs[0].finish_reason, len(r.outputs[0].token_ids))
                for (rid, _, _, _), r in zip(q, res)]

    def pending(self):
        return len(self.q)


def check_contexts(ctx, mk, tok, fmt, log):
    """Content hashes must match the frozen contexts.json; token counts are re-measured for a few prefixes."""
    m = ctx["models"][mk]
    bad = []
    for vi in range(8):
        for pk in ("plant", "noplant"):
            for ln, n in m["n_filler"][str(vi)][pk].items():
                if E.neutral_hash(E.neutral_prefix(vi, pk == "plant", n)) != m["hash"][str(vi)][pk][ln]:
                    bad.append((vi, pk, ln))
    if bad:
        raise SystemExit(f"[s1] context hash mismatch: {bad[:5]}")
    for vi, pk, ln in ((0, "plant", "1k"), (0, "plant", "24k"), (5, "noplant", "12k")):
        n = m["n_filler"][str(vi)][pk][ln]
        msgs = E.neutral_prefix(vi, pk == "plant", n) + [{"role": "user", "content": "OK, carry on."}]
        full = len(R.prompt_ids(tok, R.to_raw(msgs, fmt), S.TOOLS, fmt))
        log(f"[s1] prompt tokens v{vi} {pk} {ln}: {full} (frozen prefix {m['tokens'][str(vi)][pk][ln]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=list(MODELS))
    ap.add_argument("--stage", required=True, choices=["pilot", "main"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--mml", type=int, default=32768)
    ap.add_argument("--mnbt", type=int, default=None, help="max_num_batched_tokens")
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--kv-dtype", default="auto")
    ap.add_argument("--budget", type=int, default=None, help="prefix tokens of active groups")
    ap.add_argument("--engine", default="step", choices=["step", "rounds"])
    ap.add_argument("--variants", default=None, help="comma list (smoke)")
    ap.add_argument("--probes", default=None, help="comma list: re-pilot only these probes (reworded on the 1k pilot)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name, fmt = MODELS[a.model_key]
    log = lambda m: print(m, flush=True)
    ctx = json.loads((H / "contexts.json").read_text())

    from huggingface_hub import HfApi
    import vllm
    from vllm import LLM
    rev = HfApi().model_info(name).sha
    kw = dict(model=name, revision=rev, dtype="bfloat16", max_model_len=a.mml, gpu_memory_utilization=a.gpu_mem,
              seed=0, enable_prefix_caching=True, kv_cache_dtype=a.kv_dtype)
    if a.mnbt:
        kw["max_num_batched_tokens"] = a.mnbt
    t0 = time.time()
    try:
        try:
            llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
        except Exception as e:
            if "limit_mm" not in str(e) and "modality" not in str(e).lower():
                raise
            llm = LLM(**kw, limit_mm_per_prompt={"image": 0})
    except Exception as e:
        log(f"[s1] ENGINE INIT FAILED ({type(e).__name__}: {str(e)[:600]})")
        sys.exit(3)
    tok = llm.get_tokenizer()
    tmpl = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
    settings = {"model": name, "revision": rev, "vllm": vllm.__version__, "engine_kwargs": {k: v for k, v in kw.items()
                if k != "model"}, "decoding": DECODING, "fmt": fmt, "stage": a.stage,
                "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest(), "load_seconds": round(time.time() - t0)}
    log(f"[s1] loaded {name}@{rev[:10]} in {settings['load_seconds']}s: {settings['engine_kwargs']}")
    check_contexts(ctx, a.model_key, tok, fmt, log)

    backend = VLLMStep(llm) if a.engine == "step" else VLLMRounds(llm)
    if a.engine == "step":
        try:   # one tiny request through the step API; fall back to rounds if this vLLM build differs
            backend.submit("warmup", tok("Hello", add_special_tokens=False)["input_ids"], 4, 0)
            backend.poll()
            settings["engine"] = "step"
        except Exception as e:
            log(f"[s1] step API failed ({type(e).__name__}: {e}); using round-based generate")
            try:
                llm.llm_engine.abort_request(["warmup"])
            except Exception:
                pass
            backend, settings["engine"] = VLLMRounds(llm), "rounds"
    else:
        settings["engine"] = "rounds"

    variants = [int(x) for x in a.variants.split(",")] if a.variants else None
    if a.stage == "pilot":
        groups, _ = E.plan(ctx, a.model_key, "pilot", variants=variants or [0, 1], b_samples=8, v_samples=4,
                           np_samples=4, n_summaries=0, lengths=("1k",))
    else:
        groups, _ = E.plan(ctx, a.model_key, "main", variants=variants, n_summaries=4)
    if a.probes:
        keep = set(a.probes.split(","))
        for g in groups:
            g["branches"] = [b for b in g["branches"] if b["probe"] in keep]
    path = out / f"branches_{a.model_key}.jsonl"
    done = {}
    if path.exists():
        for l in open(path):
            if l.strip():
                r = json.loads(l)
                done[r["key"]] = r
    for g in groups:
        g["branches"] = [b for b in g["branches"] if b["key"] not in done]
    groups = [g for g in groups if g["branches"]]
    budget = a.budget or (150000 if a.model_key == "qwen" else 50000)
    run = E.Runner(backend, tok, fmt, a.mml, budget, path, log=log)
    t1 = time.time()
    n_new = sum(len(g["branches"]) for g in groups)
    log(f"[s1] {a.stage}: {n_new} branches to run in {len(groups)} groups ({len(done)} already done)")
    fin = run.run(groups)
    settings["pass1"] = {"branches": len(fin), "seconds": round(time.time() - t1), **run.stats}
    (out / f"settings_{a.model_key}.json").write_text(json.dumps(settings, indent=1))

    if a.stage == "main":   # second pass: probes on the agent-written compacted contexts
        sums = [b for b in fin if b["kind"] == "summary"] + [r for r in done.values() if r["kind"] == "summary"]
        cg = E.compact_groups(sums, a.model_key, "main")
        for g in cg:
            g["branches"] = [b for b in g["branches"] if b["key"] not in done]
        cg = [g for g in cg if g["branches"]]
        t2 = time.time()
        log(f"[s1] compaction pass: {len(sums)} summaries -> {sum(len(g['branches']) for g in cg)} branches")
        run2 = E.Runner(backend, tok, fmt, a.mml, 400000, path, log=log)
        fin2 = run2.run(cg)
        settings["pass2"] = {"branches": len(fin2), "seconds": round(time.time() - t2), **run2.stats}
    settings["total_seconds"] = round(time.time() - t0)
    (out / f"settings_{a.model_key}.json").write_text(json.dumps(settings, indent=1))
    log(f"[s1] {a.model_key} {a.stage} done in {settings['total_seconds']}s")


if __name__ == "__main__":
    main()
