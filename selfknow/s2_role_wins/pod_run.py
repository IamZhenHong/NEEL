"""vLLM runner for s2 (adapted from vigilante/src/pod_run.py). One process per model.

Runs the --initial job files, then (with --inbox) waits for further job files to appear in the inbox directory and
runs each in name order, until a file named STOP is present and nothing is pending, or nothing arrives for
--wait-minutes. With --inbox and no --initial it does not load the model until a job file (or STOP) is in the inbox.
This lets the main run start on the already-loaded model after the pilot has been read, recorded and the freeze tag
set locally (the main job file is uploaded into the inbox after `s2-freeze`). A heartbeat line every minute keeps
the launcher's stall watchdog quiet.

  /root/vllm/bin/python -m selfknow.s2_role_wins.pod_run --model Qwen/Qwen3.6-27B --revision <sha> --fmt qwen \
      --tag qwen --out selfknow/s2_role_wins/results --initial <pilot jobs> --inbox <dir> --wait-minutes 30
  --dry-run: no vLLM; a canned generator (local test of the plumbing).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from selfknow.s2_role_wins.engine import dumps, record, run_batch

DECODING = dict(temperature=0.7, top_p=0.95, top_k=20, min_p=0.0)     # as every vigilante run


def pending(inbox: Path, out: Path) -> list[Path]:
    return [p for p in sorted(inbox.glob("*.jsonl")) if not (out / p.stem / "DONE").exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--fmt", required=True, choices=["qwen", "gemma4"])
    ap.add_argument("--tag", required=True, help="results subfolder, e.g. qwen / gemma")
    ap.add_argument("--out", required=True)
    ap.add_argument("--initial", nargs="*", default=[])
    ap.add_argument("--inbox", default=None)
    ap.add_argument("--wait-minutes", type=float, default=0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    inbox = Path(a.inbox) if a.inbox else None
    if inbox:
        inbox.mkdir(parents=True, exist_ok=True)

    t_wait, last_beat = time.time(), 0.0
    if inbox and not a.initial:          # do not load until there is something to run
        while not pending(inbox, out) and not (inbox / "STOP").exists():
            if time.time() - last_beat > 60:
                print(f"[s2] {a.tag}: waiting for inbox before loading ({time.time() - t_wait:.0f}s)", flush=True)
                last_beat = time.time()
            if time.time() - t_wait > a.wait_minutes * 60:
                print(f"[s2] {a.tag}: nothing arrived in {a.wait_minutes} min; exiting", flush=True)
                return
            time.sleep(5)
        if not pending(inbox, out):
            print(f"[s2] {a.tag}: STOP and nothing pending; exiting without loading", flush=True)
            return

    settings = {"model": a.model, "fmt": a.fmt, "decoding": DECODING, "max_model_len": a.max_model_len, "files": {}}
    if a.dry_run:
        tok = None

        def generate(items):
            outs = []
            for it in items:
                last = it["messages"][-1]
                if last["role"] == "user" and "refund" in last["content"]:
                    call = ('<|tool_call>call:issue_refund{amount:60,order_id:<|"|>5521<|"|>}<tool_call|>'
                            if a.fmt == "gemma4" else
                            "<tool_call>\n<function=issue_refund>\n<parameter=order_id>\n5521\n</parameter>\n"
                            "<parameter=amount>\n60\n</parameter>\n</function>\n</tool_call>")
                    outs.append(("Sure." + call, "stop"))
                else:
                    outs.append(("I'm Pip, happy to help.\n\n- Pip", "stop"))
            return outs
    else:
        from huggingface_hub import HfApi
        import vllm
        from vllm import LLM, SamplingParams, TokensPrompt
        from vigilante.src.templates import clean_output, to_template
        rev = a.revision or HfApi().model_info(a.model).sha
        t0 = time.time()
        kw = dict(model=a.model, revision=rev, dtype="bfloat16", max_model_len=a.max_model_len,
                  gpu_memory_utilization=0.92, seed=0, enable_prefix_caching=True)
        try:
            llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
        except Exception as e:  # some multimodal configs reject a modality key; retry with images only
            print(f"[s2] LLM init with image+video limits failed ({type(e).__name__}: {str(e)[:200]}); retrying", flush=True)
            llm = LLM(**kw, limit_mm_per_prompt={"image": 0})
        tok = llm.get_tokenizer()
        tmpl = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
        settings.update({"revision": rev, "vllm": vllm.__version__, "dtype": "bfloat16",
                         "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest(),
                         "load_seconds": round(time.time() - t0)})
        print(f"[s2] loaded {a.model}@{rev[:10]} in {settings['load_seconds']}s", flush=True)
        MML = a.max_model_len

        def generate(items):
            prompts, params, skip = [], [], []
            for it in items:
                text = tok.apply_chat_template(to_template(it["messages"], a.fmt), tools=it["tools"], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
                ids = tok(text, add_special_tokens=False)["input_ids"]
                room = MML - len(ids) - 8
                skip.append(room < 64)
                prompts.append(ids)
                params.append(SamplingParams(**DECODING, max_tokens=max(1, min(it["max_tokens"], room)),
                                             seed=it["seed"], skip_special_tokens=False))
            idx = [i for i, s in enumerate(skip) if not s]
            res = llm.generate([TokensPrompt(prompt_token_ids=prompts[i]) for i in idx], [params[i] for i in idx],
                               use_tqdm=False)
            outs = [("", "context_full")] * len(items)
            for i, r in zip(idx, res):
                o = r.outputs[0]
                outs[i] = (clean_output(o.text, a.fmt), o.finish_reason)
            return outs

    def run_file(path: Path):
        jobs = [json.loads(l) for l in open(path) if l.strip()]
        jobs = [j for j in jobs if j.get("model") in (None, a.tag)]
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        d = out / path.stem
        d.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        print(f"[s2] {a.tag}: {path.name} ({len(jobs)} jobs, sha256 {sha[:12]})", flush=True)
        eps = run_batch(jobs, generate, log=lambda m: print(m, flush=True))
        with open(d / "episodes.jsonl", "w") as f:
            for e in eps:
                f.write(dumps(record(e)) + "\n")
        settings["files"][path.name] = {"n": len(jobs), "sha256": sha, "seconds": round(time.time() - t1)}
        (d / "settings.json").write_text(json.dumps(settings, indent=1))
        (d / "DONE").write_text("ok\n")
        print(f"[s2] {a.tag}: {path.name} done in {time.time() - t1:.0f}s", flush=True)

    for p in a.initial:
        run_file(Path(p))
    if inbox:
        t_wait, last_beat = time.time(), 0.0
        while True:
            todo = pending(inbox, out)
            if todo:
                run_file(todo[0])
                t_wait = time.time()
                continue
            if (inbox / "STOP").exists():
                print(f"[s2] {a.tag}: STOP; exiting", flush=True)
                break
            if time.time() - t_wait > a.wait_minutes * 60:
                print(f"[s2] {a.tag}: nothing arrived in {a.wait_minutes} min; exiting", flush=True)
                break
            if time.time() - last_beat > 60:
                print(f"[s2] {a.tag}: waiting for inbox ({time.time() - t_wait:.0f}s idle)", flush=True)
                last_beat = time.time()
            time.sleep(5)
    print(f"[s2] {a.tag}: process done", flush=True)


if __name__ == "__main__":
    main()
