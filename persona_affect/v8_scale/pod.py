"""PA-V8 pod runner (plan persona_affect/plans/V8.md, Task 3): the models of a pod group in sequence, one vLLM process
each. launch.sh starts pod.sh, which runs `pod group`; that starts one `pod run` subprocess per model.

make_engine(model) -> (gen, info)
    vLLM (imported here: only the pod's /root/vllm venv has it) at the model's pinned revision, bf16, max_model_len
    4096, gpu_memory_utilization 0.92 (Gemma 4) / 0.88 (Qwen), seed 0, prefix caching. limit_mm_per_prompt tries
    {"image": 0, "video": 0, "audio": 0} (Gemma 4 only), then {"image": 0, "video": 0}, then {"image": 0}; the last
    attempt's error is the load failure. gen(batch) takes [(messages, seed, max_tokens, with_tools)] and returns
    [(text, finish_reason)]: each prompt is render()ed and sent as token ids (add_special_tokens=False), sampled with
    mini 0's DECODING, the job's seed and max_tokens, skip_special_tokens for Qwen only and the stop ids; the text goes
    through vigilante's clean_output and vLLM's finish_reason is kept. The stop ids are those of STOP_TOKENS that the
    tokenizer knows (Qwen: <|im_end|>, <|endoftext|>; Gemma 4: <turn|>, <|tool_response>, <eos>: the eos sets).
    info = {"vllm", "mm_limits" (the limits that loaded), "chat_template_sha256", "stop_ids", "load_seconds"}.
render(tok, messages, fmt, with_tools) -> str
    The model's chat template on RN.chat(messages, fmt), with T.tools() when with_tools (decisions and nudges) and no
    tools otherwise (say, knowledge); thinking off, generation prompt on.
render_check(models) -> dict
    Local and free (`v8 render-check`): each model's tokenizer files at the pinned revision, no weights. Prints one
    decide, say and know prompt per model and raises ValueError if a decide prompt has >= 1,500 tokens or lacks its
    case file. It checks the printed decide prompt and every item x rung of both families with the family's longest
    line (labels, backups and the tidiness rewrite included), which bounds every decide prompt of every stage.
run(stage, model, out, *, gen=None, frozen=None, limit=None, chunk=256) -> dict
    v8.build_jobs(stage, model, frozen) in prio order (the first `limit`), `chunk` at a time: one gen call for the
    chunk, then one nudge call (RN.NUDGE, seed + 1, 400 tokens, tools) for its decisions without a choose_option call,
    as RN.decide; then the chunk's records are appended to decisions.jsonl / say.jsonl / know.jsonl in `out` and
    flushed. A record is its job minus messages, plus text, finish, first_try (False only for a nudged decision),
    text2 and finish2 (the nudge's reply; None unless nudged). Prints "[v8] {model} {stage} {done}/{total} ({s}s)" per
    chunk and a heartbeat line every HEARTBEAT seconds (the launcher kills a job silent for 30 min). run_info.json:
    "running" at the start (prompt_hash of the full job list, n_total, n_jobs), "done" at the end (+ counts per kind,
    nudged, seconds, engine). If make_engine raises, run_info says "load_failed" with the error and the exception
    propagates: the only record that makes a model "not servable" (Change 5). A crash after the load leaves "running"
    and every chunk already written; a missing vLLM is a SystemExit, never a load failure. `out` must hold no records
    yet (FileExistsError): a run never appends to an earlier one.
run_group(stage, group, out, *, runner=subprocess.run, hub=None, models=None, frozen=None) -> int
    The group's models (GROUPS order, or `models`), minus those `frozen` (v8.FROZEN when None) marks not servable when
    the stage is main or rewrite (Change 5; calibration runs them all). Each runs as `python -m
    persona_affect.v8_scale.pod run ... --out out/<model>` through `runner`; a failure prints "[v8] {m} FAILED (exit
    rc)" and the group goes on. After each model its HF cache folder is deleted (the container disk holds one model at
    a time): under `hub`, else $HF_HOME/hub when HF_HOME is set (the pod sets it); without either nothing is deleted,
    never the laptop's default cache. Returns the first non-zero exit code, else 0.

  /root/vllm/bin/python -m persona_affect.v8_scale.pod group --stage calib --group small --out <dir>    (pod.sh)
  /root/vllm/bin/python -m persona_affect.v8_scale.pod run --stage calib --model q9 --out <dir>/q9 [--limit N]
  PYTHONPATH=persona_affect/tests/fakevllm uv run python -m persona_affect.v8_scale.pod run --stage calib --model q08 \
      --out <scratch dir>                                                                         (local dry run)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

from persona.exemplar import tasks as T
from persona.exemplar.mini0 import DECODING
from persona.history import runner as RN
from persona_affect.v8_scale import items as I
from persona_affect.v8_scale import v8
from persona_affect.v8_scale.v8 import GROUPS, MODELS
from vigilante.src.templates import clean_output

MAX_MODEL_LEN = 4096
MAX_DECIDE_TOKENS = 1500
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "<turn|>", "<|tool_response>", "<eos>")   # unknown ones are dropped
HEARTBEAT = 120   # seconds between heartbeat lines (read at call time: tests shorten it)
FILES = {"decide": "decisions.jsonl", "say": "say.jsonl", "know": "know.jsonl"}


# ---------------------------------------------------------------- rendering and the engine

def render(tok, messages: list[dict], fmt: str, with_tools: bool) -> str:
    """The prompt text the model sees: its chat template on RN.chat(messages, fmt), thinking off."""
    return tok.apply_chat_template(RN.chat(messages, fmt), tools=T.tools() if with_tools else None, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def _template_sha(tok) -> str:
    tmpl = tok.chat_template
    tmpl = tmpl if isinstance(tmpl, str) else json.dumps(tmpl, sort_keys=True)
    return hashlib.sha256((tmpl or "").encode()).hexdigest()


def stop_ids(tok) -> list[int]:
    """The ids of STOP_TOKENS the tokenizer knows, in order, once each: an unknown token maps to None (Qwen) or to the
    unk id (Gemma) and is dropped."""
    unk, ids = getattr(tok, "unk_token_id", None), []
    for s in STOP_TOKENS:
        i = tok.convert_tokens_to_ids(s)
        if i is not None and i != unk and i not in ids:
            ids.append(i)
    return ids


def make_engine(model: str):
    """(gen, info) on vLLM; see the module docstring."""
    try:
        import vllm
    except ImportError:
        raise SystemExit("PA-V8 runs need vLLM, which only the pod's /root/vllm venv has (pod.sh starts the runner "
                         "with it); for a local dry run put persona_affect/tests/fakevllm on PYTHONPATH") from None
    d = MODELS[model]
    fmt = d["fmt"]
    kw = dict(model=d["hf"], revision=d["rev"], tokenizer_revision=d["rev"], dtype="bfloat16",
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.92 if fmt == "gemma4" else 0.88, seed=0,
              enable_prefix_caching=True)
    tries = [{"image": 0, "video": 0}, {"image": 0}]
    if fmt == "gemma4":   # E2B and E4B have an audio tower
        tries.insert(0, {"image": 0, "video": 0, "audio": 0})
    t0 = time.time()
    for k, mm in enumerate(tries):
        try:
            llm = vllm.LLM(**kw, limit_mm_per_prompt=mm)
            break
        except Exception as e:  # noqa: BLE001  the next limits; the last attempt's error is the load failure
            if k == len(tries) - 1:
                raise
            print(f"[v8] {model}: LLM(limit_mm_per_prompt={mm}) failed ({type(e).__name__}: {str(e)[:300]}); "
                  f"trying {tries[k + 1]}", flush=True)
    tok = llm.get_tokenizer()
    stops = stop_ids(tok)
    info = {"vllm": vllm.__version__, "mm_limits": mm, "chat_template_sha256": _template_sha(tok), "stop_ids": stops,
            "load_seconds": round(time.time() - t0)}
    print(f"[v8] {model} loaded {d['hf']}@{d['rev'][:10]} in {info['load_seconds']}s: vllm {info['vllm']}, "
          f"limit_mm_per_prompt {mm}, stop ids {stops}", flush=True)

    def gen(batch):
        prompts = [vllm.TokensPrompt(prompt_token_ids=tok(render(tok, messages, fmt, with_tools),
                                                          add_special_tokens=False)["input_ids"])
                   for messages, _, _, with_tools in batch]
        params = [vllm.SamplingParams(**DECODING, max_tokens=max_tokens, seed=seed, skip_special_tokens=fmt == "qwen",
                                      stop_token_ids=stops) for _, seed, max_tokens, _ in batch]
        res = llm.generate(prompts, params, use_tqdm=False)
        return [(clean_output(r.outputs[0].text, fmt), r.outputs[0].finish_reason) for r in res]
    return gen, info


def _longest_line(f: str) -> str:
    """The family's longest line over every line a stage can show: the five lines, both label candidates, tidiness."""
    return max([I.LINES[f][x] for x in I.LINE_ORDER] + [I.label_line(n) for n in I.LABEL_CANDIDATES[f]] + [I.PLACEBO2],
               key=len)


def render_check(models: list[str], *, load=None) -> dict:
    """{model: {"decide", "say", "know" (tokens of the printed prompts), "decide_max", "chat_template_sha256"}}; see
    the module docstring. `load(hf, revision)` returns a tokenizer (default: transformers.AutoTokenizer)."""
    if load is None:
        def load(hf, rev):
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(hf, revision=rev)
    res = {}
    for m in models:
        d = MODELS[m]
        fmt, tok = d["fmt"], load(d["hf"], d["rev"])

        def tokens(text):
            return len(tok(text, add_special_tokens=False)["input_ids"])

        def decide_tokens(where, msgs):
            text = render(tok, msgs, fmt, True)
            if msgs[-1]["content"] not in text:
                raise ValueError(f"{m}: the rendered decide prompt ({where}) lacks its case file")
            n = tokens(text)
            if n >= MAX_DECIDE_TOKENS:
                raise ValueError(f"{m}: the decide prompt ({where}) has {n} tokens; the limit is under 1,500")
            return text, n
        longest = {f: _longest_line(f) for f in I.FAMILIES}
        bound = [decide_tokens(f"{f} {tid} rung {r} with {longest[f]!r}",
                               I.episode(f, tid, I.persona_slot(longest[f], rung), I.PERM6[0]))[1]
                 for f in I.FAMILIES for tid in I.items(f) for r, rung in enumerate(I.LADDER[f])]
        jobs = v8.build_jobs("calib", m)
        dec = next(j for j in jobs if j["kind"] == "decide")
        know = next(j for j in jobs if j["kind"] == "know")
        say = I.say_messages("justice", I.domains("justice")[0],
                             I.persona_slot(I.LINES["justice"]["value"], I.LADDER["justice"][1]))
        dec_text, dec_n = decide_tokens(dec["key"], dec["messages"])
        shown = {"decide": dec_text, "say": render(tok, say, fmt, False),
                 "know": render(tok, know["messages"], fmt, False)}
        res[m] = {k: (dec_n if k == "decide" else tokens(v)) for k, v in shown.items()}
        res[m] |= {"decide_max": max(bound + [dec_n]), "chat_template_sha256": _template_sha(tok)}
        for k, text in shown.items():
            extra = (f"; the longest of {len(bound) + 1} decide prompts checked has {res[m]['decide_max']}"
                     if k == "decide" else "")
            print(f"===== {m} = {d['hf']}@{d['rev'][:10]} ({fmt}): {k} prompt, {res[m][k]} tokens{extra}")
            print(text)
    return res


# ---------------------------------------------------------------- one model

def _write_info(out: Path, info: dict) -> None:
    """run_info.json through a temporary file, so a crash never leaves half of it."""
    tmp = out / "run_info.json.tmp"
    tmp.write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out / "run_info.json")


def _call(gen, batch: list[tuple]) -> list[tuple]:
    outs = list(gen(batch))
    if len(outs) != len(batch):
        raise RuntimeError(f"gen returned {len(outs)} replies for {len(batch)} prompts")
    return outs


def _heartbeat(stop: threading.Event, label: str, done: list[int], total: int, t0: float) -> None:
    while not stop.wait(HEARTBEAT):
        print(f"[v8] {label} heartbeat {done[0]}/{total} ({time.time() - t0:.0f}s)", flush=True)


def run(stage: str, model: str, out, *, gen=None, frozen: dict | None = None, limit: int | None = None,
        chunk: int = 256) -> dict:
    """Generate one model's jobs of one stage into `out`; returns the final run_info. See the module docstring."""
    if chunk < 1 or (limit is not None and limit < 1):
        raise ValueError(f"chunk and limit must be at least 1, got chunk={chunk!r}, limit={limit!r}")
    out = Path(out)
    everything = v8.build_jobs(stage, model, frozen)
    jobs = everything if limit is None else everything[:limit]
    d = MODELS[model]
    fmt = d["fmt"]
    out.mkdir(parents=True, exist_ok=True)
    held = sorted(name for name in FILES.values() if (out / name).exists())
    if held:
        raise FileExistsError(f"{out} already holds {', '.join(held)}: a run never appends to an earlier one, so "
                              "move that directory away first")
    info = {"model": model, "hf": d["hf"], "rev": d["rev"], "fmt": fmt, "stage": stage, "status": "running",
            "prompt_hash": v8.prompt_hash(everything), "n_total": len(everything), "n_jobs": len(jobs),
            "limit": limit, "chunk": chunk}
    _write_info(out, info)
    label = f"{model} {stage}"
    print(f"[v8] {label}: {len(jobs)} of {len(everything)} jobs, prompt hash {info['prompt_hash']}, into {out}",
          flush=True)
    t0, done, stop = time.time(), [0], threading.Event()
    beat = threading.Thread(target=_heartbeat, args=(stop, label, done, len(jobs), t0), daemon=True)
    beat.start()
    engine, counts, nudged = None, Counter(), 0
    try:
        if gen is None:
            try:
                gen, engine = make_engine(model)
            except Exception as e:
                info.update(status="load_failed", error=f"{type(e).__name__}: {e}", seconds=round(time.time() - t0))
                _write_info(out, info)
                print(f"[v8] {label} LOAD FAILED: {info['error'][:500]}", flush=True)
                raise
        for c0 in range(0, len(jobs), chunk):
            ch = jobs[c0:c0 + chunk]
            outs = _call(gen, [(j["messages"], j["seed"], j["max_tokens"], j["tools"]) for j in ch])
            again = [i for i, (j, (text, _)) in enumerate(zip(ch, outs))
                     if j["kind"] == "decide" and RN.parse_choice(text, fmt) is None]
            second = {}
            if again:   # one nudge each, as RN.decide: the reply, then RN.NUDGE, seed + 1, with the tools
                nudges = [(ch[i]["messages"] + [{"role": "assistant", "content": outs[i][0]},
                                                {"role": "user", "content": RN.NUDGE}],
                           ch[i]["seed"] + 1, ch[i]["max_tokens"], True) for i in again]
                second = dict(zip(again, _call(gen, nudges)))
            rows = defaultdict(list)
            for i, (j, (text, finish)) in enumerate(zip(ch, outs)):
                text2, finish2 = second.get(i, (None, None))
                rows[j["kind"]].append({k: v for k, v in j.items() if k != "messages"} | {
                    "text": text, "finish": finish, "first_try": i not in second, "text2": text2, "finish2": finish2})
            for kind, recs in rows.items():
                with open(out / FILES[kind], "a", encoding="utf-8") as f:
                    f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
                    f.flush()
                counts[kind] += len(recs)
            nudged += len(again)
            done[0] += len(ch)
            print(f"[v8] {label} {done[0]}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        stop.set()
        beat.join()
    info.update(status="done", counts=dict(counts), nudged=nudged, seconds=round(time.time() - t0), engine=engine)
    _write_info(out, info)
    print(f"[v8] {label} done: {dict(counts)}, {nudged} nudged ({info['seconds']}s)", flush=True)
    return info


# ---------------------------------------------------------------- a pod group

def hub_dir(hub=None) -> Path | None:
    """Where run_group frees model caches: `hub`, else $HF_HOME/hub when HF_HOME is set, else None (keep them)."""
    if hub is not None:
        return Path(hub)
    home = os.environ.get("HF_HOME")
    return Path(home) / "hub" if home else None


def run_group(stage: str, group: str, out, *, runner=subprocess.run, hub=None, models: list[str] | None = None,
              frozen: dict | None = None) -> int:
    """Run every model of the group in turn, each in its own process; see the module docstring."""
    out = Path(out)
    models = list(GROUPS[group] if models is None else models)
    if stage in ("main", "rewrite"):
        if frozen is None:
            frozen = json.loads(v8.FROZEN.read_text(encoding="utf-8"))
        for m in models:
            if not frozen["servable"][m]:
                print(f"[v8] {m} skipped: not servable in the frozen config (Change 5)", flush=True)
        models = [m for m in models if frozen["servable"][m]]
    hub = hub_dir(hub)
    print(f"[v8] group {group} {stage}: {' '.join(models)} -> {out}; "
          + (f"each model's cache is freed from {hub} after its run" if hub else "model caches kept (no HF_HOME)"),
          flush=True)
    first, failed = 0, []
    for k, m in enumerate(models, 1):
        print(f"[v8] {m} start ({k}/{len(models)})", flush=True)
        rc = runner([sys.executable, "-m", "persona_affect.v8_scale.pod", "run", "--stage", stage, "--model", m,
                     "--out", str(out / m)]).returncode
        if rc:
            print(f"[v8] {m} FAILED (exit {rc})", flush=True)
            first, failed = first or rc, failed + [m]
        if hub is not None:
            org, name = MODELS[m]["hf"].split("/")
            shutil.rmtree(hub / f"models--{org}--{name}", ignore_errors=True)
    print(f"[v8] group {group} {stage} done: {len(models) - len(failed)}/{len(models)} ok"
          + (f"; failed {' '.join(failed)}" if failed else ""), flush=True)
    return first


# ---------------------------------------------------------------- command line

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.v8_scale.pod",
                                 description="PA-V8 pod runner: the models of a group in turn, one vLLM process each")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one model's jobs in one stage (one vLLM process)")
    r.add_argument("--stage", required=True, choices=v8.STAGES)
    r.add_argument("--model", required=True, choices=list(MODELS))
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--limit", type=int, default=None, help="only the first N jobs, in prio order")
    g = sub.add_parser("group", help="every model of a pod group in turn, one `run` subprocess each")
    g.add_argument("--stage", required=True, choices=v8.STAGES)
    g.add_argument("--group", required=True, choices=list(GROUPS))
    g.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.stage, a.model, a.out, limit=a.limit)
    else:
        sys.exit(run_group(a.stage, a.group, a.out))


if __name__ == "__main__":
    main()
