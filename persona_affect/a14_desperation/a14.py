"""PA-A14 jobs and pod loop (persona_affect/plans/A14.md, Task 4).

One job is one episode: a (model, phase, split, wording, task, arm, sample) with key
`PA-A14|{model}|{phase}|{split}|{wording}|{task_id}|{arm}|{sample}`. An episode is up to ATTEMPTS submit-and-test
rounds: the model answers the chat so far, the code in its reply runs against the split's test in the harness's
sandbox, and a failure is answered with prompts.feedback_user (HEAD + the error + the arm's sentence). The episode ends
at its first pass, after the fifth failure, or when the next prompt leaves less than MIN_ROOM tokens to generate
(`context_full`). Harsh and neutral episodes of the same (task, split, wording, sample) share attempt 1 (their prompts
are identical until the first feedback): harsh copies neutral's attempt-1 reply and test result (`shared: true`).

    # local: jobs files (under persona_affect/a14_desperation/jobs/, never under results/)
    python -m persona_affect.a14_desperation.a14 jobs --model q27 --phase pilot --out persona_affect/a14_desperation/jobs/pilot_q27.jsonl
    python -m persona_affect.a14_desperation.a14 jobs --model gemma --phase main --wording A --tasks T61 \\
        [--primary conflicting|oneoff] [--remaining-from DIR ...] --out persona_affect/a14_desperation/jobs/main_gemma.jsonl
    # pod (pod.sh): one model, the whole jobs file, chunk by chunk
    /root/vllm/bin/python -m persona_affect.a14_desperation.a14 run --model q27 --jobs FILE --out DIR \\
        [--shard i/n] [--limit-tasks N] [--sandbox-user a14sandbox] [--test-python /usr/bin/python3]

The pod loop works on chunks of whole tasks (CHUNK episodes per chunk). Within a chunk, round k generates attempt k of
every live episode in one vLLM batch, then runs all their tests in a thread pool. Each round appends one line per
attempt (copied ones included) to attempts.jsonl; each chunk appends its finished episodes to episodes.jsonl and
rewrites run_info.json. A run resumes in its own out dir: episodes already in episodes.jsonl are not run again.

Imports: stdlib, prompts, harness, tasks and common.util (numpy) at import time; vLLM only inside VllmBackend, so the
tests import this module without vLLM. Model-written code never runs as root: run() refuses a root process without a
sandbox user, and checks that the sandbox works before it loads the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

from persona_affect.a14_desperation import prompts as P
from persona_affect.a14_desperation.harness import error_text, extract_code, run_many, run_submission
from persona_affect.a14_desperation.tasks import CAPS, load_meta, load_tasks, task_set
from persona_affect.common.util import seed

MODELS = {
    "q27": {"hf": "Qwen/Qwen3.6-27B", "rev": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9", "mml": 24576,
            "util": 0.88, "serving": "bf16", "max_tokens": 2048},
    "gemma": {"hf": "RedHatAI/gemma-4-31B-it-FP8-dynamic", "rev": "d4ab4f579dd3516f97d8a6a4c98d0653480bad15",
              "mml": 24576, "util": 0.92, "serving": "fp8-dynamic", "max_tokens": 2048},   # FP8 weights, A14 only
}
SAMPLING = dict(temperature=0.7, top_p=0.95, top_k=20, min_p=0.0)     # vigilante pod_run.DECODING
MIN_ROOM = 512           # an attempt with fewer tokens of room is not generated: the episode ends context_full
MARGIN = 16              # max_tokens = min(max_tokens, max_model_len - n_prompt - MARGIN)
ATTEMPTS = 5
CHUNK = {"q27": 400, "gemma": 200}
TEST_TIMEOUT = 10.0
HEARTBEAT = 300          # seconds; the launcher aborts a job whose log is silent for 30 min
SPLITS = ("original", "conflicting", "oneoff")
JOBS_DIR = Path(__file__).parent / "jobs"
DROPPED_DIRS = ("results", "logs", "prep", "reference")     # the launcher's sync drops paths through these
SANDBOX_CHECK = ({"task_id": "_sandbox_check", "entry_point": "f", "prompt": "def f():",
                  "tests": {"check": "def check(candidate):\n    assert candidate() == 1\n"}},
                 "import sys\ndef f():\n    print(sys.version.split()[0])\n    return 1\n")


def key(model: str, phase: str, split: str, wording: str, task_id: str, arm: str, sample: int) -> str:
    return f"PA-A14|{model}|{phase}|{split}|{wording}|{task_id}|{arm}|{sample}"


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- jobs

def build_jobs(model: str, phase: str, task_ids: list[str], wordings: list[str], samples: dict[str, int],
               arms: tuple[str, ...]) -> list[dict]:
    """One job per (task, wording, split, arm, sample): {key, model, phase, split, wording, task_id, arm, sample,
    share}, where share is the neutral partner's key for a harsh job (when neutral is among `arms`), else None.

    Tasks are shuffled with random.Random(f"PA-A14|{model}|{phase}") and each task's jobs stay together, so a pod cut
    short loses whole random tasks, not the end of the dataset. `samples` maps split -> samples per task and arm."""
    if model not in MODELS:
        raise ValueError(f"build_jobs: unknown model {model!r}; known: {sorted(MODELS)}")
    if not set(arms) <= set(P.ARMS) or not arms:
        raise ValueError(f"build_jobs: arms must be among {P.ARMS}, got {arms!r}")
    if not set(wordings) <= set(P.WORDINGS) or not wordings:
        raise ValueError(f"build_jobs: wordings must be among {sorted(P.WORDINGS)}, got {wordings!r}")
    if not set(samples) <= set(SPLITS) or not samples:
        raise ValueError(f"build_jobs: splits must be among {SPLITS}, got {sorted(samples)}")
    ids = list(task_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("build_jobs: duplicate task ids")
    random.Random(f"PA-A14|{model}|{phase}").shuffle(ids)
    jobs = []
    for t in ids:
        for w in wordings:
            for split, n in samples.items():
                for arm in arms:
                    for s in range(n):
                        share = key(model, phase, split, w, t, "neutral", s) if arm == "harsh" and "neutral" in arms \
                            else None
                        jobs.append({"key": key(model, phase, split, w, t, arm, s), "model": model, "phase": phase,
                                     "split": split, "wording": w, "task_id": t, "arm": arm, "sample": s,
                                     "share": share})
    return jobs


def _chunks(jobs: list[dict], size: int) -> list[list[dict]]:
    """`jobs` cut into chunks of whole tasks (in order of first appearance) of at most `size` jobs; a task with more
    jobs than `size` is a chunk of its own."""
    by_task: dict[str, list[dict]] = {}
    for j in jobs:
        by_task.setdefault(j["task_id"], []).append(j)
    chunks, cur = [], []
    for group in by_task.values():
        if cur and len(cur) + len(group) > size:
            chunks.append(cur)
            cur = []
        cur = cur + group
    if cur:
        chunks.append(cur)
    return chunks


def _parse_shard(shard: str) -> tuple[int, int]:
    try:
        i, n = (int(x) for x in shard.split("/"))
    except ValueError:
        raise ValueError(f"shard must be i/n, got {shard!r}") from None
    if not 0 <= i < n:
        raise ValueError(f"shard must be i/n with 0 <= i < n, got {shard!r}")
    return i, n


# ---------------------------------------------------------------- the model

def clean_reply(text: str) -> str:
    """The reply without any thinking: everything through the last `</think>` dropped, then a leading `thought` line
    (Gemma's empty thought channel once its special tokens are skipped), then surrounding whitespace."""
    text = text.split("</think>")[-1]
    rest = text.lstrip()
    if rest.startswith("thought\n"):
        text = rest[len("thought\n"):]
    return text.strip()


def _mkey(messages: list[dict]) -> str:
    return json.dumps(messages, sort_keys=True)


class VllmBackend:
    """vLLM offline engine for one of MODELS, rendered with the model's own chat template, thinking off.

    n_prompt(messages) -> prompt tokens; generate([(messages, seed, max_tokens)]) -> [(reply, finish_reason, n_gen)]
    (reply through clean_reply); .info = {hf, rev, vllm, template_sha256, quantization, dtype} for run_info.json."""

    def __init__(self, model: str):
        import vllm
        from vllm import LLM, SamplingParams, TokensPrompt
        self._sampling, self._tokens_prompt = SamplingParams, TokensPrompt
        cfg = MODELS[model]
        kw = dict(model=cfg["hf"], revision=cfg["rev"], dtype="bfloat16", max_model_len=cfg["mml"],
                  gpu_memory_utilization=cfg["util"], seed=0, enable_prefix_caching=True)
        try:
            self.llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
        except Exception as e:   # some multimodal configs reject a modality key; retry with images only (as pod_run)
            print(f"[a14] LLM init with image+video limits failed ({type(e).__name__}: {str(e)[:200]}); retrying "
                  f"with images only", flush=True)
            self.llm = LLM(**kw, limit_mm_per_prompt={"image": 0})
        self.tok = self.llm.get_tokenizer()
        tmpl = self.tok.chat_template if isinstance(self.tok.chat_template, str) else json.dumps(self.tok.chat_template)
        self.info = {"hf": cfg["hf"], "rev": cfg["rev"], "vllm": vllm.__version__,
                     "template_sha256": hashlib.sha256(tmpl.encode()).hexdigest(), **self._served()}
        self._ids_of: dict[str, list[int]] = {}

    def _served(self) -> dict:
        """What the engine reports it loaded (the FP8 check reads quantization), best effort."""
        mc = getattr(self.llm, "model_config", None) or getattr(getattr(self.llm, "llm_engine", None), "model_config",
                                                                 None)
        q, d = getattr(mc, "quantization", None), getattr(mc, "dtype", None)
        return {"quantization": None if q is None else str(q), "dtype": None if d is None else str(d)}

    def _ids(self, messages: list[dict]) -> list[int]:
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def n_prompt(self, messages: list[dict]) -> int:
        ids = self._ids(messages)
        self._ids_of[_mkey(messages)] = ids         # generate() reuses it instead of tokenizing again
        return len(ids)

    def generate(self, batch: list[tuple[list[dict], int, int]]) -> list[tuple[str, str, int]]:
        prompts, params = [], []
        for messages, s, max_tokens in batch:
            ids = self._ids_of.pop(_mkey(messages), None) or self._ids(messages)
            prompts.append(self._tokens_prompt(prompt_token_ids=ids))
            params.append(self._sampling(**SAMPLING, max_tokens=max_tokens, seed=s, skip_special_tokens=True))
        self._ids_of.clear()
        res = self.llm.generate(prompts, params, use_tqdm=False)
        return [(clean_reply(r.outputs[0].text), r.outputs[0].finish_reason, len(r.outputs[0].token_ids)) for r in res]


# ---------------------------------------------------------------- the pod loop

def _git_rev() -> str | None:
    if os.environ.get("GIT_REV"):                   # infra/launch_run.py exports it on the pod
        return os.environ["GIT_REV"]
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True,
                           timeout=10)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _sandbox_check(python: str, user: str | None) -> str:
    """Run a trivial passing submission through the sandbox; SystemExit unless it passes. Returns the test Python's
    version (printed by the check), so a broken sandbox stops the run before the model loads, not after a chunk of
    `error` attempts."""
    task, code = SANDBOX_CHECK
    r = run_submission(task, "check", code, python=python, user=user)
    if r["status"] != "pass":
        raise SystemExit(f"[a14] the test sandbox does not work (python {python}, user {user}): status {r['status']}, "
                         f"rc {r['rc']}: {r['stderr'][-800:]}")
    return r["stdout"].splitlines()[0].strip()


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _append(path: Path, rows: list[dict]) -> None:
    if rows:
        with open(path, "a", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
            f.flush()


def _run_chunk(chunk: list[dict], *, cfg: dict, tasks: dict, backend, done: dict, python: str, user: str | None,
               workers: int, out: Path, stats: dict, label: str) -> list[dict]:
    """The episodes of one chunk, run round by round. Updates `stats` (secs_gen, secs_test, attempts, attempts_copied,
    gen_tokens, prompt_tokens, context_full)."""
    eps = [dict(j, attempts=[], end=None, pass_attempt=None) for j in chunk]
    by_key = {e["key"]: e for e in eps}
    live = list(eps)
    for k in range(1, ATTEMPTS + 1):
        gen, copy, full = [], [], 0
        for e in live:
            if k == 1 and e.get("share") and (e["share"] in by_key or e["share"] in done):
                copy.append(e)                       # harsh: attempt 1 is its neutral partner's
                continue
            task = tasks[e["task_id"]]
            msgs = P.messages_for(task, e["split"], e["wording"], e["arm"],
                                  [(a["reply"], a["error_shown"]) for a in e["attempts"]])
            n = backend.n_prompt(msgs)
            room = cfg["mml"] - n - MARGIN
            if room < MIN_ROOM:
                e["end"] = "context_full"
                full += 1
                continue
            gen.append((e, msgs, seed(f"{e['key']}|a{k}"), min(cfg["max_tokens"], room), n))
        t0 = time.time()
        outs = backend.generate([(m, s, mt) for _, m, s, mt, _ in gen]) if gen else []
        t1 = time.time()
        codes = [extract_code(reply, tasks[e["task_id"]]["entry_point"]) for (e, *_), (reply, _, _) in zip(gen, outs)]
        results = run_many([(tasks[e["task_id"]], e["split"], code) for (e, *_), (code, _) in zip(gen, codes)],
                           python=python, user=user, workers=workers, timeout=TEST_TIMEOUT)
        t2 = time.time()
        new = []
        for (e, _, s, _, n), (reply, finish, n_gen), (code, src), r in zip(gen, outs, codes, results):
            a = {"k": k, "seed": s, "reply": reply, "finish": finish, "n_prompt": n, "n_gen": n_gen, "code": code,
                 "code_src": src, "status": r["status"], "rc": r["rc"], "checked": r["checked"], "secs": r["secs"],
                 "error_shown": error_text(r, TEST_TIMEOUT), "shared": False}
            e["attempts"].append(a)
            new.append((e, a))
            stats["gen_tokens"] += n_gen
            stats["prompt_tokens"] += n
        for e in copy:
            partner = by_key.get(e["share"]) or done[e["share"]]
            first = partner["attempts"][0] if partner["attempts"] else None
            if first is not None and first["k"] == 1:
                a = dict(first, shared=True)
                e["attempts"].append(a)
                new.append((e, a))
            else:                                    # the partner had no attempt 1: the same prompt had no room
                e["end"] = "context_full"
                full += 1
        _append(out / "attempts.jsonl", [{"key": e["key"], **a} for e, a in new])
        for e, a in new:
            if a["status"] == "pass":
                e["end"], e["pass_attempt"] = "pass", k
        live = [e for e in live if e["end"] is None]
        if k == ATTEMPTS:
            for e in live:
                e["end"] = "max_attempts"
        stats["secs_gen"] += t1 - t0
        stats["secs_test"] += t2 - t1
        stats["attempts"] += len(gen)
        stats["attempts_copied"] += sum(a["shared"] for _, a in new)
        stats["context_full"] += full
        passed = sum(a["status"] == "pass" for _, a in new)
        print(f"[a14] {label} round {k}: generated {len(gen)} in {t1 - t0:.0f}s, tested in {t2 - t1:.0f}s, copied "
              f"{len(copy)}, passed {passed}, context_full {full}, live {len(live) if k < ATTEMPTS else 0}", flush=True)
        if not live:
            break
    return eps


def run(model: str, jobs: Path, out: Path, *, backend=None, tasks: dict | None = None, python: str = "/usr/bin/python3",
        user: str | None = "a14sandbox", shard: str | None = None, limit_tasks: int | None = None,
        workers: int | None = None) -> dict:
    """Run the jobs file `jobs` (all of one model) into `out` and return run_info.

    Selection: the file's tasks in order, those of shard i/n (seed(task_id) % n == i), the first `limit_tasks` of them;
    jobs whose key is already in out/episodes.jsonl are skipped (a task whose episodes are all there is skipped whole).
    backend: anything with n_prompt / generate / info (default: VllmBackend(model), loaded only if there is work).
    tasks: {task_id: task} (default: tasks.load_tasks(), data/tasks.jsonl). python / user: the test interpreter and the
    sandbox user (None: the current user, refused for root). run_info.json, rewritten after each chunk: model_cfg,
    backend, sampling, secs_load, secs_gen, secs_test, attempts (generated), attempts_copied (harsh's shared attempt 1),
    episodes, context_full, gen_tokens, prompt_tokens, test_python, test_python_version, sandbox_user, workers,
    chunks_done, chunks_total, shard, limit_tasks, jobs, git_rev.
    """
    if user is None and hasattr(os, "geteuid") and os.geteuid() == 0:
        raise SystemExit("[a14] refusing to run model-written code as root: give a sandbox user (--sandbox-user)")
    cfg = MODELS[model]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    workers = workers or max(4, min(32, (os.cpu_count() or 4) - 2))
    all_jobs = read_jsonl(jobs)
    other = sorted({j["model"] for j in all_jobs} - {model})
    if other:
        raise SystemExit(f"[a14] {jobs} holds jobs of {other}, not only {model}")
    ids = list(dict.fromkeys(j["task_id"] for j in all_jobs))
    if shard:
        i, n = _parse_shard(shard)
        ids = [t for t in ids if seed(t) % n == i]
    if limit_tasks is not None:
        ids = ids[:limit_tasks]
    done = {e["key"]: e for e in read_jsonl(out / "episodes.jsonl")} if (out / "episodes.jsonl").exists() else {}
    keep = set(ids)
    todo = [j for j in all_jobs if j["task_id"] in keep and j["key"] not in done]
    tasks = load_tasks() if tasks is None else tasks
    missing = sorted({j["task_id"] for j in todo} - set(tasks))
    if missing:
        raise SystemExit(f"[a14] {len(missing)} task ids of the jobs are not in the task data: {missing[:5]}")
    chunks = _chunks(todo, CHUNK[model])
    t_start = time.time()
    print(f"[a14] {model}: {len(all_jobs)} jobs, {len(ids)} tasks selected, {len(done)} episodes already in {out}, "
          f"{len(todo)} to run in {len(chunks)} chunk(s)", flush=True)
    stats = dict.fromkeys(("secs_gen", "secs_test"), 0.0) | dict.fromkeys(
        ("attempts", "attempts_copied", "episodes", "context_full", "gen_tokens", "prompt_tokens"), 0)
    info = {"model": model, "model_cfg": dict(cfg), "backend": getattr(backend, "info", None), "sampling": SAMPLING,
            "min_room": MIN_ROOM, "max_attempts": ATTEMPTS, "chunk": CHUNK[model], "secs_load": 0.0, **stats,
            "test_python": python, "test_python_version": None, "sandbox_user": user, "workers": workers,
            "chunks_done": 0, "chunks_total": len(chunks), "shard": shard, "limit_tasks": limit_tasks,
            "jobs": str(jobs), "git_rev": _git_rev()}
    if not chunks:
        return info
    stop = threading.Event()

    def beat():
        while not stop.wait(HEARTBEAT):
            print(f"[a14] heartbeat ({time.time() - t_start:.0f}s)", flush=True)
    threading.Thread(target=beat, daemon=True).start()
    try:
        info["test_python_version"] = _sandbox_check(python, user)
        print(f"[a14] sandbox ok: {python} {info['test_python_version']} as {user or 'the current user'}", flush=True)
        if backend is None:
            t = time.time()
            backend = VllmBackend(model)
            info["secs_load"] = round(time.time() - t, 1)
            print(f"[a14] loaded {cfg['hf']}@{cfg['rev'][:10]} in {info['secs_load']:.0f}s: {backend.info}", flush=True)
        info["backend"] = backend.info
        for c, chunk in enumerate(chunks, 1):
            eps = _run_chunk(chunk, cfg=cfg, tasks=tasks, backend=backend, done=done, python=python, user=user,
                             workers=workers, out=out, stats=stats, label=f"chunk {c}/{len(chunks)}")
            _append(out / "episodes.jsonl", eps)
            stats["episodes"] += len(eps)
            info.update({k: round(v, 1) if isinstance(v, float) else v for k, v in stats.items()}, chunks_done=c,
                        secs_total=round(time.time() - t_start, 1))
            _write_json(out / "run_info.json", info)
            print(f"[a14] chunk {c}/{len(chunks)} done: {stats['episodes']} episodes, {stats['attempts']} generated "
                  f"attempts, {time.time() - t_start:.0f}s", flush=True)
    finally:
        stop.set()
    return info


# ---------------------------------------------------------------- CLI

def _jobs_cmd(a, ap) -> int:
    out = Path(a.out)
    if any(part in DROPPED_DIRS for part in out.parts):
        ap.error(f"--out must not sit under a {'/'.join(DROPPED_DIRS)} directory (the launcher's sync drops it): "
                 f"{out}; use {JOBS_DIR}")
    meta = load_meta()
    if a.phase == "main":
        if not (a.wording and a.tasks):
            ap.error("--phase main needs --wording and --tasks")
        ids, wordings = task_set(meta, CAPS[a.tasks]), [a.wording]
        samples, arms = {a.primary: 3, "original": 1}, P.ARMS
    else:
        if a.wording or a.tasks or a.primary != "conflicting":
            ap.error(f"--wording, --tasks and --primary apply to --phase main only, not {a.phase}")
        ids, wordings, arms = meta["pilot"], ["A", "B"], ("neutral",)
        samples = {"conflicting": 3, "original": 1} if a.phase == "pilot" else {"oneoff": 3}
    jobs = build_jobs(a.model, a.phase, ids, wordings, samples, arms)
    have = set()
    for d in a.remaining_from:
        path = Path(d) / "episodes.jsonl"
        if not path.exists():
            ap.error(f"--remaining-from {d}: no episodes.jsonl there")
        have |= {e["key"] for e in read_jsonl(path)}
    n_all = len(jobs)
    jobs = [j for j in jobs if j["key"] not in have]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(j) + "\n" for j in jobs), encoding="utf-8")
    print(f"[a14] wrote {len(jobs)} jobs ({n_all - len(jobs)} already done) for {a.model} {a.phase}: "
          f"{len({j['task_id'] for j in jobs})} tasks, wordings {wordings}, splits {samples}, arms {list(arms)} "
          f"-> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.a14_desperation.a14",
                                 description="PA-A14: build jobs files (local) and run them (pod)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs", help="write a jobs file")
    j.add_argument("--model", choices=sorted(MODELS), required=True)
    j.add_argument("--phase", choices=["pilot", "oneoff", "main"], required=True)
    j.add_argument("--wording", choices=sorted(P.WORDINGS))
    j.add_argument("--tasks", choices=sorted(CAPS))
    j.add_argument("--primary", choices=["conflicting", "oneoff"], default="conflicting")
    j.add_argument("--remaining-from", nargs="+", action="extend", default=[], metavar="DIR",
                   help="drop the jobs whose episodes are in DIR/episodes.jsonl (top-up runs)")
    j.add_argument("--out", required=True, help=f"the jobs file, under {JOBS_DIR}")
    r = sub.add_parser("run", help="run a jobs file on this machine's GPU (pod)")
    r.add_argument("--model", choices=sorted(MODELS), required=True)
    r.add_argument("--jobs", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--shard", default=None, help="i/n: the tasks with seed(task_id) %% n == i")
    r.add_argument("--limit-tasks", type=int, default=None)
    r.add_argument("--sandbox-user", default="a14sandbox", help="run the tests as this user ('' = the current user)")
    r.add_argument("--test-python", default="/usr/bin/python3")
    r.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "jobs":
        return _jobs_cmd(a, ap)
    if a.shard is not None:
        try:
            _parse_shard(a.shard)
        except ValueError as e:
            ap.error(str(e))
    info = run(a.model, Path(a.jobs), Path(a.out), python=a.test_python, user=a.sandbox_user or None, shard=a.shard,
               limit_tasks=a.limit_tasks, workers=a.workers)
    print(f"[a14] done: {info.get('episodes', 0)} episodes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
