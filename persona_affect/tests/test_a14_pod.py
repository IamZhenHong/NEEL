"""PA-A14 Task 4 tests: jobs, the pod loop, the pod scripts (persona_affect/a14_desperation/a14.py, pod.sh, launch.sh).

Run from the repo root: `uv run pytest persona_affect/tests/test_a14_pod.py -q`.
No network, no GPU, no pod. The model is a fake (`Fake` below, or the fake `vllm` in a14_desperation/fakevllm); the
tests run only the trivial code of the fakes, inside the harness's sandbox. launch.sh is never executed: it is checked
with `bash -n` and read as text. pod.sh runs only with SKIP_SETUP, a fake vllm and a scratch copy of the code, or on
paths that exit before any setup.
"""
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from persona_affect.a14_desperation import a14 as A
from persona_affect.a14_desperation import prompts as P
from test_a14 import TASK, read_jsonl, write_jsonl
from test_a14_harness import GOOD, STATE

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "persona_affect" / "a14_desperation"


@pytest.fixture(autouse=True)
def term_handlers(monkeypatch):
    """main() installs util's TERM/HUP handlers; in-process calls here must not replace pytest's own."""
    calls = []
    monkeypatch.setattr(A.util, "install_term_handlers", lambda: calls.append(1) or {})
    return calls


class Fake:
    """The brief's test backend: records each batch; GOOD code, or STATE for the told arm from attempt 2 on."""

    def __init__(self, n_prompt=lambda m: 100):
        self._n_prompt = n_prompt
        self.batches = []
        self.info = {"hf": "fake", "rev": "0", "vllm": "fake", "template_sha256": "0"}

    def n_prompt(self, messages):
        return self._n_prompt(messages)

    def generate(self, batch):
        self.batches.append(batch)
        out = []
        for messages, _seed, _max_tokens in batch:
            k = 1 + sum(m["role"] == "assistant" for m in messages)
            code = STATE if messages[0]["content"] == P.SYSTEM["told"] and k >= 2 else GOOD
            out.append(("Here you go.\n```python\n" + code + "```", "stop", 40))
        return out


# ------------------------------------------------------------------ the brief's tests, verbatim

def test_build_jobs_counts_and_pairs():
    jobs = A.build_jobs("q27", "main", [f"t{i}" for i in range(10)], ["A"], {"conflicting": 3, "original": 1}, P.ARMS)
    keys = {j["key"]: j for j in jobs}
    assert len(jobs) == 240 == len(keys) and jobs[0]["key"].startswith("PA-A14|q27|main|")   # 10 × 6 arms × 4
    for j in jobs:
        n = keys.get(j["share"]) if j["share"] else None
        assert (j["arm"] == "harsh") == (n is not None)
        assert n is None or (n["arm"], n["task_id"], n["split"], n["sample"]) == ("neutral", j["task_id"], j["split"], j["sample"])
    ids20 = [f"t{i}" for i in range(20)]
    assert len(A.build_jobs("gemma", "pilot", ids20, ["A", "B"], {"conflicting": 3, "original": 1}, ("neutral",))) == 160


def test_run_rounds(tmp_path):
    jobs = tmp_path / "jobs.jsonl"
    write_jsonl(jobs, A.build_jobs("q27", "main", ["t0"], ["A"], {"conflicting": 3, "original": 1}, P.ARMS))
    fake, kw = Fake(), dict(tasks={"t0": TASK}, python=sys.executable, user=None)
    info = A.run("q27", jobs, tmp_path / "out", backend=fake, **kw)
    eps = {e["key"]: e for e in read_jsonl(tmp_path / "out" / "episodes.jsonl")}
    assert len(eps) == 24 == info["episodes"]
    assert info["model_cfg"] == A.MODELS["q27"] and info["backend"] == fake.info and info["model_cfg"]["max_tokens"] == 2048
    for e in eps.values():
        want = ("pass", 1) if e["split"] == "original" else ("pass", 2) if e["arm"] == "told" else ("max_attempts", None)
        assert (e["end"], e["pass_attempt"]) == want and all(a["finish"] == "stop" for a in e["attempts"])
        if e["arm"] == "harsh":
            assert e["attempts"][0]["shared"] and e["attempts"][0]["reply"] == eps[e["share"]]["attempts"][0]["reply"]
    assert len(fake.batches[0]) == 20                       # the 4 harsh episodes are copied, not generated
    assert sum(m[0]["content"] == P.SYSTEM["goal"] for m, _, _ in fake.batches[0]) == 4
    assert sum(m[-1]["content"].endswith(P.SENTENCE["harsh"]) for m, _, _ in fake.batches[1]) == 3
    assert len(read_jsonl(tmp_path / "out" / "attempts.jsonl")) == sum(len(e["attempts"]) for e in eps.values())
    fake2 = Fake(); A.run("q27", jobs, tmp_path / "out", backend=fake2, **kw)
    assert fake2.batches == []                             # resume: nothing regenerated


def test_context_full(tmp_path):
    jobs = tmp_path / "j.jsonl"; write_jsonl(jobs, A.build_jobs("q27", "main", ["t0"], ["A"], {"conflicting": 1}, P.ARMS))
    A.run("q27", jobs, tmp_path / "o", backend=Fake(n_prompt=lambda m: 24_200), tasks={"t0": TASK},   # room 360 < 512
          python=sys.executable, user=None)
    assert all(e["end"] == "context_full" and e["attempts"] == [] for e in read_jsonl(tmp_path / "o" / "episodes.jsonl"))


def test_run_refuses_root_without_sandbox(tmp_path, monkeypatch):
    jobs = tmp_path / "j.jsonl"; write_jsonl(jobs, A.build_jobs("q27", "main", ["t0"], ["A"], {"conflicting": 1}, ("neutral",)))
    monkeypatch.setattr(A.os, "geteuid", lambda: 0)
    with pytest.raises(SystemExit):
        A.run("q27", jobs, tmp_path / "o", backend=Fake(), tasks={"t0": TASK}, python=sys.executable, user=None)


def test_clean_reply():
    assert A.clean_reply("<think>\nhm\n</think>\n\nHello") == "Hello" and A.clean_reply("thought\nHello\n") == "Hello"


# ------------------------------------------------------------------ pins

def test_models_sampling_and_limits_pinned():
    assert A.MODELS == {
        "q27": {"hf": "Qwen/Qwen3.6-27B", "rev": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9", "mml": 24576,
                "util": 0.88, "serving": "bf16", "max_tokens": 2048},
        "gemma": {"hf": "RedHatAI/gemma-4-31B-it-FP8-dynamic", "rev": "d4ab4f579dd3516f97d8a6a4c98d0653480bad15",
                  "mml": 24576, "util": 0.92, "serving": "fp8-dynamic", "max_tokens": 2048}}
    assert A.SAMPLING == {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
    assert (A.MIN_ROOM, A.ATTEMPTS, A.CHUNK, A.MARGIN) == (512, 5, {"q27": 400, "gemma": 200}, 16)


# ------------------------------------------------------------------ build_jobs and chunks

def test_build_jobs_keys_fields_and_shuffle():
    jobs = A.build_jobs("gemma", "oneoff", ["a", "b", "c"], ["B"], {"oneoff": 2}, ("neutral", "harsh"))
    j = jobs[0]
    assert set(j) == {"key", "model", "phase", "split", "wording", "task_id", "arm", "sample", "share"}
    for j in jobs:
        assert j["key"] == f"PA-A14|gemma|oneoff|{j['split']}|{j['wording']}|{j['task_id']}|{j['arm']}|{j['sample']}"
    order = list(dict.fromkeys(j["task_id"] for j in jobs))
    want = ["a", "b", "c"]
    random.Random("PA-A14|gemma|oneoff").shuffle(want)
    assert order == want                                    # tasks shuffled with the model|phase seed, kept whole
    assert all(j["share"] is None for j in A.build_jobs("q27", "main", ["a"], ["A"], {"conflicting": 1}, ("harsh",)))
    for bad in (dict(model="x"), dict(arms=("nope",)), dict(samples={"weird": 1}), dict(wordings=["C"]),
                dict(task_ids=["a", "a"])):
        kw = dict(model="q27", phase="main", task_ids=["a"], wordings=["A"], samples={"conflicting": 1}, arms=P.ARMS) | bad
        with pytest.raises(ValueError):
            A.build_jobs(**kw)


def test_chunks_keep_tasks_whole():
    jobs = A.build_jobs("q27", "main", [f"t{i}" for i in range(5)], ["A"], {"conflicting": 3, "original": 1}, P.ARMS)
    chunks = A._chunks(jobs, 50)                            # 24 episodes per task: two tasks per chunk
    assert [len(c) for c in chunks] == [48, 48, 24]
    for c in chunks:
        ids = [j["task_id"] for j in c]
        assert all(sum(j["task_id"] == t for j in jobs) == ids.count(t) for t in set(ids))
    assert [len(c) for c in A._chunks(jobs, 10)] == [24] * 5   # a task bigger than a chunk is one chunk
    assert [len(c) for c in A._chunks(jobs, 48)] == [48, 48, 24]   # exactly full is still one chunk


def test_clean_reply_edges():
    assert A.clean_reply("thoughtful answer") == "thoughtful answer"     # only a whole `thought` line goes
    assert A.clean_reply("\nthought\nHi there\n") == "Hi there"
    assert A.clean_reply("a</think>b</think>\n c ") == "c"                # through the last </think>
    assert A.clean_reply("Plain reply") == "Plain reply"


# ------------------------------------------------------------------ run: sharing, resume, shards, run_info

def test_attempt_records_and_round_logging(tmp_path):
    jobs = tmp_path / "j.jsonl"
    write_jsonl(jobs, A.build_jobs("q27", "main", ["t0"], ["A"], {"conflicting": 1}, ("neutral", "harsh")))
    fake = Fake()
    A.run("q27", jobs, tmp_path / "o", backend=fake, tasks={"t0": TASK}, python=sys.executable, user=None)
    eps = {e["arm"]: e for e in read_jsonl(tmp_path / "o" / "episodes.jsonl")}
    n, h = eps["neutral"], eps["harsh"]
    a = n["attempts"][0]
    assert set(a) == {"k", "seed", "reply", "finish", "n_prompt", "n_gen", "code", "code_src", "status", "rc",
                      "checked", "secs", "error_shown", "shared"}
    assert (a["k"], a["seed"], a["n_prompt"], a["n_gen"], a["code"], a["code_src"], a["status"], a["shared"]) == \
        (1, A.seed(f"{n['key']}|a1"), 100, 40, GOOD, "fence", "fail", False)
    assert a["error_shown"].endswith("AssertionError") and "solution_test.py" in a["error_shown"]
    assert h["attempts"][0] == dict(a, shared=True)         # harsh's attempt 1 is neutral's, marked shared
    assert [x["k"] for x in h["attempts"]] == [1, 2, 3, 4, 5] and h["attempts"][1]["seed"] == A.seed(f"{h['key']}|a2")
    assert not any(x["shared"] for x in h["attempts"][1:])
    second = [m for m, _, _ in fake.batches[1]]
    msgs = next(m for m in second if m[-1]["content"].endswith(P.SENTENCE["harsh"]))
    assert msgs == P.messages_for(TASK, "conflicting", "A", "harsh", [(a["reply"], a["error_shown"])])
    assert [len(b) for b in fake.batches] == [1, 2, 2, 2, 2]  # round 1: neutral only
    assert all(mt == 2048 for b in fake.batches for _, _, mt in b)
    lines = read_jsonl(tmp_path / "o" / "attempts.jsonl")
    assert len(lines) == 10 and all(set(x) >= {"key", "k", "status", "shared"} for x in lines)
    assert sum(x["shared"] for x in lines) == 1
    info = json.loads((tmp_path / "o" / "run_info.json").read_text())
    assert (info["attempts"], info["attempts_copied"], info["gen_tokens"]) == (9, 1, 9 * 40)   # copies cost nothing


def test_harsh_ends_context_full_with_its_partner(tmp_path):
    """The partner had no room for attempt 1, so harsh (the same prompt) copies nothing and ends with it: even if
    the backend would report room later, harsh is never generated."""
    jobs = tmp_path / "j.jsonl"
    write_jsonl(jobs, A.build_jobs("q27", "main", ["t0"], ["A"], {"conflicting": 1}, ("neutral", "harsh")))
    calls = []
    fake = Fake(n_prompt=lambda m: calls.append(m) or (24_200 if len(calls) == 1 else 100))
    A.run("q27", jobs, tmp_path / "o", backend=fake, tasks={"t0": TASK}, python=sys.executable, user=None)
    eps = read_jsonl(tmp_path / "o" / "episodes.jsonl")
    assert [(e["arm"], e["end"], e["attempts"]) for e in eps] == [("neutral", "context_full", []),
                                                                  ("harsh", "context_full", [])]
    assert fake.batches == [] and len(calls) == 1


def test_max_tokens_shrinks_with_the_room(tmp_path):
    jobs = tmp_path / "j.jsonl"
    write_jsonl(jobs, A.build_jobs("gemma", "main", ["t0"], ["A"], {"conflicting": 1}, ("neutral",)))
    fake = Fake(n_prompt=lambda m: 24576 - 16 - 600 if len(m) > 1 else 100)       # attempt 2+: 600 tokens of room
    A.run("gemma", jobs, tmp_path / "o", backend=fake, tasks={"t0": TASK}, python=sys.executable, user=None)
    assert [b[0][2] for b in fake.batches] == [2048, 600, 600, 600, 600]


def test_harsh_without_a_partner_generates_its_own(tmp_path):
    jobs = A.build_jobs("q27", "main", ["t0"], ["A"], {"conflicting": 1}, ("neutral", "harsh"))
    path = tmp_path / "j.jsonl"
    write_jsonl(path, [j for j in jobs if j["arm"] == "harsh"])      # the partner is in no job file of this run
    fake = Fake()
    A.run("q27", path, tmp_path / "o", backend=fake, tasks={"t0": TASK}, python=sys.executable, user=None)
    (e,) = read_jsonl(tmp_path / "o" / "episodes.jsonl")
    assert len(fake.batches[0]) == 1 and not e["attempts"][0]["shared"]


def test_harsh_copies_a_partner_done_earlier(tmp_path):
    jobs = A.build_jobs("q27", "main", ["t0"], ["A"], {"conflicting": 1}, ("neutral", "harsh"))
    path = tmp_path / "j.jsonl"
    write_jsonl(path, jobs)
    out = tmp_path / "o"
    out.mkdir()
    att = {"k": 1, "seed": 7, "reply": "R1", "finish": "stop", "n_prompt": 5, "n_gen": 2, "code": None,
           "code_src": None, "status": "no_code", "rc": None, "checked": False, "secs": 0.0,
           "error_shown": "No code block found in your reply.", "shared": False}
    neutral = next(j for j in jobs if j["arm"] == "neutral")
    write_jsonl(out / "episodes.jsonl", [dict(neutral, attempts=[att], end="max_attempts", pass_attempt=None)])
    fake = Fake()
    A.run("q27", path, out, backend=fake, tasks={"t0": TASK}, python=sys.executable, user=None)
    eps = read_jsonl(out / "episodes.jsonl")
    assert len(eps) == 2 and eps[1]["arm"] == "harsh" and eps[1]["attempts"][0] == dict(att, shared=True)
    assert len(fake.batches) == 4 and fake.batches[0][0][0][-1]["content"].endswith(P.SENTENCE["harsh"])


def test_shard_limit_and_run_info(tmp_path):
    ids = [f"t{i}" for i in range(6)]
    jobs = tmp_path / "j.jsonl"
    write_jsonl(jobs, A.build_jobs("q27", "pilot", ids, ["A"], {"original": 1}, ("neutral",)))
    tasks = {t: dict(TASK, task_id=t) for t in ids}
    info = A.run("q27", jobs, tmp_path / "s", backend=Fake(), tasks=tasks, python=sys.executable, user=None,
                 shard="1/2")
    got = {e["task_id"] for e in read_jsonl(tmp_path / "s" / "episodes.jsonl")}
    assert got == {t for t in ids if A.seed(t) % 2 == 1} and info["episodes"] == len(got)
    info = A.run("q27", jobs, tmp_path / "l", backend=Fake(), tasks=tasks, python=sys.executable, user=None,
                 limit_tasks=2)
    order = list(dict.fromkeys(j["task_id"] for j in read_jsonl(jobs)))
    assert [e["task_id"] for e in read_jsonl(tmp_path / "l" / "episodes.jsonl")] == order[:2]
    saved = json.loads((tmp_path / "l" / "run_info.json").read_text())
    assert saved == info
    for k in ("secs_load", "secs_gen", "secs_test", "attempts", "episodes", "gen_tokens", "prompt_tokens",
              "test_python_version", "sandbox_user", "chunks_done", "git_rev", "model_cfg", "backend"):
        assert k in saved
    assert (saved["attempts"], saved["gen_tokens"], saved["prompt_tokens"], saved["chunks_done"]) == (2, 80, 200, 1)
    assert saved["test_python_version"] == ".".join(map(str, sys.version_info[:3])) and saved["sandbox_user"] is None


def test_run_refuses_jobs_of_another_model_and_missing_tasks(tmp_path):
    jobs = tmp_path / "j.jsonl"
    write_jsonl(jobs, A.build_jobs("gemma", "main", ["t0"], ["A"], {"original": 1}, ("neutral",)))
    with pytest.raises(SystemExit):
        A.run("q27", jobs, tmp_path / "o", backend=Fake(), tasks={"t0": TASK}, python=sys.executable, user=None)
    with pytest.raises(SystemExit):
        A.run("gemma", jobs, tmp_path / "o", backend=Fake(), tasks={}, python=sys.executable, user=None)


def test_run_stops_when_the_sandbox_is_broken(tmp_path):
    jobs = tmp_path / "j.jsonl"
    write_jsonl(jobs, A.build_jobs("q27", "main", ["t0"], ["A"], {"original": 1}, ("neutral",)))
    fake = Fake()
    with pytest.raises(SystemExit):
        A.run("q27", jobs, tmp_path / "o", backend=fake, tasks={"t0": TASK}, python="/nonexistent/python3", user=None)
    assert fake.batches == []


# ------------------------------------------------------------------ the vLLM backend, with the fake vllm

@pytest.fixture
def fakevllm(monkeypatch):
    monkeypatch.syspath_prepend(str(PKG / "fakevllm"))
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    import vllm
    vllm.INITS.clear()
    vllm.GENERATES.clear()
    vllm.REJECT_VIDEO = False
    yield vllm
    sys.modules.pop("vllm", None)


def test_importing_a14_needs_no_vllm():
    code = ("import sys; sys.modules['vllm'] = None\n"
            "import persona_affect.a14_desperation.a14 as A\n"
            "assert 'vllm' not in [m for m in sys.modules if sys.modules[m] is not None]\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_vllm_backend_with_fake_vllm(fakevllm):
    fakevllm.REJECT_VIDEO = True                            # the image-only retry, as vigilante's pod_run does
    b = A.VllmBackend("gemma")
    assert [kw["limit_mm_per_prompt"] for kw in fakevllm.INITS] == [{"image": 0, "video": 0}, {"image": 0}]
    kw = fakevllm.INITS[-1]
    assert {k: v for k, v in kw.items() if k != "limit_mm_per_prompt"} == {
        "model": "RedHatAI/gemma-4-31B-it-FP8-dynamic", "revision": "d4ab4f579dd3516f97d8a6a4c98d0653480bad15",
        "dtype": "bfloat16", "max_model_len": 24576, "gpu_memory_utilization": 0.92, "seed": 0,
        "enable_prefix_caching": True}
    import hashlib
    assert {k: b.info[k] for k in ("hf", "rev", "vllm", "template_sha256")} == {
        "hf": "RedHatAI/gemma-4-31B-it-FP8-dynamic", "rev": "d4ab4f579dd3516f97d8a6a4c98d0653480bad15",
        "vllm": "fake", "template_sha256": hashlib.sha256(b"fake").hexdigest()}
    msgs = P.messages_for(TASK, "conflicting", "A", "calm", [])
    text = "\n".join(f"<{m['role']}>{m['content']}" for m in msgs) + "\n<assistant>"
    assert b.n_prompt(msgs) == len(text)
    out = b.generate([(msgs, 3, 700), (msgs, 4, 700), (msgs, 5, 700), (msgs, 6, 10)])
    assert out[0][0].startswith("Here is my solution.") and "def add(*a):" in out[0][0] and out[0][1] == "stop"
    assert out[1][0].startswith("```python\ndef check(candidate):") and out[2][0].startswith("The tests contradict")
    assert out[0][2] == len(out[0][0]) and out[3][1:] == ("length", 10)   # cut at max_tokens
    params = fakevllm.GENERATES[-1]["params"]
    assert [p.kw for p in params][0] == {**A.SAMPLING, "max_tokens": 700, "seed": 3, "skip_special_tokens": True}
    assert fakevllm.GENERATES[-1]["prompts"][0] == {"prompt_token_ids": [ord(c) for c in text]}


def test_run_with_the_fake_vllm_end_to_end(tmp_path, fakevllm):
    ids = ["t0", "t1"]
    jobs = tmp_path / "j.jsonl"
    write_jsonl(jobs, A.build_jobs("q27", "pilot", ids, ["A", "B"], {"conflicting": 3, "original": 1}, ("neutral",)))
    tasks = {t: dict(TASK, task_id=t) for t in ids}
    info = A.run("q27", jobs, tmp_path / "o", tasks=tasks, python=sys.executable, user=None)
    eps = read_jsonl(tmp_path / "o" / "episodes.jsonl")
    assert len(eps) == 16 == info["episodes"] and info["backend"]["hf"] == "Qwen/Qwen3.6-27B"
    assert fakevllm.INITS[0]["gpu_memory_utilization"] == 0.88
    statuses = {a["status"] for e in eps for a in e["attempts"]}
    assert statuses <= {"pass", "fail", "no_code"} and "pass" in statuses
    override = [e for e in eps if e["end"] == "pass" and "def check" in e["attempts"][-1]["code"]]
    assert override                                         # the override reply passes the impossible test
    for e in eps:
        for a in e["attempts"]:
            want = {0: "fail", 1: "pass", 2: "no_code"}[a["seed"] % 3]
            assert a["status"] == want and (a["error_shown"] is None) == (want == "pass")


# ------------------------------------------------------------------ the CLI

META = {"order": [f"t{i}" for i in range(8)],
        "tasks": {f"t{i}": {"clean": i != 3, "oneoff_ok": True, "n_tok": {"q27": 1000 * i, "gemma": 1000 * i + 1}}
                  for i in range(8)},                   # max n_tok 1, 1001, 2001, 3001 (not clean), 4001, ...
        "pilot": ["t1", "t2"]}


@pytest.mark.parametrize("args,n,arms,splits,wordings", [
    (["--phase", "pilot"], 16, {"neutral"}, {"conflicting", "original"}, {"A", "B"}),
    (["--phase", "oneoff"], 12, {"neutral"}, {"oneoff"}, {"A", "B"}),
    (["--phase", "main", "--wording", "B", "--tasks", "T61"], 2 * 24, set(P.ARMS), {"conflicting", "original"}, {"B"}),
    (["--phase", "main", "--wording", "A", "--tasks", "T78", "--primary", "oneoff"], 3 * 24, set(P.ARMS),
     {"oneoff", "original"}, {"A"}),
])
def test_jobs_cli_presets(tmp_path, monkeypatch, args, n, arms, splits, wordings):
    monkeypatch.setattr(A, "load_meta", lambda: META)
    out = tmp_path / "jobs" / "x.jsonl"
    assert A.main(["jobs", "--model", "q27", *args, "--out", str(out)]) == 0
    jobs = read_jsonl(out)
    assert len(jobs) == n and {j["arm"] for j in jobs} == arms and {j["split"] for j in jobs} == splits
    assert {j["wording"] for j in jobs} == wordings
    want = {"pilot": {"t1", "t2"}, "oneoff": {"t1", "t2"}}.get(args[1]) or (
        {"t0", "t1", "t2"} if "T78" in args else {"t0", "t1"})       # max n_tok <= 3000 / 2000, clean only
    assert {j["task_id"] for j in jobs} == want
    samples = {(j["split"], j["sample"]) for j in jobs}
    assert max(s for sp, s in samples if sp != "original") == 2 and all(s == 0 for sp, s in samples if sp == "original")


def test_jobs_cli_remaining_from_and_refusals(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "load_meta", lambda: META)
    full = tmp_path / "jobs" / "full.jsonl"
    A.main(["jobs", "--model", "gemma", "--phase", "pilot", "--out", str(full)])
    jobs = read_jsonl(full)
    for d, part in (("d1", jobs[:3]), ("d2", jobs[3:5])):
        (tmp_path / d).mkdir()
        write_jsonl(tmp_path / d / "episodes.jsonl", [dict(j, attempts=[], end="max_attempts") for j in part])
    rest = tmp_path / "jobs" / "rest.jsonl"
    A.main(["jobs", "--model", "gemma", "--phase", "pilot", "--remaining-from", str(tmp_path / "d1"),
            str(tmp_path / "d2"), "--out", str(rest)])
    assert read_jsonl(rest) == jobs[5:]
    for bad in (["--phase", "main", "--tasks", "T78"], ["--phase", "main", "--wording", "A"],
                ["--phase", "pilot", "--wording", "A"], ["--phase", "pilot", "--tasks", "T61"],
                ["--phase", "oneoff", "--primary", "oneoff"]):
        with pytest.raises(SystemExit):
            A.main(["jobs", "--model", "q27", *bad, "--out", str(tmp_path / "jobs" / "bad.jsonl")])
    for under in ("results", "logs", "prep", "reference"):
        with pytest.raises(SystemExit):
            A.main(["jobs", "--model", "q27", "--phase", "pilot", "--out", str(tmp_path / under / "j.jsonl")])
        assert not (tmp_path / under).exists()


def test_run_cli(tmp_path, monkeypatch, term_handlers):
    seen = {}

    def fake_run(model, jobs, out, **kw):
        seen.update(model=model, jobs=jobs, out=out, **kw)
        return {"episodes": 0}
    monkeypatch.setattr(A, "run", fake_run)
    A.main(["run", "--model", "gemma", "--jobs", "j.jsonl", "--out", "o", "--shard", "1/3", "--limit-tasks", "2",
            "--sandbox-user", "", "--test-python", "/x/py"])
    assert seen == {"model": "gemma", "jobs": Path("j.jsonl"), "out": Path("o"), "python": "/x/py", "user": None,
                    "shard": "1/3", "limit_tasks": 2, "workers": None}
    A.main(["run", "--model", "q27", "--jobs", "j.jsonl", "--out", "o"])
    assert (seen["user"], seen["python"], seen["shard"], seen["limit_tasks"]) == ("a14sandbox", "/usr/bin/python3",
                                                                                   None, None)
    assert len(term_handlers) == 2                          # each main() installs the TERM/HUP handlers first
    for bad in ("2/2", "x", "1/0", "-1/2"):
        with pytest.raises(SystemExit):
            A.main(["run", "--model", "q27", "--jobs", "j.jsonl", "--out", "o", "--shard", bad])


# ------------------------------------------------------------------ the scripts

def test_scripts_parse():
    r = subprocess.run(["bash", "-n", str(PKG / "pod.sh"), str(PKG / "launch.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_launch_sh_shape():
    """launch.sh is read, never run (it would create a pod)."""
    s = (PKG / "launch.sh").read_text()
    assert "persona_affect/common/launch_gen.sh @ 055d69a" in s.split("set -euo pipefail")[0]   # ruling 10 header
    assert "--ship persona_affect --ship vigilante/pod --ship infra" in s
    assert "--results-dir \"$OUT\"" in s and "OUT=persona_affect/a14_desperation/results/$DIR" in s
    assert 'env M="$M" JOBS="$JOBS" DIR="$DIR" SHARD="${SHARD:-}" LIMIT_TASKS="${LIMIT_TASKS:-}" ' \
           "bash persona_affect/a14_desperation/pod.sh" in s
    assert "export RUNPOD_ANY_CUDA=1" in s and 'NAME=${NAME:-pa-a14-$M-$DIR}' in s and "CLOUD:-SECURE" in s
    assert 'MINI=A14' in s and 'check_cap("A14", 18.0' in s
    assert 'bash persona_affect/common/book_pod.sh "$LOG" "$MINI" "$NAME" "$MAXMIN" "$MAXCOST"' in s   # common's booking
    assert "uv run python persona_affect/common/launch_wrap.py --name" in s and "infra/launch_run.py" not in s
    assert "(trap '' INT TERM HUP; exec tee \"$LOG\")" in s
    body = s.split("set -euo pipefail", 1)[1]
    assert body.index("[pre] refusing to launch") < body.index("check_cap(") < body.index("launch_wrap.py --name")
    assert "load_tasks()" in body                                                    # the preflight comes first
    assert '"NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"' in s
    for check in ("*[!0-9]*", "persona_affect/*", "*/results/*|*/logs/*|*/prep/*|*/reference/*", "HF_TOKEN"):
        assert check in s


def _run_pod_sh(env, cwd=ROOT):
    full = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), **env}
    return subprocess.run(["bash", str(Path(cwd) / "persona_affect/a14_desperation/pod.sh")], env=full,
                          capture_output=True, text=True, timeout=120)


@pytest.fixture
def scratch_repo(tmp_path):
    """A copy of the code pod.sh needs (no vigilante/: setup_vllm.sh can't run there), with three synthetic tasks."""
    root = tmp_path / "repo"
    pa = root / "persona_affect"
    (pa / "common").mkdir(parents=True)
    for f in ("__init__.py",):
        shutil.copy(ROOT / "persona_affect" / f, pa / f)
    for f in ("__init__.py", "util.py"):
        shutil.copy(ROOT / "persona_affect" / "common" / f, pa / "common" / f)
    shutil.copytree(PKG, pa / "a14_desperation", ignore=shutil.ignore_patterns("__pycache__", "results", "data", "jobs"))
    data = pa / "a14_desperation" / "data"
    data.mkdir()
    write_jsonl(data / "tasks.jsonl", [dict(TASK, task_id=t) for t in ("t0", "t1", "t2")])
    jobs = pa / "a14_desperation" / "jobs" / "dry.jsonl"
    jobs.parent.mkdir()
    write_jsonl(jobs, A.build_jobs("q27", "pilot", ["t0", "t1", "t2"], ["A", "B"], {"conflicting": 3, "original": 1},
                                   ("neutral",)))
    return root


def test_pod_sh_refusals(scratch_repo):
    out = scratch_repo / "persona_affect/a14_desperation/results/x"
    r = _run_pod_sh({"M": "q27", "JOBS": "persona_affect/a14_desperation/jobs/none.jsonl", "DIR": "x"}, cwd=scratch_repo)
    assert r.returncode == 2 and "JOBS not found" in r.stdout and not (out / "setup_vllm.log").exists()
    for missing in ("M", "JOBS", "DIR"):
        env = {"M": "q27", "JOBS": "persona_affect/a14_desperation/jobs/dry.jsonl", "DIR": "x", "SKIP_SETUP": "1"}
        del env[missing]
        r = _run_pod_sh(env, cwd=scratch_repo)
        assert r.returncode != 0 and missing in r.stderr


def test_pod_sh_dry_run_in_a_scratch_copy(scratch_repo):
    """The Task 6 dry run on a copy of the code: SKIP_SETUP, the fake vllm, no sandbox user, two of three tasks."""
    root = scratch_repo
    pa = root / "persona_affect"
    env = {"SKIP_SETUP": "1", "PY": sys.executable, "TEST_PY": sys.executable, "SANDBOX_USER": "",
           "PYTHONPATH": str(pa / "a14_desperation" / "fakevllm"), "M": "q27", "DIR": "dryrun", "LIMIT_TASKS": "2",
           "JOBS": "persona_affect/a14_desperation/jobs/dry.jsonl"}
    r = _run_pod_sh(env, cwd=root)
    out = pa / "a14_desperation" / "results" / "dryrun"
    assert r.returncode == 0, r.stdout + r.stderr
    eps = read_jsonl(out / "episodes.jsonl")
    assert len(eps) == 16 and (out / "attempts.jsonl").exists() and (out / "run_info.json").exists()
    assert any(e["end"] == "pass" and "def check" in e["attempts"][-1]["code"] for e in eps)
    assert "[a14]" in (out / "run.log").read_text()
    r = _run_pod_sh(dict(env, SKIP_SETUP=""), cwd=root)     # on a pod an empty SANDBOX_USER is refused before setup
    assert r.returncode == 2 and "SANDBOX_USER" in r.stdout and not (out / "setup_vllm.log").exists()
