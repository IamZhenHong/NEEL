"""Tests for persona_affect.common.vig / pod_gen and the two pod scripts (pod_gen.sh, launch_gen.sh).

Run from the repo root: uv run pytest persona_affect/tests/test_common_vig.py -q
Nothing here launches a pod or touches the network. register() writes into vigilante's module-level dicts, so the
`registry` fixture puts them back after each test. The shell tests run the real scripts inside a throwaway repo root,
with a fake `uv` (standing in for the launcher, the local job check, the cap check, the HF token lookup and the ledger
call), a fake python, a fake setup_vllm.sh and, for the signal tests, a fake infra/launch_run.py run by the real
launch_wrap.py.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import vigilante.cases as vcases
from vigilante.src import episode, personas, pod_run
from vigilante.src.sim import Case, tool

from persona_affect.common import pod_gen, vig

REPO = Path(__file__).resolve().parents[2]
COMMON = REPO / "persona_affect" / "common"
SCRIPTS = [COMMON / "pod_gen.sh", COMMON / "launch_gen.sh", COMMON / "book_pod.sh"]
WRAP = COMMON / "launch_wrap.py"


@pytest.fixture
def registry():
    """Put vigilante's case and persona registries back after the test (register() mutates them in place). The ids
    these tests use all start with pa_test: any left there by an earlier test file are hidden for the test, then restored."""
    saved = [(d, dict(d)) for d in (vcases.ALL, personas.FAMOUS)]
    for d, _ in saved:
        for k in [k for k in d if k.startswith("pa_test")]:
            del d[k]
    yield
    for d, old in saved:
        d.clear()
        d.update(old)


def mk_case(cid="pa_test", **kw):
    base = dict(id=cid, family="CHAT", title="t", org="Org", context="You help.", permissions="", task="Say hi.")
    return Case(**{**base, **kw})


# ---------------------------------------------------------------- register

def test_registered_case_and_cond_reach_init_episode(registry):
    case = mk_case()
    vig.register({"pa_test": case}, {"pa_test_cond": "You are Testy."})
    assert vcases.ALL["pa_test"] is case
    ep = episode.init_episode({"key": "k", "case": "pa_test", "cond": "pa_test_cond", "seed": 1})
    assert ep["messages"][0]["content"].startswith("You are Testy.")


def test_registered_case_runs_through_run_batch(registry):
    case = mk_case(tools=[tool("lookup", "Look something up.", q=("string", "Query"))],
                   responders={"lookup": lambda args, state: f"RESULT for {args['q']}"})
    vig.register({"pa_test": case}, {"pa_test_cond": "You are Testy."})
    replies = iter(["Checking.\n<tool_call>\n<function=lookup>\n<parameter=q>\nhello\n</parameter>\n</function>\n"
                    "</tool_call>", "All done."])
    seen = []

    def generate(items):                                                 # stands in for vLLM: one reply per live episode
        seen.extend(it["messages"][0]["content"] for it in items)
        return [(next(replies), "stop") for _ in items]

    job = {"key": "k", "case": "pa_test", "cond": "pa_test_cond", "seed": 1, "fmt": "qwen"}
    (ep,) = episode.run_batch([job], generate, log=lambda m: None)
    assert ep["end"] == "reply" and [c["name"] for c in ep["calls"]] == ["lookup"]
    assert [m["content"] for m in ep["messages"] if m["role"] == "tool"] == ["RESULT for hello"]
    assert all(t.startswith("You are Testy.") for t in seen)
    assert [t["finish"] for t in ep["turns"]] == ["stop", "stop"]


def test_a_cond_with_other_text_raises(registry):
    vig.register(conds={"pa_test_cond": "You are Testy."})
    with pytest.raises(ValueError, match="pa_test_cond"):
        vig.register(conds={"pa_test_cond": "You are somebody else."})
    assert personas.condition("pa_test_cond") == ("You are Testy.", None)      # the first text stays


def test_registering_the_same_thing_again_is_fine(registry):
    case = mk_case()
    vig.register({"pa_test": case}, {"pa_test_cond": "You are Testy."})
    vig.register({"pa_test": case}, {"pa_test_cond": "You are Testy."})                      # the same objects
    vig.register({"pa_test": mk_case()}, {"pa_test_cond": "You are " + "Testy."})             # equal ones, rebuilt
    assert vcases.ALL["pa_test"] is case
    assert personas.condition("pa_test_cond") == ("You are Testy.", None)


def test_a_different_case_under_a_taken_id_raises_and_changes_nothing(registry):
    case = mk_case()
    vig.register({"pa_test": case})
    with pytest.raises(ValueError, match="pa_test"):
        vig.register({"pa_test": mk_case(task="Say bye.")})
    assert vcases.ALL["pa_test"] is case
    builtin = next(iter(vcases.ALL))                                    # a case that ships with vigilante
    before = vcases.ALL[builtin]
    with pytest.raises(ValueError, match=re.escape(builtin)):
        vig.register({builtin: mk_case(builtin)})
    assert vcases.ALL[builtin] is before


def test_cond_ids_that_vigilante_already_knows_are_taken(registry):
    # condition() looks in PERSONAS (and the other dicts) before FAMOUS, so a FAMOUS entry under such an id would be
    # silently ignored: refuse it
    with pytest.raises(ValueError, match="P1"):
        vig.register(conds={"P1": "You are Robin."})
    with pytest.raises(ValueError, match="N_riley"):
        vig.register(conds={"N_riley": "You are Riley."})                # an agent-name condition, not a text
    vig.register(conds={"P1": personas.PERSONAS["P1"]})                  # the very same text: nothing to add
    assert "P1" not in personas.FAMOUS and personas.condition("P1") == (personas.PERSONAS["P1"], None)


def test_a_refused_call_registers_nothing(registry):
    vig.register({"pa_test_old": mk_case("pa_test_old")}, {"pa_test_taken": "You are Taken."})
    with pytest.raises(ValueError):                                      # good new case, bad cond
        vig.register({"pa_test_new": mk_case("pa_test_new")}, {"pa_test_taken": "You are Other."})
    assert "pa_test_new" not in vcases.ALL
    with pytest.raises(ValueError):                                      # bad case, good new cond
        vig.register({"pa_test_old": mk_case("pa_test_old", task="x")}, {"pa_test_new_cond": "You are New."})
    with pytest.raises(KeyError):
        personas.condition("pa_test_new_cond")


def test_register_rejects_malformed_input(registry):
    with pytest.raises(ValueError, match="wrong_key"):
        vig.register({"wrong_key": mk_case("pa_test")})                  # the key must be the case's id
    with pytest.raises(ValueError):
        vig.register(conds={"pa_test_bad": None})                             # a persona text is a string
    with pytest.raises(ValueError):
        vig.register(conds={"": "You are nobody."})
    with pytest.raises(ValueError):
        vig.register([mk_case()])                                        # a dict {case_id: Case}, not a list
    assert vig.register() is None and vig.register({}, {}) is None       # nothing to do is fine


# ---------------------------------------------------------------- pod_gen

def test_pod_gen_imports_the_modules_then_hands_the_rest_to_pod_run(tmp_path, monkeypatch, registry):
    (tmp_path / "pa_fake_mini_cases.py").write_text(
        "from persona_affect.common.vig import register\n"
        "from vigilante.src.sim import Case\n"
        "register({'pa_test_fake': Case(id='pa_test_fake', family='CHAT', title='t', org='O', context='c', permissions='',"
        " task='hi')}, {'pa_test_fake_cond': 'You are Fake.'})\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = {}

    def fake_main():
        seen.update(argv=list(sys.argv), case="pa_test_fake" in vcases.ALL, cond=personas.condition("pa_test_fake_cond"))

    monkeypatch.setattr(pod_run, "main", fake_main)
    monkeypatch.setattr(sys, "argv", ["pod_gen"])
    args = ["--jobs", "j.jsonl", "--out", "o", "--model", "m", "--fmt", "gemma4", "--max-model-len", "1234"]
    try:
        pod_gen.main(["--register", "pa_fake_mini_cases", "--", *args])
    finally:
        sys.modules.pop("pa_fake_mini_cases", None)
    assert seen["argv"][1:] == args                                      # pod_run sees only what follows the dashes
    assert seen["case"] and seen["cond"] == ("You are Fake.", None)      # registered before pod_run.main() ran


def test_pod_gen_reads_sys_argv_by_default_and_takes_several_modules(tmp_path, monkeypatch):
    for name in ("pa_fake_one", "pa_fake_two"):
        (tmp_path / f"{name}.py").write_text("IMPORTED = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    calls = []
    monkeypatch.setattr(pod_run, "main", lambda: calls.append(list(sys.argv[1:])))
    monkeypatch.setattr(sys, "argv", ["pod_gen", "--register", "pa_fake_one", "pa_fake_two", "--", "--jobs", "j"])
    try:
        pod_gen.main()
        assert "pa_fake_one" in sys.modules and "pa_fake_two" in sys.modules
    finally:
        sys.modules.pop("pa_fake_one", None)
        sys.modules.pop("pa_fake_two", None)
    assert calls == [["--jobs", "j"]]


def test_pod_gen_with_no_modules_just_runs_pod_run(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["pod_gen"])                        # pod_gen.main() replaces it: restore afterwards
    monkeypatch.setattr(pod_run, "main", lambda: calls.append(list(sys.argv[1:])))
    pod_gen.main(["--register", "--", "--jobs", "j", "--out", "o"])
    assert calls == [["--jobs", "j", "--out", "o"]]


def test_pod_gen_stops_before_pod_run_on_a_bad_module_or_a_missing_separator(monkeypatch):
    calls = []
    monkeypatch.setattr(pod_run, "main", lambda: calls.append(1))
    with pytest.raises(ModuleNotFoundError):
        pod_gen.main(["--register", "pa_no_such_module_anywhere", "--", "--jobs", "j"])
    with pytest.raises(SystemExit) as e:
        pod_gen.main(["--register", "--jobs", "j"])                      # no `--`
    assert e.value.code == 2
    assert calls == []


def test_pod_gen_runs_as_a_module_and_pod_run_parses_its_args():
    # end to end without vLLM: --help makes pod_run's argparse print its usage and exit before anything heavy loads
    r = subprocess.run([sys.executable, "-m", "persona_affect.common.pod_gen", "--register", "--", "--help"],
                       cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "--jobs" in r.stdout and "--max-model-len" in r.stdout


# ---------------------------------------------------------------- pod_gen --check (final review I4)

CHECK_MOD = ("from persona_affect.common.vig import register\n"
             "from vigilante.src.sim import Case\n"
             "register({'pa_test_chk': Case(id='pa_test_chk', family='CHAT', title='t', org='O', context='c', "
             "permissions='', task='hi')}, {'pa_test_chk_cond': 'You are Checked.'})\n")


@pytest.fixture
def check_env(tmp_path, monkeypatch, registry):
    """A module that registers case pa_test_chk and cond pa_test_chk_cond, one that registers nothing, and a pod_run
    that must never run."""
    (tmp_path / "pa_fake_chk.py").write_text(CHECK_MOD)
    (tmp_path / "pa_fake_typo.py").write_text("X = 1   # imports fine, registers nothing\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    def no_pod_run():
        raise AssertionError("--check must not start pod_run")
    monkeypatch.setattr(pod_run, "main", no_pod_run)
    yield tmp_path
    for m in ("pa_fake_chk", "pa_fake_typo"):
        sys.modules.pop(m, None)


def _jobs_file(tmp_path, jobs, name="jobs.jsonl"):
    path = tmp_path / name
    path.write_text("".join((j if isinstance(j, str) else json.dumps(j)) + "\n" for j in jobs))
    return str(path)


def _cjob(key="a", **kw):
    return {"key": key, "case": "pa_test_chk", "cond": "pa_test_chk_cond", "seed": 1, "fmt": "gemma4", **kw}


def _check(*argv):
    pod_gen.main(["--register", "pa_fake_chk", *argv])


def test_check_passes_jobs_whose_fmt_is_the_pods_and_whose_ids_are_registered(check_env, capsys):
    chat = {"key": "b", "case": None, "cond": "P1", "prompt": "hi", "seed": 2, "fmt": "gemma4"}
    _check("--check", _jobs_file(check_env, [_cjob(), chat]), "--fmt", "gemma4")
    out = capsys.readouterr().out
    assert "check ok" in out and "2 jobs" in out and "gemma4" in out


@pytest.mark.parametrize("job, fmt", [
    (_cjob(fmt="gemma4"), "qwen"),            # a Gemma job on a Qwen pod: its turns would be parsed as Qwen's
    (_cjob(fmt="qwen"), "gemma4"),
    ({k: v for k, v in _cjob().items() if k != "fmt"}, "gemma4"),   # no fmt means qwen
], ids=["gemma-job-qwen-pod", "qwen-job-gemma-pod", "no-fmt-gemma-pod"])
def test_check_refuses_a_job_whose_fmt_is_not_the_pods(check_env, job, fmt):
    path = _jobs_file(check_env, [_cjob("ok", fmt=fmt), {**job, "key": "odd"}])
    with pytest.raises(SystemExit) as e:
        _check("--check", path, "--fmt", fmt)
    assert "odd" in str(e.value) and "fmt" in str(e.value) and "1 of 2" in str(e.value)


@pytest.mark.parametrize("job", [_cjob(case="pa_test_nope"), _cjob(cond="pa_test_nocond"),
                                 {"key": "a", "case": None, "cond": "P1", "seed": 1, "fmt": "gemma4"}],
                         ids=["unknown case", "unknown cond", "chat job without a prompt"])
def test_check_refuses_a_job_that_init_episode_cannot_start(check_env, job):
    with pytest.raises(SystemExit) as e:
        _check("--check", _jobs_file(check_env, [_cjob("fine"), job]), "--fmt", "gemma4")
    msg = str(e.value)
    assert "init_episode" in msg and "'a'" in msg and "fine" not in msg


def test_check_catches_a_register_module_that_registers_nothing(check_env, capsys):
    with pytest.raises(SystemExit) as e:                       # a REGISTER typo: the module imports but adds no case
        pod_gen.main(["--register", "pa_fake_typo", "--check", _jobs_file(check_env, [_cjob()]), "--fmt", "gemma4"])
    assert "pa_test_chk" in str(e.value)
    assert "+0 cases, +0 conds" in capsys.readouterr().out


def test_check_refuses_an_empty_or_unreadable_jobs_file(check_env):
    with pytest.raises(SystemExit, match="no jobs"):
        _check("--check", _jobs_file(check_env, ["", "  "]), "--fmt", "gemma4")
    with pytest.raises(SystemExit, match="line 2"):
        _check("--check", _jobs_file(check_env, [_cjob(), "{not json", "[1, 2]"]), "--fmt", "gemma4")


def test_check_refuses_an_fmt_the_engine_does_not_know(check_env):
    with pytest.raises(SystemExit, match="gemma"):
        _check("--check", _jobs_file(check_env, [_cjob(fmt="gemma")]), "--fmt", "gemma")


def test_check_and_the_pod_run_arguments_do_not_mix(check_env):
    path = _jobs_file(check_env, [_cjob()])
    with pytest.raises(SystemExit) as e:
        _check("--check", path, "--fmt", "gemma4", "--", "--jobs", path)
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:                       # pod_run's own --fmt goes after the dashes
        _check("--fmt", "gemma4", "--", "--jobs", path)
    assert e.value.code == 2


def test_check_runs_as_a_module_from_the_repo_root(tmp_path):
    builtin = next(iter(vcases.ALL))                           # a case that ships with vigilante
    good = _jobs_file(tmp_path, [{"key": "g", "case": builtin, "cond": "P1", "seed": 1}], "good.jsonl")
    bad = _jobs_file(tmp_path, [{"key": "g", "case": builtin, "cond": "P1", "seed": 1, "fmt": "gemma4"}], "bad.jsonl")
    run = lambda path: subprocess.run([sys.executable, "-m", "persona_affect.common.pod_gen", "--register",
                                       "--check", path, "--fmt", "qwen"], cwd=REPO, capture_output=True, text=True,
                                      timeout=120)
    ok, no = run(good), run(bad)
    assert ok.returncode == 0 and "check ok" in ok.stdout, ok.stderr
    assert no.returncode != 0 and "'g'" in no.stderr and "gemma4" in no.stderr


# ---------------------------------------------------------------- the scripts

@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_parses(script):
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_is_executable(script):
    assert os.access(script, os.X_OK)


def _fake(path: Path, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)


@pytest.fixture
def root(tmp_path):
    """A throwaway repo root holding copies of the scripts and the launcher wrapper: they `cd` to their own ../.. and
    write relative paths. It has no infra/ (a test that needs a launcher writes a fake one), so nothing can reach the
    real one."""
    r = tmp_path / "repo"
    (r / "persona_affect" / "common").mkdir(parents=True)
    for s in [*SCRIPTS, WRAP]:
        shutil.copy2(s, r / "persona_affect" / "common" / s.name)
    return r


# env vars the scripts read: dropped from the developer's shell so a stray MODEL or SKIP_SETUP cannot leak in
SCRIPT_VARS = ("MODEL", "NAME", "JOBS", "OUT", "REGISTER", "FMT", "MML", "GPUS", "MAXMIN", "MAXCOST", "CONTAINER_GB",
               "CLOUD", "MINI", "SKIP_SETUP", "PY", "RUNPOD_ANY_CUDA", "HF_TOKEN", "CAP")


def _env(**over):
    env = {k: v for k, v in os.environ.items() if k not in SCRIPT_VARS}
    env.update(over)
    return env


def _calls(root):
    f = root / "calls.txt"
    return f.read_text().splitlines() if f.exists() else []


def run_pod_gen(root, *, setup_rc=0, py_rc=0, jobs=True, **over):
    if jobs:
        (root / "persona_affect" / "jobs.jsonl").write_text('{"key": "k"}\n')
    _fake(root / "vigilante" / "pod" / "setup_vllm.sh", f'echo setup >> "{root}/calls.txt"\necho setup-output\nexit {setup_rc}\n')
    _fake(root / "fakepy", f'#!/bin/sh\necho "py $*" >> "{root}/calls.txt"\necho python-output\nexit {py_rc}\n')
    env = _env(PY=str(root / "fakepy"), JOBS="persona_affect/jobs.jsonl", OUT="out/x", MODEL="M/m", REGISTER="a.b c.d",
               FMT="gemma4", MML="9999")
    env.update(over)
    return subprocess.run(["bash", str(root / "persona_affect" / "common" / "pod_gen.sh")], env=env, capture_output=True,
                          text=True, timeout=60)


PY_LINE = ("py -m persona_affect.common.pod_gen --register a.b c.d -- --jobs persona_affect/jobs.jsonl --out out/x "
           "--model M/m --fmt gemma4 --max-model-len 9999")


def test_pod_gen_sh_sets_up_vllm_then_runs_pod_gen_and_logs(root):
    r = run_pod_gen(root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _calls(root) == ["setup", PY_LINE]
    assert "setup-output" in (root / "out" / "x" / "setup_vllm.log").read_text()
    log = (root / "out" / "x" / "run.log").read_text()
    assert "python-output" in log and "pod done (exit 0)" in log


def test_pod_gen_sh_skip_setup(root):
    r = run_pod_gen(root, SKIP_SETUP="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _calls(root) == [PY_LINE]


def test_pod_gen_sh_exits_with_the_runs_code(root):
    r = run_pod_gen(root, py_rc=3)
    assert r.returncode == 3
    assert "pod done (exit 3)" in (root / "out" / "x" / "run.log").read_text()


def test_pod_gen_sh_stops_when_setup_fails(root):
    r = run_pod_gen(root, setup_rc=1)
    assert r.returncode == 1
    assert _calls(root) == ["setup"] and "setup FAILED" in r.stdout


def test_pod_gen_sh_checks_jobs_before_the_slow_setup(root):
    r = run_pod_gen(root, jobs=False)
    assert r.returncode == 2 and _calls(root) == []
    assert "persona_affect/jobs.jsonl" in r.stdout


def test_pod_gen_sh_defaults_match_pod_run(root):
    r = run_pod_gen(root, SKIP_SETUP="1", MODEL="", FMT="", MML="", REGISTER="")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _calls(root) == ["py -m persona_affect.common.pod_gen --register -- --jobs persona_affect/jobs.jsonl --out out/x "
                            "--model Qwen/Qwen3.6-27B --fmt qwen --max-model-len 40960"]


FAKE_UV = """#!/bin/sh
# stands in for uv: `uv run python persona_affect/common/launch_wrap.py ...` (@LAUNCH@ is what the launcher does), the
# local job check, the cap check and the ledger call (each recorded, each exiting with its FAKE_*_RC); the HF token
# lookup prints $FAKE_HF_TOKEN; any other uv call prints nothing
case "$*" in
  *launch_wrap.py*)
    { printf 'RUNPOD_ANY_CUDA=%s\\n' "$RUNPOD_ANY_CUDA"; printf '%s\\n' "$@"; } > "@ROOT@/launch_args.txt"
    printf 'HF_TOKEN=%s\\n' "${HF_TOKEN:-}" > "@ROOT@/launch_env.txt"
    @LAUNCH@ ;;
  *pod_gen*--check*)
    printf '%s\\n' "$@" > "@ROOT@/check_args.txt"; exit ${FAKE_CHECK_RC:-0} ;;
  *check_cap*)
    printf '%s\\n' "$@" > "@ROOT@/cap_args.txt"; exit ${FAKE_CAP_RC:-0} ;;
  *ledger_add*)
    printf '%s\\n' "$@" > "@ROOT@/ledger_args.txt"; exit ${FAKE_LEDGER_RC:-0} ;;
  *HF_TOKEN*)
    printf '%s\\n' "${FAKE_HF_TOKEN:-}" ;;
esac
exit 0
"""


def fake_uv(root, tmp_path, launch):
    _fake(tmp_path / "bin" / "uv", FAKE_UV.replace("@LAUNCH@", launch).replace("@ROOT@", str(root)))


POD_LINE = "[runpod] pod ab12cd (NVIDIA A100-SXM4-80GB) at $1.79/hr; time cap 40 min; budget $1.20; boot self-destruct 70 min"
DONE_LINE = "[pod] job exit code 0; elapsed cost ~$0.512"
TERM_LINE = "[runpod] terminated ab12cd; elapsed cost ~$0.534"
REGISTER = "persona_affect.a.cases vigilante.cases"
KILL_CMD = ("uv run python -c \"import sys; sys.path.insert(0, 'infra'); from runpod_guard import terminate; "
            "terminate('ab12cd')\"")
UPPER = "pa-t (upper bound, pod not confirmed terminated)"


def run_launch(root, tmp_path, launcher_log, *, launcher_rc=0, jobs="persona_affect/jobs.jsonl", make_jobs=True, **over):
    if make_jobs and not jobs.startswith("/"):
        (root / jobs).parent.mkdir(parents=True, exist_ok=True)
        (root / jobs).write_text('{"key": "k"}\n')
    (root / "fake_launch.log").write_text(launcher_log)
    fake_uv(root, tmp_path, f'cat "{root}/fake_launch.log"; exit {launcher_rc}')
    env = _env(PATH=f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}", MINI="T1", NAME="pa-t", JOBS=jobs, OUT="out/x",
               REGISTER=REGISTER, MODEL="M/m", MAXMIN="40")
    env.update(over)
    env = {k: v for k, v in env.items() if v is not None}
    return subprocess.run(["bash", str(root / "persona_affect" / "common" / "launch_gen.sh")], env=env,
                          capture_output=True, text=True, timeout=60)


def _ledger_call(root):
    f = root / "ledger_args.txt"
    return f.read_text().splitlines() if f.exists() else None


def test_launch_gen_runs_the_guarded_launcher_with_the_agreed_arguments(root, tmp_path):
    r = run_launch(root, tmp_path, "\n".join([POD_LINE, DONE_LINE, TERM_LINE]) + "\n")
    assert r.returncode == 0, r.stdout + r.stderr
    lines = (root / "launch_args.txt").read_text().splitlines()
    assert lines[0] == "RUNPOD_ANY_CUDA=1"
    a = lines[1:]
    assert a[:3] == ["run", "python", "persona_affect/common/launch_wrap.py"]     # infra/launch_run.py, through the wrapper
    val = lambda flag: a[a.index(flag) + 1]
    assert val("--name") == "pa-t" and val("--max-minutes") == "40" and val("--max-cost") == "1.20"   # 40 min x $1.80/h
    assert val("--container-gb") == "120" and val("--cloud") == "COMMUNITY" and val("--results-dir") == "out/x"
    assert "--allow-concurrent" in a
    g = a.index("--gpus")
    assert a[g + 1:g + 3] == ["NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"] and a[g + 3].startswith("--")
    assert [a[i + 1] for i, x in enumerate(a) if x == "--ship"] == [
        "persona_affect", "vigilante/src", "vigilante/cases", "vigilante/pod", "vigilante/__init__.py", "infra"]
    assert a[a.index("--") + 1:] == ["env", "MODEL=M/m", "JOBS=persona_affect/jobs.jsonl", "OUT=out/x",
                                     f"REGISTER={REGISTER}", "FMT=qwen", "MML=40960", "bash",
                                     "persona_affect/common/pod_gen.sh"]


def test_launch_gen_tees_the_launcher_output_and_books_the_last_cost_line(root, tmp_path):
    log = "\n".join([POD_LINE, DONE_LINE, TERM_LINE]) + "\n"
    r = run_launch(root, tmp_path, log)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (root / "out" / "x.launch.log").read_text() == log
    assert log in r.stdout                                               # still shown on the terminal
    assert _ledger_call(root)[-3:] == ["T1", "0.534", "pa-t"]            # the last match, not the "[pod] job exit" one
    assert _ledger_call(root)[:3] == ["run", "python", "-c"] and "ledger_add" in _ledger_call(root)[3]


def test_launch_gen_books_the_cost_and_passes_on_the_exit_code_when_the_launcher_fails(root, tmp_path):
    r = run_launch(root, tmp_path, "\n".join([POD_LINE, "Traceback ...", TERM_LINE]) + "\n", launcher_rc=7)
    assert r.returncode == 7
    assert _ledger_call(root)[-3:] == ["T1", "0.534", "pa-t"]


def test_launch_gen_books_the_upper_bound_and_warns_when_a_pod_ran_but_no_cost_line_came(root, tmp_path):
    r = run_launch(root, tmp_path, POD_LINE + "\n[pod] ssh root@1.2.3.4 -p 22\n", launcher_rc=1)
    assert r.returncode == 1
    assert _ledger_call(root)[-3:] == ["T1", "2.10", UPPER]          # MAXCOST 1.20 x (40 + 30) / 40, not 1.20
    assert "WARNING" in r.stderr and "ab12cd" in r.stderr and KILL_CMD in r.stderr


def test_launch_gen_never_books_the_job_exit_cost_as_final(root, tmp_path):
    # the launcher prints "elapsed cost" at the job's exit, then pulls the results and terminates the pod; killed in
    # between, it leaves a pod running that the job-exit cost would under-book
    r = run_launch(root, tmp_path, "\n".join([POD_LINE, DONE_LINE, "[local] pull error: timed out"]) + "\n", launcher_rc=1)
    assert _ledger_call(root)[-3:] == ["T1", "2.10", UPPER]
    assert "WARNING" in r.stderr and KILL_CMD in r.stderr


def test_launch_gen_books_nothing_when_no_pod_was_ever_created(root, tmp_path):
    # retry.sh reruns a launcher that found no stock; those rounds cost nothing and must not reach the ledger
    r = run_launch(root, tmp_path, "[runpod] round 1/5: no stock on COMMUNITY\nno GPU stock after all rounds; nothing "
                   "launched, nothing charged\n", launcher_rc=1)
    assert r.returncode == 1
    assert _ledger_call(root) is None
    assert "nothing booked" in r.stdout


def test_launch_gen_gpus_are_pipe_separated(root, tmp_path):
    run_launch(root, tmp_path, TERM_LINE + "\n", GPUS="NVIDIA A40|NVIDIA L40S", CONTAINER_GB="150", CLOUD="SECURE",
               FMT="gemma4", MML="12288", MAXMIN="90")
    a = (root / "launch_args.txt").read_text().splitlines()[1:]
    g = a.index("--gpus")
    assert a[g + 1:g + 3] == ["NVIDIA A40", "NVIDIA L40S"]
    assert a[a.index("--max-cost") + 1] == "2.70"
    assert a[a.index("--container-gb") + 1] == "150" and a[a.index("--cloud") + 1] == "SECURE"
    assert "FMT=gemma4" in a and "MML=12288" in a


@pytest.mark.parametrize("over, why", [
    (dict(MINI=None), "MINI"),
    (dict(JOBS=None), "JOBS"),
    (dict(OUT=None), "OUT"),
    (dict(MAXMIN="soon"), "MAXMIN"),
    (dict(jobs="persona_affect/a/results/jobs.jsonl"), "results"),             # the launcher's sync drops results/
    (dict(jobs="persona_affect/a/logs/jobs.jsonl"), "logs"),
    (dict(jobs="vigilante/exp/jobs.jsonl"), "persona_affect"),                 # not under the one shipped data dir
    (dict(jobs="/tmp/jobs.jsonl"), "persona_affect"),
])
def test_launch_gen_refuses_before_launching_anything(root, tmp_path, over, why):
    r = run_launch(root, tmp_path, TERM_LINE + "\n", **over)
    assert r.returncode != 0
    assert not (root / "launch_args.txt").exists() and _ledger_call(root) is None
    assert why in r.stdout + r.stderr


def test_launch_gen_refuses_a_jobs_file_that_does_not_exist(root, tmp_path):
    r = run_launch(root, tmp_path, TERM_LINE + "\n", jobs="persona_affect/missing.jsonl", make_jobs=False)
    assert r.returncode == 2 and "missing.jsonl" in r.stdout + r.stderr
    assert not (root / "launch_args.txt").exists()


FAKE_LAUNCHER_PY = """import time
print("[runpod] pod ab12cd (NVIDIA A100-SXM4-80GB) at $1.79/hr; time cap 40 min", flush=True)
try:
    time.sleep(120)
finally:    # what PodSession.__exit__ does on Ctrl-C: terminating the pod takes a moment, then the cost line
    time.sleep(0.5)
    print("[runpod] terminated ab12cd; elapsed cost ~$0.100", flush=True)
"""


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP], ids=lambda s: s.name)
def test_launch_gen_still_books_after_a_signal(root, tmp_path, sig):
    # Ctrl-C, a harness stop (TERM to the process group) and a closed terminal (HUP) all reach the launcher as Ctrl-C:
    # it terminates the pod and prints the terminated line, tee keeps writing the log, and that cost is booked
    (root / "persona_affect" / "jobs.jsonl").write_text('{"key": "k"}\n')
    (root / "infra").mkdir()
    (root / "infra" / "launch_run.py").write_text(FAKE_LAUNCHER_PY)      # run by the real launch_wrap.py
    fake_uv(root, tmp_path, f'shift 2; exec {sys.executable} "$@"')
    env = _env(PATH=f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}", MINI="T1", NAME="pa-t",
               JOBS="persona_affect/jobs.jsonl", OUT="out/x", MAXMIN="40")
    # own process group (killpg is what Ctrl-C does), and SIGINT at its default whatever the test runner inherited
    p = subprocess.Popen(["bash", str(root / "persona_affect" / "common" / "launch_gen.sh")], env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
                         preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    log = root / "out" / "x.launch.log"
    try:
        deadline = time.time() + 30
        while not (log.exists() and "[runpod] pod ab12cd" in log.read_text()):
            assert time.time() < deadline and p.poll() is None, "the fake launcher never started"
            time.sleep(0.05)
        os.killpg(p.pid, sig)
        out, _ = p.communicate(timeout=60)
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    assert p.returncode == 128 + sig, out
    assert _ledger_call(root)[-3:] == ["T1", "0.100", "pa-t"], out
    assert "[runpod] terminated ab12cd" in (root / "out" / "x.launch.log").read_text()


# ---------------------------------------------------------------- launch_gen.sh preflight (final review I4, M5)

@pytest.mark.parametrize("over, why", [
    (dict(MODEL="google/gemma-4-31B-it", FMT="qwen", FAKE_HF_TOKEN="hf_x"), "FMT=gemma4"),   # Gemma 4 on the Qwen format
    (dict(MODEL="Google/Gemma-4-E4B-it", FAKE_HF_TOKEN="hf_x"), "FMT=gemma4"),
    (dict(FAKE_CHECK_RC="1"), "pod_gen --check"),                   # the local check refused the jobs (fmt, ids, modules)
    (dict(MODEL="google/gemma-4-31B-it", FMT="gemma4"), "HF_TOKEN"),            # gated, and no token in .env
    (dict(MODEL="meta-llama/Llama-3.1-8B-Instruct"), "HF_TOKEN"),
    (dict(REGISTER="persona_affect.a.cases mymod.cases"), "mymod.cases"),       # not shipped: fails on the pod after setup
    (dict(REGISTER="vigilante.exp.r21.cases"), "vigilante.exp.r21.cases"),
    (dict(OUT="/abs/out/x"), "OUT"),                                # the launcher would pull /root/NEEL/abs/... in vain
    (dict(OUT="persona_affect/../out/x"), "OUT"),
    (dict(OUT="../out/x"), "OUT"),
    (dict(MAXMIN="0"), "MAXMIN"),
    (dict(MAXCOST="lots"), "MAXCOST"),
    (dict(CAP="5", FAKE_CAP_RC="1"), "CAP"),                        # the ledger plus this pod's upper bound pass the cap
], ids=lambda v: str(v) if isinstance(v, dict) else v)
def test_launch_gen_preflight_refuses_before_launching(root, tmp_path, over, why):
    r = run_launch(root, tmp_path, TERM_LINE + "\n", **over)
    assert r.returncode == 2, r.stdout + r.stderr
    assert not (root / "launch_args.txt").exists() and _ledger_call(root) is None
    assert why in r.stdout + r.stderr


def test_launch_gen_checks_the_jobs_locally_with_the_pods_fmt_and_modules(root, tmp_path):
    r = run_launch(root, tmp_path, TERM_LINE + "\n", MODEL="google/gemma-4-31B-it", FMT="gemma4", FAKE_HF_TOKEN="hf_x")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (root / "check_args.txt").read_text().splitlines() == [
        "run", "python", "-m", "persona_affect.common.pod_gen", "--register", "persona_affect.a.cases", "vigilante.cases",
        "--check", "persona_affect/jobs.jsonl", "--fmt", "gemma4"]
    assert (root / "launch_env.txt").read_text() == "HF_TOKEN=hf_x\n"            # a gated model gets the token
    a = (root / "launch_args.txt").read_text().splitlines()
    assert "FMT=gemma4" in a and "MODEL=google/gemma-4-31B-it" in a


def test_launch_gen_checks_the_cap_with_the_pods_upper_bound_only_when_asked(root, tmp_path):
    r = run_launch(root, tmp_path, TERM_LINE + "\n", CAP="5")
    assert r.returncode == 0, r.stdout + r.stderr
    cap = (root / "cap_args.txt").read_text().splitlines()
    assert "check_cap" in cap[3] and cap[-3:] == ["T1", "5", "2.10"]            # MAXCOST 1.20 x (40 + 30) / 40
    (root / "cap_args.txt").unlink()
    assert run_launch(root, tmp_path, TERM_LINE + "\n").returncode == 0 and not (root / "cap_args.txt").exists()


def test_launch_gen_header_puts_out_under_results_and_says_how_to_stop_a_run():
    head = "".join(l for l in (COMMON / "launch_gen.sh").read_text().splitlines(True) if l.startswith("#"))
    assert re.search(r"OUT=persona_affect/\S+/results/\S+", head) and "/out/" not in head
    assert "Ctrl-C" in head and "kill -TERM -- -" in head and "PID alone" in head


# ---------------------------------------------------------------- book_pod.sh (final review I3)

NO_STOCK = "[runpod] round 1/5: no stock on COMMUNITY\nno GPU stock after all rounds; nothing launched, nothing charged\n"


def run_book(root, tmp_path, log, args=None, *, cwd=None, **over):
    """book_pod.sh on a launcher log (None: no log file at all), with the fake uv; by default LOG MINI NAME MAXMIN MAXCOST
    are out/pod.launch.log T1 pa-t 40 1.20."""
    (root / "out").mkdir(exist_ok=True)
    if log is not None:
        (root / "out" / "pod.launch.log").write_text(log)
    fake_uv(root, tmp_path, "exit 0")
    env = _env(PATH=f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}", **over)
    args = ["out/pod.launch.log", "T1", "pa-t", "40", "1.20"] if args is None else args
    return subprocess.run(["bash", str(root / "persona_affect" / "common" / "book_pod.sh"), *args], cwd=cwd or root,
                          env=env, capture_output=True, text=True, timeout=60)


def test_book_pod_books_the_terminated_line_as_final(root, tmp_path):
    r = run_book(root, tmp_path, "\n".join([POD_LINE, DONE_LINE, TERM_LINE]) + "\n")
    assert r.returncode == 0, r.stdout + r.stderr
    call = _ledger_call(root)
    assert call[:3] == ["run", "python", "-c"] and "ledger_add" in call[3] and call[-3:] == ["T1", "0.534", "pa-t"]
    assert "[ledger] T1 gpu $0.534 (pa-t)" in r.stdout and "WARNING" not in r.stdout + r.stderr


@pytest.mark.parametrize("log", [
    POD_LINE + "\n",                                                         # the launcher died before any cost line
    "\n".join([POD_LINE, DONE_LINE]) + "\n",                                 # killed between the job's exit and terminate
    "\n".join([POD_LINE, "[runpod] terminated zz99; elapsed cost ~$0.200"]) + "\n",   # another pod's line is not ours
], ids=["pod line only", "job-exit cost only", "another pod terminated"])
def test_book_pod_books_the_upper_bound_and_says_how_to_kill_a_pod_it_cannot_confirm(root, tmp_path, log):
    r = run_book(root, tmp_path, log)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _ledger_call(root)[-3:] == ["T1", "2.10", UPPER]
    assert "WARNING" in r.stderr and "ab12cd" in r.stderr and KILL_CMD in r.stderr and "--kill-ours" in r.stderr


def test_book_pod_books_nothing_when_no_pod_was_created(root, tmp_path):
    # retry.sh reruns a launcher that found no stock; those rounds cost nothing and must not reach the ledger
    r = run_book(root, tmp_path, NO_STOCK)
    assert r.returncode == 0 and _ledger_call(root) is None and "nothing booked" in r.stdout


def test_book_pod_books_a_terminated_line_even_without_a_pod_line(root, tmp_path):
    r = run_book(root, tmp_path, TERM_LINE + "\n")
    assert r.returncode == 0 and _ledger_call(root)[-3:] == ["T1", "0.534", "pa-t"]


def test_book_pod_reads_the_log_relative_to_where_it_is_called_from(root, tmp_path):
    r = run_book(root, tmp_path, TERM_LINE + "\n", ["pod.launch.log", "T1", "pa-t", "40", "1.20"], cwd=root / "out")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _ledger_call(root)[-3:] == ["T1", "0.534", "pa-t"]


@pytest.mark.parametrize("maxmin, maxcost, bound", [("40", "1.20", "2.10"), ("90", "2.70", "3.60"), ("040", "1.20", "2.10"),
                                                    ("7", "1.00", "5.29"), ("60", "0", "0.00")])
def test_book_pod_upper_bound_is_maxcost_times_maxmin_plus_30_over_maxmin_rounded_up(root, tmp_path, maxmin, maxcost,
                                                                                   bound):
    r = run_book(root, tmp_path, None, ["--bound", maxmin, maxcost])
    assert r.returncode == 0 and r.stdout == bound + "\n", r.stderr


@pytest.mark.parametrize("args", [[], ["out/pod.launch.log", "T1"], ["out/pod.launch.log", "T1", "pa-t", "0", "1.20"],
                                  ["out/pod.launch.log", "T1", "pa-t", "soon", "1.20"],
                                  ["out/pod.launch.log", "T1", "pa-t", "40", "lots"],
                                  ["out/pod.launch.log", "", "pa-t", "40", "1.20"], ["--bound", "0", "1.20"],
                                  ["--bound", "40"]], ids=str)
def test_book_pod_refuses_bad_arguments(root, tmp_path, args):
    r = run_book(root, tmp_path, TERM_LINE + "\n", args)
    assert r.returncode == 2 and _ledger_call(root) is None, r.stdout + r.stderr


def test_book_pod_without_a_log_warns_and_books_nothing(root, tmp_path):
    r = run_book(root, tmp_path, None)
    assert r.returncode == 1 and _ledger_call(root) is None and "WARNING" in r.stderr


def test_book_pod_names_the_amount_when_the_booking_fails(root, tmp_path):
    r = run_book(root, tmp_path, TERM_LINE + "\n", FAKE_LEDGER_RC="1")
    assert r.returncode == 1 and "FAILED to book $0.534" in r.stderr


# ---------------------------------------------------------------- launch_wrap.py (final review I3)

WRAP_LAUNCHER = """import sys, time
import helper_next_to_me             # what `python infra/launch_run.py` can import: its own directory comes first
print("argv", sys.argv[1:], flush=True)
if sys.argv[1:] == ["--sleep"]:
    print("started", flush=True)
    try:
        time.sleep(60)
    finally:                          # the launcher's cleanup (terminate the pod): a second signal arrives in here
        time.sleep(0.5)
        print("cleaned up", flush=True)
sys.exit(int(sys.argv[-1]) if sys.argv[-1].isdigit() else 0)
"""


@pytest.fixture
def wrap_root(tmp_path):
    r = tmp_path / "wroot"
    (r / "persona_affect" / "common").mkdir(parents=True)
    shutil.copy2(WRAP, r / "persona_affect" / "common" / WRAP.name)
    (r / "infra").mkdir()
    (r / "infra" / "launch_run.py").write_text(WRAP_LAUNCHER)
    (r / "infra" / "helper_next_to_me.py").write_text("")
    return r


def test_launch_wrap_hands_on_the_arguments_and_the_exit_code(wrap_root):
    r = subprocess.run([sys.executable, str(wrap_root / "persona_affect" / "common" / WRAP.name), "--name", "x", "--",
                        "env", "A=1", "7"], cwd=wrap_root / "persona_affect", capture_output=True, text=True, timeout=60)
    assert r.returncode == 7, r.stderr
    assert "argv ['--name', 'x', '--', 'env', 'A=1', '7']" in r.stdout


@pytest.mark.parametrize("sig, rc", [(signal.SIGTERM, 128 + signal.SIGTERM), (signal.SIGHUP, 128 + signal.SIGHUP),
                                     (signal.SIGINT, -signal.SIGINT)], ids=lambda v: getattr(v, "name", str(v)))
def test_launch_wrap_stops_the_launcher_as_ctrl_c_does_and_a_second_signal_cannot_cut_its_cleanup(wrap_root, sig, rc):
    p = subprocess.Popen([sys.executable, str(wrap_root / "persona_affect" / "common" / WRAP.name), "--sleep"],
                         cwd=wrap_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    try:
        assert p.stdout.readline().startswith("argv") and p.stdout.readline().strip() == "started"
        p.send_signal(sig)
        if sig != signal.SIGINT:      # uv run passes a TERM sent to the process group on to its child: it comes twice
            time.sleep(0.1)
            p.send_signal(sig)
        out, err = p.communicate(timeout=30)
    finally:
        if p.poll() is None:
            p.kill()
    assert "cleaned up" in out and p.returncode == rc, err


def test_ship_list_is_enough_for_the_pod_to_import_the_engine(tmp_path):
    ships = re.findall(r"--ship (\S+)", (COMMON / "launch_gen.sh").read_text())
    assert ships
    tree = tmp_path / "pod"
    for p in ships:
        src, dst = REPO / p, tree / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "results", "logs"))
        else:
            shutil.copy2(src, dst)
    code = ("import persona_affect.common.pod_gen, persona_affect.common.vig, vigilante.src.pod_run as p, vigilante, os\n"
            "from vigilante.cases import ALL\n"
            "from vigilante.src.episode import init_episode\n"
            "assert os.path.realpath(vigilante.__file__).startswith(os.path.realpath(os.getcwd())), vigilante.__file__\n"
            "init_episode({'key': 'k', 'case': next(iter(ALL)), 'cond': 'P1', 'seed': 1})\n"
            "print('ok', len(ALL))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=tree, capture_output=True, text=True, timeout=120,
                       env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    assert r.returncode == 0 and r.stdout.startswith("ok"), r.stderr
