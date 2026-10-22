"""Tests for persona_affect.common.vig / pod_gen and the two pod scripts (pod_gen.sh, launch_gen.sh).

Run from the repo root: uv run pytest persona_affect/tests/test_common_vig.py -q
Nothing here launches a pod or touches the network. register() writes into vigilante's module-level dicts, so the
`registry` fixture puts them back after each test. The shell tests run the real scripts inside a throwaway repo root,
with a fake `uv` (standing in for infra/launch_run.py and the ledger call), a fake python and a fake setup_vllm.sh.
"""
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
from vigilante.src.sim import Case

from persona_affect.common import pod_gen, vig

REPO = Path(__file__).resolve().parents[2]
COMMON = REPO / "persona_affect" / "common"
SCRIPTS = [COMMON / "pod_gen.sh", COMMON / "launch_gen.sh"]


@pytest.fixture
def registry():
    """Put vigilante's case and persona registries back after the test (register() mutates them in place)."""
    saved = [(d, dict(d)) for d in (vcases.ALL, personas.FAMOUS)]
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
    vig.register({"pa_old": mk_case("pa_old")}, {"pa_taken": "You are Taken."})
    with pytest.raises(ValueError):                                      # good new case, bad cond
        vig.register({"pa_new": mk_case("pa_new")}, {"pa_taken": "You are Other."})
    assert "pa_new" not in vcases.ALL
    with pytest.raises(ValueError):                                      # bad case, good new cond
        vig.register({"pa_old": mk_case("pa_old", task="x")}, {"pa_new_cond": "You are New."})
    with pytest.raises(KeyError):
        personas.condition("pa_new_cond")


def test_register_rejects_malformed_input(registry):
    with pytest.raises(ValueError, match="wrong_key"):
        vig.register({"wrong_key": mk_case("pa_test")})                  # the key must be the case's id
    with pytest.raises(ValueError):
        vig.register(conds={"pa_bad": None})                             # a persona text is a string
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
        "register({'pa_fake': Case(id='pa_fake', family='CHAT', title='t', org='O', context='c', permissions='',"
        " task='hi')}, {'pa_fake_cond': 'You are Fake.'})\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = {}

    def fake_main():
        seen.update(argv=list(sys.argv), case="pa_fake" in vcases.ALL, cond=personas.condition("pa_fake_cond"))

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
    """A throwaway repo root holding copies of the two scripts: they `cd` to their own ../.. and write relative paths."""
    r = tmp_path / "repo"
    (r / "persona_affect" / "common").mkdir(parents=True)
    for s in SCRIPTS:
        shutil.copy2(s, r / "persona_affect" / "common" / s.name)
    return r


# env vars the scripts read: dropped from the developer's shell so a stray MODEL or SKIP_SETUP cannot leak in
SCRIPT_VARS = ("MODEL", "NAME", "JOBS", "OUT", "REGISTER", "FMT", "MML", "GPUS", "MAXMIN", "MAXCOST", "CONTAINER_GB",
               "CLOUD", "MINI", "SKIP_SETUP", "PY", "RUNPOD_ANY_CUDA", "HF_TOKEN")


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
# stands in for `uv run python infra/launch_run.py ...` (@LAUNCH@ is what the launcher does) and for the ledger call
# (recorded); any other uv call, the HF token lookup, prints nothing
case "$*" in
  *infra/launch_run.py*)
    { printf 'RUNPOD_ANY_CUDA=%s\\n' "$RUNPOD_ANY_CUDA"; printf '%s\\n' "$@"; } > "@ROOT@/launch_args.txt"
    @LAUNCH@ ;;
  *ledger_add*)
    printf '%s\\n' "$@" > "@ROOT@/ledger_args.txt" ;;
esac
exit 0
"""


def fake_uv(root, tmp_path, launch):
    _fake(tmp_path / "bin" / "uv", FAKE_UV.replace("@LAUNCH@", launch).replace("@ROOT@", str(root)))


POD_LINE = "[runpod] pod ab12cd (NVIDIA A100-SXM4-80GB) at $1.79/hr; time cap 40 min; budget $1.20; boot self-destruct 70 min"
DONE_LINE = "[pod] job exit code 0; elapsed cost ~$0.512"
TERM_LINE = "[runpod] terminated ab12cd; elapsed cost ~$0.534"


def run_launch(root, tmp_path, launcher_log, *, launcher_rc=0, jobs="persona_affect/jobs.jsonl", make_jobs=True, **over):
    if make_jobs and not jobs.startswith("/"):
        (root / jobs).parent.mkdir(parents=True, exist_ok=True)
        (root / jobs).write_text('{"key": "k"}\n')
    (root / "fake_launch.log").write_text(launcher_log)
    fake_uv(root, tmp_path, f'cat "{root}/fake_launch.log"; exit {launcher_rc}')
    env = _env(PATH=f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}", MINI="T1", NAME="pa-t", JOBS=jobs, OUT="out/x",
               REGISTER="a.b c.d", MODEL="M/m", MAXMIN="40")
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
    assert a[:3] == ["run", "python", "infra/launch_run.py"]
    val = lambda flag: a[a.index(flag) + 1]
    assert val("--name") == "pa-t" and val("--max-minutes") == "40" and val("--max-cost") == "1.20"   # 40 min x $1.80/h
    assert val("--container-gb") == "120" and val("--cloud") == "COMMUNITY" and val("--results-dir") == "out/x"
    assert "--allow-concurrent" in a
    g = a.index("--gpus")
    assert a[g + 1:g + 3] == ["NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"] and a[g + 3].startswith("--")
    assert [a[i + 1] for i, x in enumerate(a) if x == "--ship"] == [
        "persona_affect", "vigilante/src", "vigilante/cases", "vigilante/pod", "vigilante/__init__.py", "infra"]
    assert a[a.index("--") + 1:] == ["env", "MODEL=M/m", "JOBS=persona_affect/jobs.jsonl", "OUT=out/x", "REGISTER=a.b c.d",
                                     "FMT=qwen", "MML=40960", "bash", "persona_affect/common/pod_gen.sh"]


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


def test_launch_gen_books_the_upper_bound_when_a_pod_ran_but_no_cost_line_came(root, tmp_path):
    r = run_launch(root, tmp_path, POD_LINE + "\n[pod] ssh root@1.2.3.4 -p 22\n", launcher_rc=1)
    assert r.returncode == 1
    assert _ledger_call(root)[-3:] == ["T1", "1.20", "pa-t (upper bound, no cost line)"]


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


@pytest.mark.parametrize("sig, booked", [
    (signal.SIGINT, ["T1", "0.100", "pa-t"]),                              # Ctrl-C: the launcher cleans up, prints its cost
    (signal.SIGTERM, ["T1", "1.20", "pa-t (upper bound, no cost line)"]),  # killed outright: only the pod line exists
])
def test_launch_gen_still_books_after_a_signal(root, tmp_path, sig, booked):
    (root / "persona_affect" / "jobs.jsonl").write_text('{"key": "k"}\n')
    (root / "fake_launcher.py").write_text(FAKE_LAUNCHER_PY)
    fake_uv(root, tmp_path, f"exec {sys.executable} {root}/fake_launcher.py")
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
    assert _ledger_call(root)[-3:] == booked, out


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
