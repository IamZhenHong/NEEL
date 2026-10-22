"""PA-V8 jobs, keys, frozen config and the ledger helper (plan persona_affect/plans/V8.md, Task 2).

MODELS  short name -> {hf, rev, fmt, group, series, size_b}: the ten models at pinned revisions (HF checked 10-22).
        series "qwen35" (q08 q2 q4 q9 q27), "gemma" (ge2 ge4 g12 g31), "anchor" (q36, the source of mini 0's +23);
        size_b is the size curve's x-axis.
GROUPS  pod order: small (48 GB pod, up to 12B), big (A100 80GB: 27B and 31B).
FROZEN  v8_scale/frozen.json, written at the freeze by `v8 calib-report` (Task 4): {"rungs": {m: {f: int}},
        "on_curve": {m: {f: bool}}, "servable": {m: bool}, "labels": {f: name}, "calib": {...}}. A model that is not
        servable (Change 5) has no rungs. Pods take calibration only from this file, never from results/.

build_jobs(stage, model, frozen=None) lists every generation job of one model in one stage, in run order:
  calib    the none line at each of the 5 rungs: 16 items x 2 families x 5 rungs x 4 samples = 640 decisions, and
           knowledge of the 4 label candidates x 16 = 64. Reads no frozen config: calibration comes before the freeze.
  main     at each family's calibrated base: 16 x 2 x 5 lines x 8 = 1,280 decisions; say 2 x 5 lines x 4 domains x 8 =
           320; knowledge of the 2 frozen labels x 16 = 32; and in a family whose rung is not 1, the uncalibrated rung-1
           secondary, 640 decisions (at most 2,912 jobs; a rung of 1 adds nothing, so no key is duplicated).
  rewrite  the tidiness placebo (line "placebo2", Change 2's one gate rewrite) at the calibrated bases:
           16 x 2 x 8 = 256 decisions.
main and rewrite read FROZEN when `frozen` is None (FileNotFoundError without the file) and refuse a model that is not
servable. The label line is label_line(frozen["labels"][f]), so a knowledge-gated backup label reaches the prompts.

Job fields: key, seed, stage, model, fmt, kind (decide | say | know), family, base (the rung), line, unit (item id,
domain, or "{name}~{rot}" for knowledge), sample, perm (decisions: the PERM6 order of the options, else None), prio,
max_tokens (400 / 80 / 16), tools (True for decisions only: the with_tools flag of the pod's gen), messages; decisions
add amap {option id: role}, knowledge jobs add mapping {letter: value}. A knowledge job has no system prompt: its base
is None and its line "label" (the line whose character it checks); sample s asks with the values rotated by s % 4.
prio: 0 primary decisions, 1 say, 2 knowledge, 3 rung-1 secondary. The list is in prio order, each prio shuffled by
random.Random(f"PA-V8|{stage}|{model}|{prio}"), so a run's first chunks mix every cell.

Keys are PA-V8|{stage}|{model}|{kind}|{family}|r{base}|{line}|{unit}|{sample}. The seed is util.seed of the key with
the line set to "*" (and the stage to "main" for rewrite jobs), so the lines of one item x sample share seed and option
order (common random numbers) and the tidiness rewrite reuses the main run's draws.

  uv run python -m persona_affect.v8_scale.v8 jobs --stage calib --model q9
  uv run python -m persona_affect.v8_scale.v8 ledger --log persona_affect/v8_scale/results/calib_small.launch.log \
      --name v8-calib-small --maxcost 0.90
  uv run python -m persona_affect.v8_scale.v8 render-check --models q08 q4 q36 ge2 g12 g31   (pod.render_check, Task 3)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from persona_affect.common import util
from persona_affect.v8_scale import items as I

_MODELS = (   # short, HF id, revision, fmt, pod group, series, size_b
    ("q08", "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", "qwen", "small", "qwen35", 0.8),
    ("q2", "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", "qwen", "small", "qwen35", 2),
    ("q4", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "qwen", "small", "qwen35", 4),
    ("q9", "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "qwen", "small", "qwen35", 9),
    ("q27", "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654", "qwen", "big", "qwen35", 27),
    ("q36", "Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9", "qwen", "big", "anchor", 27),
    ("ge2", "google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7", "gemma4", "small", "gemma", 2),
    ("ge4", "google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2", "gemma4", "small", "gemma", 4),
    ("g12", "google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", "gemma4", "small", "gemma", 12),
    ("g31", "google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475", "gemma4", "big", "gemma", 31),
)
MODELS = {m: dict(zip(("hf", "rev", "fmt", "group", "series", "size_b"), row)) for m, *row in _MODELS}
GROUPS = {"small": ["q9", "q4", "q2", "q08", "g12", "ge4", "ge2"], "big": ["q27", "g31", "q36"]}   # pod order
assert {m: g for g, ms in GROUPS.items() for m in ms} == {m: d["group"] for m, d in MODELS.items()}
FROZEN = Path(__file__).with_name("frozen.json")   # read at call time (tests monkeypatch it)

STAGES = ("calib", "main", "rewrite")
N_CALIB, N_MAIN, N_SAY, N_KNOW = 4, 8, 8, 16   # samples per item x rung, item x line, domain x line, label
MAX_TOKENS = {"decide": 400, "say": 80, "know": 16}


def _frozen(stage: str, model: str, frozen: dict | None) -> tuple[dict, dict]:
    """(rungs {f: int}, labels {f: name}) of one model from the frozen config; FROZEN is read when `frozen` is None."""
    if frozen is None:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if not frozen["servable"][model]:
        raise ValueError(f"{model} is not servable in the frozen config (Change 5): it has no {stage} jobs")
    rungs, labels = frozen["rungs"][model], frozen["labels"]
    for f in I.FAMILIES:
        if type(rungs[f]) is not int or not 0 <= rungs[f] < len(I.LADDER[f]):
            raise ValueError(f"frozen rung of {model} {f} must be an int in 0-{len(I.LADDER[f]) - 1}, got {rungs[f]!r}")
        if labels[f] not in I.LABEL_CANDIDATES[f]:
            raise ValueError(f"frozen {f} label must be one of {I.LABEL_CANDIDATES[f]}, got {labels[f]!r}")
    return rungs, labels


def build_jobs(stage: str, model: str, frozen: dict | None = None) -> list[dict]:
    """Every job of one model (short name) in one stage, in prio order; see the module docstring."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}: one of {list(MODELS)}")
    fmt = MODELS[model]["fmt"]
    by_prio = defaultdict(list)

    def add(prio, kind, f, base, line, unit, sample, messages, perm=None, **extra):
        seed_key = f"PA-V8|{'main' if stage == 'rewrite' else stage}|{model}|{kind}|{f}|r{base}|*|{unit}|{sample}"
        by_prio[prio].append(dict(
            key=f"PA-V8|{stage}|{model}|{kind}|{f}|r{base}|{line}|{unit}|{sample}", seed=util.seed(seed_key),
            stage=stage, model=model, fmt=fmt, kind=kind, family=f, base=base, line=line, unit=unit, sample=sample,
            perm=perm, prio=prio, max_tokens=MAX_TOKENS[kind], tools=kind == "decide", messages=messages, **extra))

    def decide(prio, f, base, lines, n):   # lines: [(name, text)]; one option order per item x sample, every line
        for line, text in lines:
            slot = I.persona_slot(text, I.LADDER[f][base])
            for t, tid in enumerate(I.items(f)):
                amap = {oid: role for oid, role, _ in I.options(f, tid)}
                for s in range(n):
                    perm = I.PERM6[I.perm_index(t, s)]
                    add(prio, "decide", f, base, line, tid, s, I.episode(f, tid, slot, perm), perm, amap=dict(amap))

    def know(f, name):
        for s in range(N_KNOW):
            messages, mapping = I.know_messages(name, s)
            add(2, "know", f, None, "label", f"{name}~{s % 4}", s, messages, mapping=mapping)

    if stage == "calib":
        for f in I.FAMILIES:
            for base in range(len(I.LADDER[f])):
                decide(0, f, base, [("none", I.LINES[f]["none"])], N_CALIB)
            for name in I.LABEL_CANDIDATES[f]:
                know(f, name)
    else:
        rungs, labels = _frozen(stage, model, frozen)
        for f in I.FAMILIES:
            base = rungs[f]
            if stage == "rewrite":
                decide(0, f, base, [("placebo2", I.PLACEBO2)], N_MAIN)
                continue
            lines = [(x, I.label_line(labels[f]) if x == "label" else I.LINES[f][x]) for x in I.LINE_ORDER]
            decide(0, f, base, lines, N_MAIN)
            for line, text in lines:
                slot = I.persona_slot(text, I.LADDER[f][base])
                for dom in I.domains(f):
                    for s in range(N_SAY):
                        add(1, "say", f, base, line, dom, s, I.say_messages(f, dom, slot))
            know(f, labels[f])
            if base != 1:
                decide(3, f, 1, lines, N_MAIN)

    jobs = []
    for prio in sorted(by_prio):
        group = by_prio[prio]
        random.Random(f"PA-V8|{stage}|{model}|{prio}").shuffle(group)
        jobs += group
    assert len({j["key"] for j in jobs}) == len(jobs), f"duplicate keys in {stage} {model}"
    return jobs


def prompt_hash(jobs: list[dict]) -> str:
    """16 hex of sha256 over key + seed + messages of every job, in order (as persona/history/e_label.py)."""
    h = hashlib.sha256()
    for j in jobs:
        h.update((j["key"] + str(j["seed"]) + json.dumps(j["messages"], sort_keys=True, ensure_ascii=False)).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- ledger

COST = re.compile(r"elapsed cost ~\$([0-9.]+)")
POD = "[runpod] pod "


def ledger_from_log(text: str, name: str, maxcost: float) -> float | None:
    """Book one pod's GPU cost to the ledger (mini "V8") from its launch log, as common/launch_gen.sh does.

    The cost is the last `elapsed cost ~$X` in the log ("[runpod] terminated ...; elapsed cost ~$X" comes after the
    "[pod] job exit code ..." one). A pod that was created ("[runpod] pod ...") but left no cost line, because the
    launcher died, is booked at `maxcost`, an upper bound. Without a pod line no pod was created (no stock, a refusal):
    nothing is booked. Returns the amount booked, or None.
    """
    costs = COST.findall(text)
    if costs:
        usd, note = float(costs[-1]), name
    elif POD in text:
        usd, note = float(maxcost), f"{name} (upper bound, no cost line)"
    else:
        print(f"[ledger] no pod was created for {name} (no cost line and no pod line): nothing booked")
        return None
    util.ledger_add("V8", "gpu", usd, note=note)
    print(f"[ledger] V8 gpu ${usd:.3f} ({note})")
    return usd


# ---------------------------------------------------------------- command line

def _cmd_jobs(a: argparse.Namespace) -> None:
    try:
        jobs = build_jobs(a.stage, a.model)
    except FileNotFoundError:
        raise SystemExit(f"no frozen config at {FROZEN}: {a.stage} jobs are built only after the freeze "
                         "(v8 calib-report writes it)") from None
    m = MODELS[a.model]
    print(f"{a.model} = {m['hf']} @ {m['rev'][:12]} ({m['fmt']}, {m['group']} pod): {a.stage}, {len(jobs)} jobs, "
          f"prompt hash {prompt_hash(jobs)}")
    n = Counter((j["prio"], j["kind"], j["family"], f"r{j['base']}", j["line"]) for j in jobs)
    print(f"{'prio':<6}{'kind':<8}{'family':<14}{'base':<7}{'line':<10}{'n':>5}")
    for (p, k, f, b, x), c in sorted(n.items(), key=lambda kv: tuple(map(str, kv[0]))):
        print(f"{p:<6}{k:<8}{f:<14}{b:<7}{x:<10}{c:>5}")
    slots = {}   # the system prompt's first paragraph of each primary cell: the line and base the model sees
    for j in jobs:
        if j["kind"] == "decide" and j["prio"] == 0:
            slots.setdefault((j["family"], j["base"], j["line"]), j["messages"][0]["content"].split("## Deployment")[0])
    for (f, b, x), slot in sorted(slots.items(), key=lambda kv: tuple(map(str, kv[0]))):
        print(f"  {f} r{b} {x}: {slot.strip()!r}")


def _cmd_ledger(a: argparse.Namespace) -> None:
    ledger_from_log(a.log.read_text(encoding="utf-8", errors="replace"), a.name, a.maxcost)


def _cmd_render_check(a: argparse.Namespace) -> None:
    from persona_affect.v8_scale import pod   # here, so that `jobs` and `ledger` never import the runner
    pod.render_check(a.models)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.v8_scale.v8",
                                 description="PA-V8: value lines, labels and how-to-act lines across sizes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs", help="one model's jobs in a stage: counts per cell, prompt hash, the slot texts")
    j.add_argument("--stage", required=True, choices=STAGES)
    j.add_argument("--model", required=True, choices=list(MODELS))
    j.set_defaults(fn=_cmd_jobs)
    lg = sub.add_parser("ledger", help="book a pod's GPU cost (mini V8) from its launch log")
    lg.add_argument("--log", required=True, type=Path, help="the launcher output tee'd by launch.sh")
    lg.add_argument("--name", required=True, help="the pod name: the ledger note")
    lg.add_argument("--maxcost", required=True, type=float, help="the pod's MAXCOST, booked when it left no cost line")
    lg.set_defaults(fn=_cmd_ledger)
    rc = sub.add_parser("render-check", help="print one decide, say and know prompt per model with its own tokenizer "
                                             "(local, tokenizer files only); fail on a decide prompt of >= 1,500 "
                                             "tokens or one without its case file")
    rc.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS), metavar="MODEL",
                    help=f"short names (default: all ten): {' '.join(MODELS)}")
    rc.set_defaults(fn=_cmd_render_check)
    # Tasks 4-5 register calib-report and analyze the same way (set_defaults(fn=...))
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
