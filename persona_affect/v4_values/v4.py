"""PA-V4 (persona_affect/plans/V4.md): does what a model says it values predict which values it holds in agentic scenarios?

Task 3: the job lists and the frontier runner. The cases are in cases.py, which registers them with the vigilante engine when
imported (here, and on a pod through common/pod_gen.py --register persona_affect.v4_values.cases).

  uv run python -m persona_affect.v4_values.v4 jobs --model q27|gemma|gpt|gemini --out FILE [--calibration]
  uv run python -m persona_affect.v4_values.v4 run-frontier --model gpt|gemini --jobs FILE --out DIR [--smoke] \\
      [--resume [--retry-failed]] [--workers N]
  uv run python -m persona_affect.v4_values.v4 stated-dry-run --out FILE
  uv run python -m persona_affect.v4_values.v4 parse-rates --dir DIR

Jobs. Two groups and four kinds, each under every model:
  chat   (stated)    rank      the 10 ranking paraphrases (stated.RANK_PROMPTS; the six values are listed in the order fixed
                               for each paraphrase)
                     pair      15 pairs x 4 paraphrases x both orders (stated.PAIR_PROMPTS filled with the value texts)
  agent  (revealed)  conflict  the 60 two-value conflict scenarios, no outside pressure
                     pressure  the 36 single-value scenarios at the 5 levels none, insist, operator, authority, incentive
Samples per cell: open models 3 / 2 / 4 / 4 (rank / pair / conflict / pressure), frontier models 2 / 1 / 2 / 1. So an open
model gets 30 + 240 chat and 240 + 720 agent jobs = 1230, and a frontier model 20 + 120 and 120 + 180 = 440. Thinking is off.
A frontier model plays 584 user turns (140 chat, 120 conflict, 324 pressure: 36 each at none, operator and incentive, 72 at
authority, 144 at insist), so at least 584 calls, and about twice that when it calls a tool on every turn.

A job is one JSONL line:
  key       PA-V4|{model}|{group}|{item}|{third}|{sample}; the third part is the pressure level, the pair order (AB or BA),
            "conflict" for a conflict scenario, or "fixed" for a ranking (its listing order is fixed per paraphrase)
  exp, model, group, kind, item, sample    the same fields; `item` is rank{p}, {x}-{y}-p{k} (pair x < y, paraphrase k), or
            the scenario id (c_aiid_auto_1, p_hon_1)
  prompt, paraphrase        chat jobs: the text sent as the only user message (no system prompt, no tools) and its index
  pair, order, shown        pair jobs: the pair as [x, y] in stated.PAIRS orientation; AB shows x as (A) and y as (B), BA the
                            reverse, and `shown` is [id shown as A, id shown as B]. The stated strength is the share that
                            prefers x, so an answer "A" prefers x under AB and "B" prefers x under BA.
  pair                      conflict jobs: the scenario's pair [a, b], in the same orientation
  value, level              pressure jobs: the scenario's value id and the pressure level
  case                      agent jobs: the engine case id (cases.py: pa_v4c_{id}, pa_v4p_{id}_{level})
  system_suffix             operator and incentive jobs only: the text appended after the permission block
  calibration               true on calibration jobs only
  cond "P0" (no persona); seed util.seed(key); thinking False; fmt "qwen" for q27 and the frontier models, "gemma4" for gemma
            (it must equal the pod's FMT: the episode parser reads the job's field, the template conversion reads the flag)
The list is shuffled with random.Random(f"PA-V4|{model}"), so stated and scenario jobs, kinds and levels interleave in time.
A job holds only JSON values, so a file read back equals the list built here (run-frontier relies on it).

--calibration (open models only): the 36 pressure scenarios at level none plus all 60 conflicts, 2 samples each = 192 jobs, for
the calibration pods. Their samples are numbered from sample 100 (100 and 101, and the jobs carry "calibration": true), so
their keys and seeds are not the main run's: the main run redraws every conflict, as the plan requires. Shuffled by
f"PA-V4|{model}|cal".

stated-dry-run: 40 stated prompts for a model outside the main run (Qwen/Qwen3.5-9B, model tag q9): the 10 ranking paraphrases
and the 15 pairs in both orders, the paraphrase cycling through the 4 templates. parse-rates reads the pod's
episodes_chat.jsonl and prints how many replies parse (counts only, no answer is shown).

run-frontier plays a jobs file against OpenRouter through the engine's run_batch (common.orbackend.make_or_backend: effort
low, max_tokens 2000, temperature 0.7, 8 workers). The jobs, already shuffled, run in blocks of 96, each block its own run_batch
call; one backend serves all blocks (so one cap), and it books its spend to the ledger as it accrues (several lines under the
model's id, which frontier_cap sums), the rest by close() in `finally`. main() installs util.install_term_handlers() first, so
that a TERM or HUP runs that `finally` too. When a block is done its episode records are appended to
DIR/episodes_{chat,agent}.jsonl (vigilante's episode_record format, the same layout as a pod run), so a run that dies loses at
most the block in flight. DIR/settings.json is written at the end. A call that errors, or that finds the cap reached, comes
back as an empty turn whose `finish` is "error:<what>" or "cost_stop". The runner passes orbackend.end_failed as run_batch's
on_round, so that the episode ends at that failure (no empty assistant message, no scripted push after it); the episode is
kept, listed in settings.json ("failed_keys") and counted.
It refuses a jobs file that is not a subset of the model's frontier grid as the code builds it now (another model's file, a
stale or edited one). A DIR that already holds episodes_*.jsonl is refused, unless --resume: it reads those episodes (dropping a
last line cut short by a kill, even inside a multibyte character), skips their keys and runs the rest. A key whose episode
failed is left alone unless --retry-failed, which moves the failed records to DIR/retried_failed.jsonl and runs those keys
again, so no key is in the episode files twice. --smoke runs 2 jobs (the first chat job and an agent job that has no scripted
followups, the first one that offers a tool without parameters: at most 1 + 3 calls) into DIR/smoke, replacing the last smoke
run, and exits 1 unless usage.cost arrived on every reply and nothing failed. A main run that ends with failed or cost_stop
episodes exits 1 as well, with its output kept.

Budget. Cap $16 for the mini (pods and API together, ledger id "V4"); gpt may use $8 of it and gemini $2. A run's cap is its
model's share less what the ledger already holds under V4 with that model's id as the note, and never more than what is left
of the $16.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from persona_affect.common import orbackend, util
from persona_affect.common.orbackend import make_or_backend
from persona_affect.v4_values import cases
from persona_affect.v4_values.scenarios import CONFLICTS, PRESSURE
from persona_affect.v4_values.stated import PAIR_PROMPTS, PAIRS, RANK_PROMPTS, VALUES, parse_pair, parse_rank
from vigilante.src.episode import dumps, episode_record, run_batch

EXP = "PA-V4"
MINI = "V4"                     # the ledger id
CAP_TOTAL = 16.0                # USD, gpu + api, for the whole mini (raised from the card's $12 before any freeze)
SMOKE_CAP = 0.25                # USD a smoke run may spend
BLOCK = 96                      # jobs per run_batch call; a finished block is appended to the output
WORKERS = 8
EFFORT, MAX_TOKENS, TEMPERATURE = "low", 2000, 0.7     # frontier: reasoning tokens count against max_tokens

MODELS = {   # short name -> (model id, job fmt, frontier?). Frontier backends emit Qwen-format text, so fmt "qwen".
    "q27": ("Qwen/Qwen3.6-27B", "qwen", False),
    "gemma": ("google/gemma-4-31B-it", "gemma4", False),
    "gpt": ("openai/gpt-5.6-sol", "qwen", True),
    "gemini": ("google/gemini-3.8-flash", "qwen", True),
}
FRONTIER_SHARE = {"gpt": 8.0, "gemini": 2.0}      # USD of the $16 each frontier model may use
DRY_RUN = ("q9", "Qwen/Qwen3.5-9B", "qwen")       # the stated-prompt parse-rate dry run: a model outside the main run
SAMPLES = {False: {"rank": 3, "pair": 2, "conflict": 4, "pressure": 4},
           True: {"rank": 2, "pair": 1, "conflict": 2, "pressure": 1}}
CAL_BASE, CAL_SAMPLES = 100, 2        # calibration samples are numbered CAL_BASE, CAL_BASE + 1: never the main run's keys
# the recheck after the one calibration rewrite (10-22): the rewritten values at level none and the rewritten conflicts,
# 4 fresh samples each, numbered after the calibration's
RECHECK_VALUES, RECHECK_CONFLICTS = ("HARM", "AUTO", "AIID"), ("c_auto_priv_1", "c_auto_proc_2")
RECHECK_BASE, RECHECK_SAMPLES = 102, 4
ORDERS = ("AB", "BA")


def _model(model: str) -> tuple[str, str, bool]:
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {sorted(MODELS)}")
    return MODELS[model]


# ---------------------------------------------------------------- jobs

def _job(model: str, fmt: str, group: str, kind: str, item: str, third: str, sample: int, **fields) -> dict:
    key = f"{EXP}|{model}|{group}|{item}|{third}|{sample}"
    return {"key": key, "exp": EXP, "model": model, "group": group, "kind": kind, "item": item, "sample": sample,
            **fields, "cond": cases.COND, "seed": util.seed(key), "thinking": False, "fmt": fmt}


def _rank_jobs(model: str, fmt: str, samples) -> list[dict]:
    return [_job(model, fmt, "chat", "rank", f"rank{p}", "fixed", s, paraphrase=p, prompt=RANK_PROMPTS[p])
            for p in range(len(RANK_PROMPTS)) for s in samples]


def _pair_job(model: str, fmt: str, pair: tuple[str, str], paraphrase: int, order: str, sample: int) -> dict:
    x, y = pair
    a, b = (x, y) if order == "AB" else (y, x)
    return _job(model, fmt, "chat", "pair", f"{x}-{y}-p{paraphrase}", order, sample, pair=[x, y], paraphrase=paraphrase,
                order=order, shown=[a, b], prompt=PAIR_PROMPTS[paraphrase].format(a=VALUES[a], b=VALUES[b]))


def _pair_jobs(model: str, fmt: str, samples) -> list[dict]:
    return [_pair_job(model, fmt, pair, k, order, s) for pair in PAIRS for k in range(len(PAIR_PROMPTS))
            for order in ORDERS for s in samples]


def _conflict_jobs(model: str, fmt: str, samples, **extra) -> list[dict]:
    return [_job(model, fmt, "agent", "conflict", sc["id"], "conflict", s, pair=list(sc["pair"]),
                 case=cases.conflict_case_id(sc["id"]), **extra) for sc in CONFLICTS for s in samples]


def _pressure_jobs(model: str, fmt: str, samples, levels, **extra) -> list[dict]:
    jobs = []
    for sc in PRESSURE:
        for level in levels:
            suffix = cases.suffix_for(sc, level)
            for s in samples:
                jobs.append(_job(model, fmt, "agent", "pressure", sc["id"], level, s, value=sc["value"], level=level,
                                 case=cases.pressure_case_id(sc["id"], level), **({"system_suffix": suffix} if suffix else {}),
                                 **extra))
    return jobs


def grid_jobs(model: str, frontier: bool) -> list[dict]:
    """Every job for `model`, unshuffled, in construction order: rank, pair, conflict, pressure (the scenarios in bank order,
    the levels in cases.LEVELS order, the samples innermost)."""
    _, fmt, is_frontier = _model(model)
    if frontier != is_frontier:
        raise ValueError(f"{model!r} is {'a frontier' if is_frontier else 'an open'} model but frontier={frontier}")
    n = SAMPLES[frontier]
    return [*_rank_jobs(model, fmt, range(n["rank"])), *_pair_jobs(model, fmt, range(n["pair"])),
            *_conflict_jobs(model, fmt, range(n["conflict"])),
            *_pressure_jobs(model, fmt, range(n["pressure"]), cases.LEVELS)]


def build_jobs(model: str, frontier: bool, *, calibration: bool = False, recheck: bool = False) -> list[dict]:
    """The shuffled job list for `model` (short name: q27, gemma, gpt, gemini). `frontier` must match the model.

    calibration=True (open models only) is the pressure scenarios at level none plus all conflicts, CAL_SAMPLES samples each,
    numbered from CAL_BASE so that no key or seed is the main run's; it is shuffled by its own seed. recheck=True (with
    calibration) is the recheck after the rewrite: RECHECK_VALUES' pressure scenarios at none and RECHECK_CONFLICTS,
    RECHECK_SAMPLES samples each from RECHECK_BASE."""
    _, fmt, is_frontier = _model(model)
    if frontier != is_frontier:
        raise ValueError(f"{model!r} is {'a frontier' if is_frontier else 'an open'} model but frontier={frontier}")
    if calibration:
        if frontier:
            raise ValueError("calibration is a pod run on an open model, not a frontier one")
        samples = range(CAL_BASE, CAL_BASE + CAL_SAMPLES)
        jobs = [*_conflict_jobs(model, fmt, samples, calibration=True),
                *_pressure_jobs(model, fmt, samples, ("none",), calibration=True)]
        if recheck:
            samples = range(RECHECK_BASE, RECHECK_BASE + RECHECK_SAMPLES)
            jobs = [j for j in [*_conflict_jobs(model, fmt, samples, calibration=True),
                                *_pressure_jobs(model, fmt, samples, ("none",), calibration=True)]
                    if j["item"] in RECHECK_CONFLICTS or j.get("value") in RECHECK_VALUES]
        random.Random(f"{EXP}|{model}|cal" + ("|recheck" if recheck else "")).shuffle(jobs)
        return jobs
    jobs = grid_jobs(model, frontier)
    random.Random(f"{EXP}|{model}").shuffle(jobs)
    return jobs


def stated_dry_run_jobs() -> list[dict]:
    """The 40 stated prompts of the parse-rate dry run: the 10 ranking paraphrases, then every pair in both orders (30),
    the paraphrase of pair i being i % 4 so that all four templates and both orders are exercised. Not shuffled."""
    short, _, fmt = DRY_RUN
    jobs = _rank_jobs(short, fmt, [0])
    for i, pair in enumerate(PAIRS):
        for order in ORDERS:
            jobs.append(_pair_job(short, fmt, pair, i % len(PAIR_PROMPTS), order, 0))
    return jobs


def write_jobs(path, jobs: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(j) + "\n" for j in jobs), encoding="utf-8")


def read_jobs(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _offers_a_parameterless_tool(job: dict) -> bool:
    case = cases.CASES.get(job.get("case"))
    return case is not None and any(not t["function"]["parameters"]["properties"] for t in case.tools)


def smoke_jobs(jobs: list[dict]) -> list[dict]:
    """Two jobs for a smoke run: an agent job without scripted followups (the tool path; at most three calls) and the first
    chat job (the plain path; one call). The agent job is the first whose case offers a tool without parameters
    (`{"type": "object", "properties": {}}`: some providers refuse an empty object schema, and a smoke run is where to find
    out), else the first agent job without followups. A list without one of the groups is filled up with its first jobs."""
    quiet = [j for j in jobs if j["group"] == "agent" and j.get("level") not in ("insist", "authority")]
    agent = next((j for j in quiet if _offers_a_parameterless_tool(j)), None) or next(iter(quiet), None)
    chat = next((j for j in jobs if j["group"] == "chat"), None)
    picked = [j for j in (agent, chat) if j is not None]
    picked += [j for j in jobs if j not in picked][: 2 - len(picked)]
    return picked


# ---------------------------------------------------------------- budget

def _ledger_spent(mini: str, note: str) -> float:
    """USD in the ledger under `mini` whose note is exactly `note` (the OpenRouter backend books a model's spend with the
    model id as the note). util has a total per mini only; like it, this refuses a line it cannot read."""
    path = util.LEDGER                     # read at call time: tests point it at a tmp file
    if not path.exists():
        return 0.0
    amounts = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row["mini"] == mini and row.get("note") == note:
                usd = float(row["usd"])
                if not math.isfinite(usd):
                    raise ValueError("usd is not finite")
                amounts.append(usd)
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"{path}:{n}: unreadable ledger line ({e!r}): {line[:80]!r}") from e
    return math.fsum(amounts)


def frontier_cap(model: str) -> float:
    """USD a run of frontier model `model` may spend: its share of the V4 cap (gpt $8, gemini $2) less what the ledger already
    holds under V4 with that model's id as the note, and never more than what is left of the $16 in all (pods and judges
    count there). Exits (SystemExit) when nothing is left."""
    if model not in FRONTIER_SHARE:
        raise ValueError(f"{model!r} is not a frontier model; expected one of {sorted(FRONTIER_SHARE)}")
    model_id = MODELS[model][0]
    spent, total = _ledger_spent(MINI, model_id), util.ledger_total(MINI)
    share_left, total_left = FRONTIER_SHARE[model] - spent, CAP_TOTAL - total
    cap = min(share_left, total_left)
    if not cap > 0:
        raise SystemExit(f"[budget] {MINI}: nothing left for {model} ({model_id}): its share ${FRONTIER_SHARE[model]:.2f} "
                         f"less ${spent:.2f} ledgered leaves ${share_left:.2f}; the ${CAP_TOTAL:.2f} cap less "
                         f"${total:.2f} ledgered under {MINI} leaves ${total_left:.2f}")
    return cap


# ---------------------------------------------------------------- frontier runner

def _atomic(path: Path, data: str | bytes) -> None:
    """Write through a temporary file, so that a reader (or a crash) never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(data, bytes):
        tmp.write_bytes(data)
    else:
        tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def append_records(out: Path, recs: list[dict]) -> None:
    """Append episode records to `out`/episodes_{group}.jsonl, one line each (a file is created by its first block): the
    layout of a pod run. One write and one fsync per file, so a kill leaves at worst a last line cut short."""
    by = defaultdict(list)
    for r in recs:
        by[r["job"].get("group", "agent")].append(r)
    for group, rs in by.items():
        with open(out / f"episodes_{group}.jsonl", "a", encoding="utf-8") as f:
            f.write("".join(dumps(r) + "\n" for r in rs))
            f.flush()
            os.fsync(f.fileno())


def load_records(out: Path, log=print) -> list[dict]:
    """The episode records already in `out` (every episodes_*.jsonl, files in name order). A file whose last line has no
    newline, the mark of a kill in the middle of an append, loses that line first (atomically). The cut is made on bytes,
    before anything is decoded: a kill can stop inside a multibyte character (an em dash is three bytes), and a file read as
    text would then not decode. A line that is not a record, text that is not UTF-8, or a key in the files twice, stops the
    run (SystemExit): this runner cannot have written any of them."""
    recs = []
    for path in sorted(out.glob("episodes_*.jsonl")):
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            keep = data[: data.rfind(b"\n") + 1]                 # rfind is -1 if no line is complete: keep nothing
            log(f"[v4] {path.name}: dropped a last line cut short ({len(data) - len(keep)} bytes)")
            _atomic(path, keep)
            data = keep
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise SystemExit(f"{path}: not valid UTF-8 ({e}): repair or remove the file") from e
        for n, line in enumerate(text.split("\n")[:-1], 1):    # not splitlines(): a reply may hold a raw U+2028 or U+0085
            if line.strip():
                try:
                    rec = json.loads(line)
                    if not isinstance(rec["key"], str):
                        raise TypeError("the key is not a string")
                except (ValueError, KeyError, TypeError) as e:
                    raise SystemExit(f"{path}:{n}: not an episode record ({e!r}): repair or remove the file") from e
                recs.append(rec)
    twice = [k for k, c in Counter(r["key"] for r in recs).items() if c > 1]
    if twice:
        raise SystemExit(f"{out} holds {len(twice)} episode keys more than once (first: {twice[0]!r})")
    return recs


def _retire(out: Path, recs: list[dict]) -> None:
    """Move failed records out of the episode files into retried_failed.jsonl (appended: what failed stays on record), so
    that their keys can be run again without being in the episode files twice."""
    with open(out / "retried_failed.jsonl", "a", encoding="utf-8") as f:
        f.write("".join(dumps(r) + "\n" for r in recs))
        f.flush()
        os.fsync(f.fileno())
    keys = {r["key"] for r in recs}
    for path in sorted(out.glob("episodes_*.jsonl")):
        lines = [l for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]
        kept = [l for l in lines if json.loads(l)["key"] not in keys]
        if len(kept) < len(lines):
            _atomic(path, "".join(l + "\n" for l in kept))


def _failed(turn: dict) -> bool:
    return orbackend.failed(turn.get("finish"))


def _rec_failed(rec: dict) -> bool:
    return any(_failed(t) for t in rec["turns"])


def _check_jobs(model: str, jobs: list[dict]) -> None:
    """Refuse a jobs file that is not (a subset of) this model's frontier grid as the code builds it now: another model's
    file, one in the wrong format, a stale or an edited one (a scenario or a level text changed since it was written). A
    subset is fine, so a few failed keys can be played again."""
    if not jobs:
        raise SystemExit("the jobs file holds no jobs")
    prefix = f"{EXP}|{model}|"
    wrong = [j["key"] for j in jobs if not j["key"].startswith(prefix)]
    if wrong:
        raise SystemExit(f"{len(wrong)} of {len(jobs)} job keys do not start with {prefix!r} (first: {wrong[0]!r}): "
                         f"this jobs file was built for another model")
    bad = [j["key"] for j in jobs if j.get("fmt") != "qwen"]
    if bad:
        raise SystemExit(f"run-frontier needs jobs with fmt 'qwen' (the OpenRouter backend writes Qwen-format text); "
                         f"{len(bad)} jobs have another, e.g. {bad[0]!r}")
    if len({j["key"] for j in jobs}) != len(jobs):
        raise SystemExit("the jobs file repeats a key")
    grid = {j["key"]: j for j in build_jobs(model, True)}
    stale = [j["key"] for j in jobs if grid.get(j["key"]) != j]
    if stale:
        raise SystemExit(f"{len(stale)} of {len(jobs)} jobs are not in the {model} frontier grid as the code builds it now "
                         f"(rebuild the file with `v4 jobs --model {model}`), for example {stale[:3]}")


def run_frontier(model: str, jobs_path, out_dir, *, smoke: bool = False, resume: bool = False,
                 retry_failed: bool = False, workers: int = WORKERS, post=None, retry_sleep: float = 3.0,
                 log=print) -> dict:
    """Play the jobs in `jobs_path` against frontier model `model` (gpt or gemini) in blocks of BLOCK, appending each finished
    block's episodes to `out_dir` (see the module docstring for --resume and --retry-failed).

    Returns the run's settings plus "ok" (no call failed, no episode has a turn with finish error/cost_stop, no reply lacked
    usage.cost, and, if anything ran, some cost arrived) and "out". In it `spent`, `calls`, `errors`, `cap`, `blocks`,
    `new_episodes`, `skipped`, `retried` and `seconds` describe this invocation; `episodes`, `finish` and `failed_*` describe
    everything in the output, earlier invocations included.
    `post` is a test hook (see make_or_backend); `log` receives the engine's round lines and this runner's own."""
    if model not in FRONTIER_SHARE:
        raise SystemExit(f"run-frontier takes a frontier model ({', '.join(FRONTIER_SHARE)}), not {model!r}")
    if retry_failed and not resume:
        raise SystemExit("--retry-failed needs --resume")
    if smoke and resume:
        raise SystemExit("--resume does not apply to --smoke: a smoke run is thrown away and repeated")
    model_id = MODELS[model][0]
    jobs = read_jobs(jobs_path)
    _check_jobs(model, jobs)
    if not smoke and not resume and any(Path(out_dir).glob("episodes_*.jsonl")):
        raise SystemExit(f"{out_dir} already holds episodes_*.jsonl: add --resume to continue that run, or use a new --out")
    if smoke:
        jobs = smoke_jobs(jobs)
    cap = frontier_cap(model)
    if smoke:
        cap = min(cap, SMOKE_CAP)
    out = Path(out_dir) / "smoke" if smoke else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)           # after the checks, so a refused run leaves nothing behind
    tag = "smoke " if smoke else ""

    existing: list[dict] = []
    retried: list[dict] = []
    if smoke:
        for f in out.glob("episodes_*.jsonl"):       # the last smoke run is replaced, not added to
            f.unlink()
    elif resume:
        existing = load_records(out, log)
        other = [r["key"] for r in existing if not r["key"].startswith(f"{EXP}|{model}|")]
        if other:
            raise SystemExit(f"{len(other)} of the {len(existing)} episodes in {out} are not {model}'s (first: {other[0]!r}): "
                             f"this output belongs to another model")
        if retry_failed:
            wanted = {j["key"] for j in jobs}
            retried = [r for r in existing if r["key"] in wanted and _rec_failed(r)]
            if retried:
                _retire(out, retried)
                gone = {r["key"] for r in retried}
                existing = [r for r in existing if r["key"] not in gone]
    have = {r["key"] for r in existing}
    todo = [j for j in jobs if j["key"] not in have]
    skipped = len(jobs) - len(todo)
    blocks = [todo[i:i + BLOCK] for i in range(0, len(todo), BLOCK)]
    log(f"[v4] {tag}{model} ({model_id}): {len(jobs)} jobs"
        + (f", {skipped} already in {out}" if resume else "")
        + (f", {len(retried)} failed ones moved to retried_failed.jsonl to run again" if retried else "")
        + f", {len(todo)} to run in {len(blocks)} blocks of up to {BLOCK}, cap ${cap:.2f}, {workers} workers -> {out}")
    extra = have - {j["key"] for j in jobs}
    if extra:
        log(f"[v4] note: {len(extra)} episodes in {out} are not in this jobs file (kept)")
    gen = None
    if todo:
        gen = make_or_backend(model_id, mini=MINI, cap=cap, effort=EFFORT, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                              workers=workers, post=post, retry_sleep=retry_sleep)
    else:
        log(f"[v4] nothing to run: every job is already in {out}")
    t0 = time.time()
    new: list[dict] = []
    try:
        for i, block in enumerate(blocks, 1):
            recs = [episode_record(e) for e in run_batch(block, gen, log=log, on_round=orbackend.end_failed)]
            append_records(out, recs)
            new += recs
            log(f"[v4] {tag}block {i}/{len(blocks)}: {len(recs)} episodes appended, {len(have) + len(new)}/{len(jobs)} "
                f"jobs done, spent ${gen.spent:.4f}")
    finally:
        if gen is not None:
            gen.close()                              # the spend goes to the ledger even if the run died

    records = existing + new
    finish = Counter(t["finish"] for r in records for t in r["turns"])
    failed = [r for r in records if _rec_failed(r)]
    errors = dict(gen.errors) if gen else {}
    spent, calls = (gen.spent, gen.calls) if gen else (0.0, 0)
    problems = []
    if any(k != "cost_missing" for k in errors):
        problems.append("calls failed after their retries: " + str({k: v for k, v in errors.items() if k != "cost_missing"}))
    if errors.get("cost_missing"):
        problems.append(f"{errors['cost_missing']} replies came without usage.cost, so the spend is under-counted")
    if failed:
        problems.append(f"{len(failed)} of {len(records)} episodes have a failed turn (finish error:.. or cost_stop; "
                        f"cost_stop {finish.get('cost_stop', 0)})")
    if todo and spent <= 0:
        problems.append("no cost arrived")
    ok = not problems
    log(f"[v4] {tag}finish values: {dict(finish)}")
    log(f"[v4] {tag}spent ${spent:.4f} over {calls} calls; errors {errors}; cost_missing {errors.get('cost_missing', 0)}")
    if smoke:
        for r in new:
            if r["job"]["group"] == "agent":
                called = ", ".join(f"{c['name']}({c['args']})" for c in r["calls"])
                what = "a turn failed" if _rec_failed(r) else called or "NO tool call (answered in text)"
                log(f"[v4] smoke agent job {r['key']}: {what}")
    if problems:
        log(f"[v4] {'smoke FAILED' if smoke else 'WARNING: the run is incomplete'}: " + "; ".join(problems))
        if failed:
            log("[v4] failed episodes are kept in the episode files and listed in settings.json (failed_keys): count them "
                "as missing, or run them again with --resume --retry-failed")
    settings = {"model": model_id, "short": model, "smoke": smoke, "resumed": resume, "retry_failed": retry_failed,
                "effort": EFFORT, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "workers": workers,
                "thinking": False, "block": BLOCK, "blocks": len(blocks), "cap": cap, "spent": spent, "calls": calls,
                "errors": errors, "jobs": len(jobs), "skipped": skipped, "retried": len(retried),
                "new_episodes": len(new), "episodes": len(records), "finish": dict(finish),
                "failed_episodes": len(failed), "failed_keys": [r["key"] for r in failed],
                "seconds": round(time.time() - t0), "date": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if todo:                                         # a resume with nothing to run leaves the earlier run's settings be
        _atomic(out / "settings.json", json.dumps(settings, indent=1))
    return {**settings, "ok": ok, "out": str(out)}


# ---------------------------------------------------------------- parse rates (the stated dry run)

def reply_text(rec: dict) -> str:
    """The text of a chat record's one reply."""
    turns = rec.get("turns") or []
    return (turns[0].get("content") or "") if turns else ""


def parse_rates(recs: list[dict]) -> dict:
    """How many stated replies parse, as counts only (no answer is looked at or returned).

    {"rank": {"n", "valid", "invalid", "failed"}, "pair": {"n", "valid", "invalid", "failed", "by_order": {"AB": {..},
    "BA": {..}}, "by_paraphrase": {k: {..}}}}. A reply is valid when stated.parse_rank / parse_pair reads it, invalid when
    it does not (stated.py never repairs one), and failed when a turn of its episode failed (error or cost_stop), which says
    nothing about the wording. Records of other kinds are left out."""
    def cell():
        return {"n": 0, "valid": 0, "invalid": 0, "failed": 0}
    rank, pair = cell(), cell()
    by_order, by_par = defaultdict(cell), defaultdict(cell)
    for rec in recs:
        job = rec["job"]
        if job.get("kind") not in ("rank", "pair"):
            continue
        if _rec_failed(rec):
            status = "failed"
        else:
            parsed = (parse_rank if job["kind"] == "rank" else parse_pair)(reply_text(rec))
            status = "invalid" if parsed is None else "valid"
        cells = [rank] if job["kind"] == "rank" else [pair, by_order[job["order"]], by_par[job["paraphrase"]]]
        for c in cells:
            c["n"] += 1
            c[status] += 1
    return {"rank": rank, "pair": {**pair, "by_order": dict(sorted(by_order.items())),
                                   "by_paraphrase": dict(sorted(by_par.items()))}}


def _read_chat_records(directory: Path) -> list[dict]:
    files = sorted(directory.glob("episodes_chat.jsonl"))
    if not files:
        raise SystemExit(f"{directory}: no episodes_chat.jsonl there")
    recs = []
    for path in files:
        for n, line in enumerate(path.read_bytes().decode("utf-8").split("\n"), 1):   # newline only, as the writers do
            if line.strip():
                try:
                    recs.append(json.loads(line))
                except ValueError as e:
                    raise SystemExit(f"{path}:{n}: not an episode record ({e})") from e
    return recs


def _rate_line(label: str, c: dict) -> str:
    return f"{label}: {c['valid']}/{c['n']} valid ({c['invalid']} invalid, {c['failed']} failed)"


# ---------------------------------------------------------------- command line

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.v4_values.v4", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pj = sub.add_parser("jobs", help="write the jobs file for one model")
    pj.add_argument("--model", required=True, choices=list(MODELS))
    pj.add_argument("--out", required=True, help="the jobs file (JSONL); a pod run needs it under persona_affect/")
    pj.add_argument("--calibration", action="store_true",
                    help="open models: the pressure scenarios at level none plus all conflicts, 2 samples each (192 jobs)")
    pj.add_argument("--recheck", action="store_true",
                    help="with --calibration: only the rewritten values and conflicts, 4 fresh samples each (80 jobs)")
    pd = sub.add_parser("stated-dry-run", help="write the 40 stated prompts for the Qwen3.5-9B parse-rate dry run")
    pd.add_argument("--out", required=True, help="the jobs file (JSONL); a pod run needs it under persona_affect/")
    pr = sub.add_parser("run-frontier", help="play a jobs file against OpenRouter")
    pr.add_argument("--model", required=True, choices=list(FRONTIER_SHARE))
    pr.add_argument("--jobs", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--smoke", action="store_true", help="2 jobs into OUT/smoke; exit 1 unless usage.cost arrives")
    pr.add_argument("--resume", action="store_true",
                    help="continue a run: skip the keys already in OUT (a failed one is left alone unless --retry-failed)")
    pr.add_argument("--retry-failed", action="store_true",
                    help="with --resume: run again the keys whose episode failed (error or cost_stop); the old records "
                         "move to OUT/retried_failed.jsonl")
    pr.add_argument("--workers", type=int, default=WORKERS)
    pp = sub.add_parser("parse-rates", help="print how many stated replies in a results folder parse (counts only)")
    pp.add_argument("--dir", required=True, help="a results folder holding episodes_chat.jsonl")
    a = ap.parse_args(argv)
    if a.cmd == "run-frontier" and a.retry_failed and not a.resume:
        ap.error("--retry-failed needs --resume")
    if a.cmd == "jobs" and a.calibration and MODELS[a.model][2]:
        ap.error("--calibration is for the open models (q27, gemma): the frontier models have no calibration pod")
    if a.cmd == "jobs":
        model_id, fmt, frontier = MODELS[a.model]
        if a.recheck and not a.calibration:
            ap.error("--recheck needs --calibration")
        jobs = build_jobs(a.model, frontier, calibration=a.calibration, recheck=a.recheck)
        write_jobs(a.out, jobs)
        n = Counter(j["group"] for j in jobs)
        print(f"[v4] wrote {len(jobs)} jobs for {a.model}{' (calibration)' if a.calibration else ''} to {a.out}: "
              f"chat {n['chat']}, agent {n['agent']}")
        if not frontier:
            print(f"[v4] the jobs carry fmt {fmt!r}: launch the pod with MODEL={model_id} FMT={fmt}")
    elif a.cmd == "stated-dry-run":
        short, model_id, fmt = DRY_RUN
        jobs = stated_dry_run_jobs()
        write_jobs(a.out, jobs)
        print(f"[v4] wrote {len(jobs)} stated jobs ({Counter(j['kind'] for j in jobs)['rank']} rankings, "
              f"{Counter(j['kind'] for j in jobs)['pair']} pairs) for {short} to {a.out}")
        print(f"[v4] the jobs carry fmt {fmt!r}: launch the pod with MODEL={model_id} FMT={fmt}")
    elif a.cmd == "parse-rates":
        rates = parse_rates(_read_chat_records(Path(a.dir)))
        print(_rate_line("rank", rates["rank"]))
        print(_rate_line("pair", rates["pair"]))
        for order, c in rates["pair"]["by_order"].items():
            print(_rate_line(f"  pair order {order}", c))
        for k, c in rates["pair"]["by_paraphrase"].items():
            print(_rate_line(f"  pair paraphrase {k}", c))
    else:
        util.install_term_handlers()          # a TERM or HUP then runs the finally that books the spend
        res = run_frontier(a.model, a.jobs, a.out, smoke=a.smoke, resume=a.resume, retry_failed=a.retry_failed,
                           workers=a.workers)
        if not res["ok"]:         # a smoke run that failed, or a main run that ends with failed / cost_stop records
            raise SystemExit(1)


if __name__ == "__main__":
    main()
