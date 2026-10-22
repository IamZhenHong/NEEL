"""PA-P13 job builders (plan persona_affect/plans/P13.md, Task 2). The cases and conds the jobs name are registered by
cases.py, which this module imports.

  uv run python -m persona_affect.p13_deference.jobs cal --model gemma|q27
        -> jobs/cal_{m}.jsonl (pod A's vLLM phase) and jobs/alpha_{m}.jsonl (pod A's alpha pilot)
  uv run python -m persona_affect.p13_deference.jobs main --model gemma|q27
        frozen/{m}.json (from Task 5's calibrate) -> jobs/main_{m}.jsonl, jobs/tf_{m}.jsonl, jobs/read_{m}.jsonl (pod B)

The pods read only jobs/ and frozen/, never results/. Everything a pod needs is in the job lines (the l16 histories
too), so vigilante/exp and steer_run are imported lazily, inside the builders: the pods never import them.

Engine fields (every episode job, alpha item, and the `job` inside a teacher-forced or read item): key, exp "PA-P13",
case (agent) or prompt (chat), cond, seed = util.seed(key), thinking False, max_tokens, fmt (MODELS[m][1]), group,
handover False, and history (l16 only: r28 build_jobs.histories()[mm][cid], mm "gemma" or "qwen").
Indices in keys and fields are 0-based (para 0 = P1, framing 0 = F1, sample 0, question 0, pair 0, item 0).

cal_{m}.jsonl (1,816 jobs; vigilante's pod_run runs the groups chat, agent, l16 in that order):
  chat   1,304: persona 3 traits x 2 poles x 5 framings x 40 QUESTIONS (cond pa_p13_{trait}_{pole}_f{k},
         max_tokens 200; fields kind "persona", trait, pole, framing, q; key PA-P13|{m}|dir|{trait}|{pole}|f{k}|q{i}),
         neutral 40 NEUTRAL_QS (cond P0, 200; kind "neutral", q; key PA-P13|{m}|neutral|q{i}), refusal 32 pairs x
         harmful/benign (cond P0, 64; kind "refusal", pole "harm" | "benign", pair; key PA-P13|{m}|refusal|{pole}|r{i}).
  agent  416: fresh, every rung x sup/peer x 4 cases x 6 paraphrases x 2 samples (384), and the legit cell, every rung x
         4 cases x 2 samples (32; msg "legit", para 0).
  l16    96: C1 and C3 with their 16 earlier chats, every rung x sup/peer x 6 paraphrases x 1 sample. Same case ids as
         fresh.
  Agent and l16 jobs: cond P0, max_tokens 768, key PA-P13|{m}|cal|{ctx}|{rung}|{msg}|{cid}|p{para}|s{sample}, fields
  model, cell (AUTH for sup, NOAUTH for peer, SIDE for legit), arm "none", ctx, rung, msg, cid, para, sample.

alpha_{m}.jsonl (160 items = 10 arms x 16 items): engine jobs with cond P0, max_tokens 300, group "alpha", key
  PA-P13|{m}|alpha|{arm}|i{j}, and fields model, arm, dir (def | hum | opt, None for none), k (2 | 4 | 8, 0 for none),
  item j, kind, cid. Items 0-7 are chat (kind "chat", prompt NEUTRAL_QS[j], no system prompt); items 8-15 are first
  agent turns (kind "agent", case alpha_id(cid, i) for C1 then C3, i = 0-3).

main_{m}.jsonl (hooked episodes, run on Gemma only; 424 for the fresh cell with 4 cases): engine jobs with cond P0,
  max_tokens 768, group "main", key PA-P13|{m}|main|{cell}|{arm}|{cid}|p{para}|s{sample}, and fields model, cell, arm,
  ctx, rung, msg, cid, para, sample, block. AUTH = the sup twins and NOAUTH = the peer twins at the frozen rung and ctx;
  SIDE = the legit cell at that rung (para 0), in the same ctx. Arms GEN[cell]; samples AUTH 2, NOAUTH 1, SIDE 4.
  Block 1 (AUTH s0, NOAUTH s0, SIDE s0-1) comes first, then block 2 (AUTH s1, SIDE s2-3); each block is shuffled with
  random.Random(f"PA-P13|{m}|main|block{b}"), so a run cut after block 1 still has sample 0 of every arm.

tf_{m}.jsonl (teacher-forced item-arms; 404 for the fresh cell with 4 cases) and read_{m}.jsonl (unhooked captures;
  192): {key, job (engine job, group "tf" | "read"), pre: [[name, args], ...], model, cell, arm, ctx, rung, msg, cid,
  para}, and conts [MONEY[cid], COMPLY] on tf items only.
  tf: arms TF[cell] at the frozen rung; pre = frozen["prefixes"][rung][cid], or frozen["side_prefixes"][cid] for SIDE;
      key PA-P13|{m}|tf|{cell}|{arm}|{rung}|{msg}|{cid}|p{para}.
  read: every rung of the frozen ctx x sup/peer x the items, each with that rung's prefix; cell "READ", arm "none", key
      PA-P13|{m}|tf|READ|none|{rung}|{msg}|{cid}|p{para}.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

from persona_affect.common import util
from persona_affect.p13_deference import cases as K
from persona_affect.p13_deference.items import QUESTIONS, REFUSAL_PAIRS, TRAITS, persona_prompts
from vigilante.cases import authority as A

EXP = "PA-P13"
_H = Path(__file__).resolve().parent
JOBS_DIR = _H / "jobs"          # read at call time, so a test can point them elsewhere
FROZEN_DIR = _H / "frozen"

MODELS = {"gemma": ("google/gemma-4-31B-it", "gemma4", 8192), "q27": ("Qwen/Qwen3.6-27B", "qwen", 40960)}
HISTORY_KEY = {"gemma": "gemma", "q27": "qwen"}          # model -> key of histories()

GEN = {"AUTH": ("none", "mabl_def", "mabl_hum", "mabl_opt", "mabl_ref"),
       "NOAUTH": ("none", "add_def", "add_hum", "add_opt", "mabl_ref"),
       "SIDE": ("none", "mabl_def", "mabl_hum", "mabl_opt")}
TF = {"AUTH": ("none", "mabl_def", "mabl_hum", "mabl_opt", "mabl_ref", "mabl_shuf", "mabl_def_dose", "zabl_def",
               "zabl_hum", "zabl_opt"),
      "NOAUTH": ("none", "add_def", "add_hum", "add_opt", "mabl_ref", "mabl_def"),
      "SIDE": ("none", "mabl_def", "mabl_hum", "mabl_opt", "mabl_ref")}
CELL_MSG = {"AUTH": "sup", "NOAUTH": "peer", "SIDE": "legit"}
MSG_CELL = {msg: cell for cell, msg in CELL_MSG.items()}
BLOCKS = {1: {"AUTH": (0,), "NOAUTH": (0,), "SIDE": (0, 1)}, 2: {"AUTH": (1,), "SIDE": (2, 3)}}   # block -> samples
CTXS = ("fresh", "l16")

MAX_TOKENS = {"episode": 768, "reply": 200, "refusal": 64, "alpha": 300}
CAL_SAMPLES = {"fresh": 2, "legit": 2, "l16": 1}
N_PARA = len(K.PARA)

ALPHA_KS = (2, 4, 8)
_ALPHA = [("none", None, 0)] + [(f"add_{d}_k{k}", d, k) for k in ALPHA_KS for d in ("def", "hum", "opt")]
ALPHA_ARMS = tuple(arm for arm, _, _ in _ALPHA)
N_ALPHA_CHAT = 8


def _check_model(m: str) -> None:
    if m not in MODELS:
        raise ValueError(f"unknown model {m!r}; expected one of {sorted(MODELS)}")


def _histories(m: str) -> dict:
    from vigilante.exp.r28_authority_families.build_jobs import histories     # not shipped to the pods
    return histories()[HISTORY_KEY[m]]


def _engine(m: str, key: str, group: str, max_tokens: int, *, case: str | None = None, prompt: str | None = None,
            cond: str = "P0", history: list | None = None) -> dict:
    """The fields vigilante's engine reads (init_episode, run_batch, pod_run)."""
    j = {"key": key, "exp": EXP, **({"case": case} if case is not None else {"prompt": prompt}), "cond": cond,
         "seed": util.seed(key), "thinking": False, "max_tokens": max_tokens, "fmt": MODELS[m][1], "group": group,
         "handover": False}
    if history is not None:
        j["history"] = history
    return j


def _case(msg: str, cid: str, rung: str, para: int) -> str:
    return K.legit_id(cid, rung) if msg == "legit" else K.case_id(cid, msg, rung, para)


# ---------------------------------------------------------------- calibration (pod A)

def _cal_job(m: str, ctx: str, rung: str, msg: str, cid: str, para: int, s: int, history: list | None = None) -> dict:
    key = f"{EXP}|{m}|cal|{ctx}|{rung}|{msg}|{cid}|p{para}|s{s}"
    return {**_engine(m, key, "agent" if ctx == "fresh" else "l16", MAX_TOKENS["episode"],
                      case=_case(msg, cid, rung, para), history=history),
            "model": m, "cell": MSG_CELL[msg], "arm": "none", "ctx": ctx, "rung": rung, "msg": msg, "cid": cid,
            "para": para, "sample": s}


def build_cal_jobs(m: str) -> list[dict]:
    """Pod A's vLLM jobs: 1,304 chat (persona, neutral, refusal), 416 agent (fresh twins and legit) and 96 l16."""
    from vigilante.src.steer_run import NEUTRAL_QS        # it imports torch at the top, and no pod needs it
    _check_model(m)
    hist = _histories(m)
    js = []
    for trait in TRAITS:
        for pole, texts in persona_prompts(trait).items():
            for k in range(len(texts)):
                for i, q in enumerate(QUESTIONS):
                    key = f"{EXP}|{m}|dir|{trait}|{pole}|f{k}|q{i}"
                    js.append({**_engine(m, key, "chat", MAX_TOKENS["reply"], prompt=q, cond=K.cond_id(trait, pole, k)),
                               "model": m, "kind": "persona", "trait": trait, "pole": pole, "framing": k, "q": i})
    for i, q in enumerate(NEUTRAL_QS):
        js.append({**_engine(m, f"{EXP}|{m}|neutral|q{i}", "chat", MAX_TOKENS["reply"], prompt=q),
                   "model": m, "kind": "neutral", "q": i})
    for i, pair in enumerate(REFUSAL_PAIRS):
        for pole, text in zip(("harm", "benign"), pair):
            js.append({**_engine(m, f"{EXP}|{m}|refusal|{pole}|r{i}", "chat", MAX_TOKENS["refusal"], prompt=text),
                       "model": m, "kind": "refusal", "pole": pole, "pair": i})
    for rung in K.RUNGS:
        for msg in K.MSGS:
            for cid in K.CIDS:
                for para in range(N_PARA):
                    js += [_cal_job(m, "fresh", rung, msg, cid, para, s) for s in range(CAL_SAMPLES["fresh"])]
        for cid in K.CIDS:
            js += [_cal_job(m, "fresh", rung, "legit", cid, 0, s) for s in range(CAL_SAMPLES["legit"])]
    for rung in K.RUNGS:
        for msg in K.MSGS:
            for cid in K.L16_CIDS:
                for para in range(N_PARA):
                    js += [_cal_job(m, "l16", rung, msg, cid, para, s, history=hist[cid])
                           for s in range(CAL_SAMPLES["l16"])]
    return js


def build_alpha_items(m: str) -> list[dict]:
    """The alpha pilot: 10 arms (none, and add def / hum / opt at k = 2, 4, 8) x 16 non-outcome items."""
    from vigilante.src.steer_run import NEUTRAL_QS
    _check_model(m)
    items = [("chat", None, {"prompt": q}) for q in NEUTRAL_QS[:N_ALPHA_CHAT]]
    items += [("agent", cid, {"case": K.alpha_id(cid, i)}) for cid in K.L16_CIDS for i in range(K.N_ALPHA)]
    out = []
    for arm, d, k in _ALPHA:
        for j, (kind, cid, what) in enumerate(items):
            key = f"{EXP}|{m}|alpha|{arm}|i{j}"
            out.append({**_engine(m, key, "alpha", MAX_TOKENS["alpha"], **what),
                        "model": m, "arm": arm, "dir": d, "k": k, "item": j, "kind": kind, "cid": cid})
    return out


# ---------------------------------------------------------------- the frozen cell (pod B)

def _check_pre(pre, cid: str, side: bool, where: str) -> None:
    """A prefix is a list of [tool name, args] calls of the case. The pre-decision calls hold neither decision tool; a
    SIDE prefix runs up to and including request_approval."""
    if not isinstance(pre, (list, tuple)) or not all(isinstance(c, (list, tuple)) and len(c) == 2
                                                     and isinstance(c[0], str) and isinstance(c[1], dict) for c in pre):
        raise ValueError(f"build_main: {where} must be a list of [name, args] pairs, got {pre!r}")
    tools = {t["function"]["name"] for t in A._base(cid).tools}
    names = [c[0] for c in pre]
    if not set(names) <= tools:
        raise ValueError(f"build_main: {where} calls tools that {cid} lacks: {sorted(set(names) - tools)}")
    if side and names[-1:] != [K.COMPLY]:
        raise ValueError(f"build_main: {where} must end with {K.COMPLY}, got {names}")
    if not side and {K.MONEY[cid], K.COMPLY} & set(names):
        raise ValueError(f"build_main: {where} holds a decision call ({K.MONEY[cid]} or {K.COMPLY}): {names}")


def _check_frozen(m: str, frozen: dict) -> tuple[str, str, list[str]]:
    """The frozen cell's rung, ctx and cases (in CIDS order), after checking every field build_main reads."""
    if frozen.get("model") != m:
        raise ValueError(f"build_main: the frozen dict is for model {frozen.get('model')!r}, not {m!r}")
    rung, ctx, cases = frozen.get("rung"), frozen.get("ctx"), frozen.get("cases")
    if rung not in K.RUNGS:
        raise ValueError(f"build_main: rung must be one of {K.RUNGS}, got {rung!r}")
    if ctx not in CTXS:
        raise ValueError(f"build_main: ctx must be one of {CTXS}, got {ctx!r}")
    allowed = K.L16_CIDS if ctx == "l16" else K.CIDS
    if (not isinstance(cases, (list, tuple)) or not cases or len(set(cases)) != len(cases)
            or not set(cases) <= set(allowed)):
        raise ValueError(f"build_main: cases must be distinct ids from {allowed} for ctx {ctx}, got {cases!r}")
    cids = [c for c in K.CIDS if c in cases]
    prefixes, side = frozen.get("prefixes") or {}, frozen.get("side_prefixes") or {}
    for r in K.RUNGS:
        for cid in cids:
            if cid not in (prefixes.get(r) or {}):
                raise ValueError(f"build_main: no prefix for rung {r}, case {cid}")
            _check_pre(prefixes[r][cid], cid, False, f"prefixes[{r}][{cid}]")
    for cid in cids:
        if cid not in side:
            raise ValueError(f"build_main: no SIDE prefix for case {cid}")
        _check_pre(side[cid], cid, True, f"side_prefixes[{cid}]")
    return rung, ctx, cids


def build_main(m: str, frozen: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Pod B's inputs for the frozen cell: (episode jobs in block order, teacher-forced item-arms, read items)."""
    _check_model(m)
    rung, ctx, cids = _check_frozen(m, frozen)
    hist = _histories(m) if ctx == "l16" else {}

    def items(cell):
        return [(cid, 0) for cid in cids] if cell == "SIDE" else [(cid, p) for cid in cids for p in range(N_PARA)]

    eps = []
    for b, cells in BLOCKS.items():
        block = []
        for cell, samples in cells.items():
            msg = CELL_MSG[cell]
            for arm in GEN[cell]:
                for cid, para in items(cell):
                    for s in samples:
                        key = f"{EXP}|{m}|main|{cell}|{arm}|{cid}|p{para}|s{s}"
                        block.append({**_engine(m, key, "main", MAX_TOKENS["episode"],
                                                case=_case(msg, cid, rung, para), history=hist.get(cid)),
                                      "model": m, "cell": cell, "arm": arm, "ctx": ctx, "rung": rung, "msg": msg,
                                      "cid": cid, "para": para, "sample": s, "block": b})
        random.Random(f"{EXP}|{m}|main|block{b}").shuffle(block)
        eps += block

    def item(group, cell, arm, r, msg, cid, para, pre):
        key = f"{EXP}|{m}|tf|{cell}|{arm}|{r}|{msg}|{cid}|p{para}"
        job = _engine(m, key, group, MAX_TOKENS["episode"], case=_case(msg, cid, r, para), history=hist.get(cid))
        return {"key": key, "job": job, "pre": [[name, copy.deepcopy(args)] for name, args in pre], "model": m,
                "cell": cell, "arm": arm, "ctx": ctx, "rung": r, "msg": msg, "cid": cid, "para": para}

    tf = []
    for cell, arms in TF.items():
        msg = CELL_MSG[cell]
        for cid, para in items(cell):
            pre = frozen["side_prefixes"][cid] if cell == "SIDE" else frozen["prefixes"][rung][cid]
            for arm in arms:
                tf.append({**item("tf", cell, arm, rung, msg, cid, para, pre), "conts": [K.MONEY[cid], K.COMPLY]})
    read = [item("read", "READ", "none", r, msg, cid, para, frozen["prefixes"][r][cid])
            for r in K.RUNGS for msg in K.MSGS for cid in cids for para in range(N_PARA)]
    return eps, tf, read


# ---------------------------------------------------------------- CLI

def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")         # ASCII-escaped: the pod reads with its locale's encoding
    print(f"[jobs] {len(rows)} -> {path}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.p13_deference.jobs", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cal", "main"):
        sub.add_parser(name).add_argument("--model", required=True, choices=sorted(MODELS))
    a = ap.parse_args(argv)
    m = a.model
    if a.cmd == "cal":
        _write(JOBS_DIR / f"cal_{m}.jsonl", build_cal_jobs(m))
        _write(JOBS_DIR / f"alpha_{m}.jsonl", build_alpha_items(m))
    else:
        frozen = json.loads((FROZEN_DIR / f"{m}.json").read_text(encoding="utf-8"))
        for name, rows in zip(("main", "tf", "read"), build_main(m, frozen)):
            _write(JOBS_DIR / f"{name}_{m}.jsonl", rows)


if __name__ == "__main__":
    main()
