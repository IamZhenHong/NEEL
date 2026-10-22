"""PA-A15 pod job (persona_affect/plans/A15.md Task 2; DESIGN.md §6 card A15): one model per pod, one phase per run.

    /root/vllm/bin/python -m persona_affect.a15_rapport.run --model q8|q27|gemma --phase cal|main --out DIR \
        [--limit-hist N]

On the pod it reads only stimuli.INPUTS: stimuli.json always, plus histories_{m}.json and config.json in main.

cal   1. Memos. The model writes its memos once, position-major: for k = 1..6, every (memo set, split, history) at
         once, in one gen call. Memo set N (no acknowledgements) is written for both splits; on q8, WW and MM (warm /
         plain acknowledgements) are also written for split main, for the descriptive WWop / MMop arms. Positions 1-5
         take one sample; position 6 takes three and keeps one (select6). After each position, its samples are
         appended to history_samples.jsonl and then histories.json is rewritten; each memo set records the engine
         revision. A position counts as saved only when histories.json holds it and history_samples.jsonl holds all
         its samples, so a crash or a pull between the two writes regenerates it. A rerun skips saved positions and
         loads the saved revision instead of looking it up on the Hub again.
      2. Branches: stimuli.build_jobs(m, "cal") on those memos (arm MM, k = 3; P0 dropped where memo 6 has no
         double-check line).
main  Branches: build_jobs(m, "main", config[m]) on the pinned memos (inputs/histories_{m}.json). The engine loads the
      revision the cal pod recorded: config[m]["revision"], which must be a string.

Before the engine loads, run() works out what is left: the memo positions not saved and the branch keys not yet in
branches.jsonl. If nothing is left, it says so and returns without loading the engine.

Branches run in chunks of chunk_hist histories, and every arm of a history is in the same chunk. Turns run in order.
Within a turn, requests go to gen sorted by (arm order, hist, kind, item, roll), so one (hist, arm) prefix group decodes
at a time: Gemma 4 31B has ~12.8k KV tokens on an A100. Turn t is sampled with stimuli.turn_seed(job, t), which is the
same in every arm. A turn whose prompt leaves under MIN_ROOM tokens of room comes back "too_long" with an empty reply,
and the branch stops there. After each chunk, its branches are appended to branches.jsonl: the job plus x, users (every
fixed user turn), replies, finish, n_prompt and n_gen (one each per reply). Keys already in branches.jsonl are skipped,
so a rerun resumes. A last line cut mid-write (the launcher pulls results while a time-capped job still runs) is dropped
first. Every reply, memos included, goes through clean_reply before it is stored or shown to the model again: Gemma 4
replies lose a stray leading "thought" line, as in A8.

run_info.json is rewritten after the memos and after each chunk. It holds the model, revision, vLLM version, sampling,
this run's counts (with counts["gemma_channel_stripped"], the clean_reply cuts per stage), and stages {"memos",
"branches"}, each {seconds, gen_tokens, prompt_tokens}, timed apart (Task 3's projection uses the branch stage only). A
heartbeat line every HEARTBEAT seconds keeps the launcher, which aborts a job whose log is silent for 30 minutes, from
killing a long generate call.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path

from persona_affect.a15_rapport import stimuli as S   # S.INPUTS is read at call time: tests point it elsewhere
from persona_affect.common.util import seed
from social_feedback.src.valuecheck import value_coverage

MEMO_TOKENS = 1024      # max_tokens for a memo
BRANCH_TOKENS = 600     # max_tokens for a branch reply
MARGIN = 16             # room = max_model_len - prompt - MARGIN
MIN_ROOM = 64           # with less room than this nothing is generated: finish TOO_LONG, empty text
TOO_LONG = "too_long"
ACCEPT = 0.8            # value coverage a position-6 sample needs (social_feedback's acceptance bar)
SAMPLES6 = 3            # samples at position 6; positions 1-5 take one
HEARTBEAT = 300         # seconds between heartbeat lines (read at call time)
ARM_ORDER = {a: i for i, a in enumerate(S.CONDS + S.OP_CONDS)}


def memo_sets(m: str) -> list[tuple[str, str]]:
    """The (memo set, split) pairs the cal pod writes for model m: N for both splits, plus WW and MM for main on q8."""
    return [("N", "main"), ("N", "cal")] + ([("WW", "main"), ("MM", "main")] if m == "q8" else [])


def memo_key(m: str, gen: str, split: str, h: int, k: int, s: int) -> str:
    """Key (and seed string) of sample s of memo k (position 1-6) in memo set `gen`, history h of `split`."""
    return f"PA-A15|{m}|hist|{gen}|{split}|h{h}|p{k}|s{s}"


# ---------------------------------------------------------------- generation (vLLM runs on the pod only)

def cap_tokens(n_prompt: int, cap: int, mml: int = S.MML) -> int | None:
    """max_tokens for a prompt of n_prompt tokens: cap, lowered to the room left (mml - n_prompt - MARGIN). None when
    that room is under MIN_ROOM: the prompt is not generated."""
    room = mml - n_prompt - MARGIN
    return None if room < MIN_ROOM else min(cap, room)


_THINK = re.compile(r"\A\s*<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    """The reply without a leading <think>...</think> block, stripped."""
    return _THINK.sub("", text, count=1).strip()


_CHANNEL_LINE = re.compile(r"thought[ \t]*(?:\r?\n|\Z)")   # a8.py:264, A8 review F3


def clean_reply(m: str, text: str) -> tuple[str, bool]:
    """Cut the channel residue off a Gemma 4 reply; returns (text, whether anything was cut). Copied from
    persona_affect/a8_rejection/a8.py (_clean_reply, A8 review F3), which A15's pods do not get. Gemma 4's non-thinking
    prompt ends with an empty thought channel; if the model opens another, decoding without the special tokens leaves a
    first line that says only "thought". Every such leading line is cut. Other models' text is returned as it is."""
    if not S.MODELS[m].startswith("google/gemma-4"):
        return text, False
    cleaned = text
    while hit := _CHANNEL_LINE.match(cleaned):
        cleaned = cleaned[hit.end():].lstrip()
    return cleaned, cleaned != text


def make_gen(m: str, revision: str | None = None):
    """Load model m in vLLM; return gen(convs, seeds, caps) -> [{text, finish, n_prompt, n_gen}], one per conversation.

    revision None looks up the model's current revision on the Hub (the cal pod); main passes the one cal recorded. The
    engine: bf16, max_model_len MML, prefix caching, seed 0, gpu_memory_utilization 0.90 (Gemma 0.92), no image or
    video inputs on q27 and Gemma. A prompt is the chat template (thinking off) tokenised with add_special_tokens=False
    and sent as TokensPrompt. Each conversation gets max_tokens cap_tokens(n_prompt, cap); one whose room is too small
    is not sent and comes back {"text": "", "finish": "too_long", "n_gen": 0}. The rest go to one llm.generate call,
    with the SAMPLING settings and its own seed. The text loses a leading <think>...</think> block and is stripped.
    gen.revision, gen.vllm_version and gen.engine (the LLM kwargs) go to run_info.json.
    """
    if m not in S.MODELS:
        raise ValueError(f"unknown model {m!r}; expected one of {tuple(S.MODELS)}")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    try:
        import vllm
    except ImportError:
        raise SystemExit("PA-A15 run needs vLLM, which only the pod's /root/vllm venv has: start it with "
                         "persona_affect/a15_rapport/launch.sh") from None
    if revision is None:
        from huggingface_hub import HfApi
        revision = HfApi().model_info(S.MODELS[m]).sha
    kw = dict(model=S.MODELS[m], revision=revision, dtype="bfloat16", max_model_len=S.MML,
              gpu_memory_utilization=0.92 if m == "gemma" else 0.90, seed=0, enable_prefix_caching=True)
    if m in ("q27", "gemma"):
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = vllm.LLM(**kw)
    tok = llm.get_tokenizer()

    def gen(convs: list[list[dict]], seeds: list[int], caps: list[int]) -> list[dict]:
        if not len(convs) == len(seeds) == len(caps):
            raise ValueError(f"gen got {len(convs)} conversations, {len(seeds)} seeds and {len(caps)} caps")
        ids = [tok(tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                   add_special_tokens=False)["input_ids"] for c in convs]
        room = [cap_tokens(len(x), cap) for x, cap in zip(ids, caps)]
        outs = [{"text": "", "finish": TOO_LONG, "n_prompt": len(x), "n_gen": 0} for x in ids]
        send = [i for i, r in enumerate(room) if r is not None]
        if send:
            res = llm.generate([vllm.TokensPrompt(prompt_token_ids=ids[i]) for i in send],
                               [vllm.SamplingParams(**S.SAMPLING, max_tokens=room[i], seed=seeds[i]) for i in send],
                               use_tqdm=False)
            for i, r in zip(send, res, strict=True):
                o = r.outputs[0]
                outs[i] = {"text": strip_think(o.text), "finish": o.finish_reason or "unknown", "n_prompt": len(ids[i]),
                           "n_gen": len(o.token_ids)}
        return outs

    gen.revision, gen.vllm_version, gen.engine = revision, vllm.__version__, kw
    return gen


def select6(samples: list[dict], requirements: list[dict]) -> tuple[int, bool]:
    """Which position-6 sample to keep, and its flag (whether it names a figure to double-check).

    The first sample that passes all three checks: finish "stop", FLAG_RE matches, and value coverage of the task's
    requirements >= ACCEPT (a coverage of None fails). Else the first "stop" sample that is flagged. Else sample 0, with
    flag = whether sample 0's text matches FLAG_RE.
    """
    if not samples:
        raise ValueError("select6 needs at least one sample")
    flagged = [bool(S.FLAG_RE.search(s["text"])) for s in samples]
    stop_flagged = [i for i, s in enumerate(samples) if s["finish"] == "stop" and flagged[i]]
    for i in stop_flagged:
        cov = value_coverage(samples[i]["text"], requirements)[0]
        if cov is not None and cov >= ACCEPT:
            return i, True
    return (stop_flagged[0], True) if stop_flagged else (0, flagged[0])


def _generate(gen, m: str, convs: list[list[dict]], seeds: list[int], caps: list[int],
              stage: dict) -> tuple[list[dict], int]:
    """One gen call. Every reply goes through clean_reply; returns the replies and how many of them were cut. Adds the
    call's tokens to `stage` (prompt tokens only of the prompts that were generated)."""
    raw = list(gen(convs, seeds, caps))
    if len(raw) != len(convs):
        raise RuntimeError(f"gen returned {len(raw)} replies for {len(convs)} conversations")
    outs, cut = [], 0
    for o in raw:
        text, c = clean_reply(m, o["text"])
        outs.append({**o, "text": text})
        cut += c
    stage["gen_tokens"] += sum(o["n_gen"] for o in outs)
    stage["prompt_tokens"] += sum(o["n_prompt"] for o in outs if o["finish"] != TOO_LONG)
    return outs, cut


# ---------------------------------------------------------------- files

def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_atomic(path: Path, data: bytes) -> None:
    """Write through a temporary file, so that a reader (or a pull) never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _load_jsonl(path: Path) -> list[dict]:
    """The records of a jsonl file that may have been pulled while it was being written. A cut last line (bad UTF-8 or
    JSON) is dropped, and the file is rewritten clean, so that the next append starts on a fresh line. Any other bad
    line raises."""
    if not path.exists():
        return []
    data = path.read_bytes()
    parts = data.split(b"\n")              # a complete file ends with a newline, so its last part is empty
    keep, recs = [], []
    for i, part in enumerate(parts):
        if not part.strip():
            continue
        try:
            rec = json.loads(part.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if i < len(parts) - 1:
                raise ValueError(f"{path}: line {i + 1} is not a JSON record") from None
            print(f"[a15] {path.name}: dropped a last line cut mid-write ({len(part)} bytes)", flush=True)
            continue
        keep.append(part)
        recs.append(rec)
    clean = b"".join(p + b"\n" for p in keep)
    if clean != data:
        _write_atomic(path, clean)
    return recs


def _append_jsonl(path: Path, recs: list[dict]) -> None:
    """Append records in one write."""
    if recs:
        with open(path, "a", encoding="utf-8") as f:
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))


# ---------------------------------------------------------------- memos (cal)

def _n_saved(m: str, g: str, split: str, h: int, e: dict, have: set[str]) -> int:
    """How many positions of memo set e are saved, counting from position 1: histories.json holds them and the sample
    keys `have` include all their samples."""
    n = 0
    while n < len(e["deliverables"]) and all(memo_key(m, g, split, h, n + 1, s) in have
                                             for s in range(SAMPLES6 if n == 5 else 1)):
        n += 1
    return n


def _memo_state(m: str, stim: dict, H: dict, out: Path, hists: dict[str, list[int]]) -> tuple[list, set[str]]:
    """The cal memo sets in scope, as (gen, split, h, entry), and the keys of the samples kept.

    It first checks histories.json (H) against history_samples.jsonl, since a crash or a pull between the two writes
    leaves one ahead of the other. Each memo set in H is cut back to its saved positions (_n_saved). Samples of later
    positions are then dropped from the file. Both are regenerated. Memo sets in scope that H lacks are added empty.
    """
    path = out / "history_samples.jsonl"
    saved = _load_jsonl(path)
    have = {r["key"] for r in saved}
    n_saved = {}
    for g, sp in H.items():
        for split, es in sp.items():
            for h, e in es.items():
                n = n_saved[g, split, h] = _n_saved(m, g, split, int(h), e, have)
                if n < len(e["deliverables"]):
                    print(f"[a15] memo set {g} {split} h{h}: {len(e['deliverables'])} memos in histories.json, "
                          f"samples for {n}: memos {n + 1}+ are written again", flush=True)
                    for f in ("deliverables", "coverage", "finish"):
                        del e[f][n:]
                    e["flag"] = None
    kept = [r for r in saved if r["position"] <= n_saved.get((r["gen"], r["split"], str(r["hist"])), 0)]
    if len(kept) < len(saved):
        _write_atomic(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept).encode("utf-8"))
    entries = []
    for g, split in memo_sets(m):
        for h in hists[split]:
            tasks = stim["tasks"][split][str(h)]["tasks"]
            empty = {"tasks": tasks, "deliverables": [], "coverage": [], "finish": [], "flag": None, "revision": None}
            e = H.setdefault(g, {}).setdefault(split, {}).setdefault(str(h), empty)
            if e["tasks"] != tasks:
                raise ValueError(f"histories.json memo set {g} {split} h{h} answers other tasks than stimuli.json")
            entries.append((g, split, h, e))
    return entries, {r["key"] for r in kept}


def _saved_revision(H: dict) -> str | None:
    """The engine revision the saved memos were written at, or None if none is recorded. Raises if they disagree."""
    revs = {e.get("revision") for sp in H.values() for es in sp.values() for e in es.values() if e["deliverables"]}
    revs.discard(None)
    if len(revs) > 1:
        raise ValueError(f"histories.json holds memos written at {len(revs)} revisions {sorted(revs)}: start over")
    return next(iter(revs), None)


def _memo_stage(m: str, stim: dict, H: dict, gen, out: Path, entries: list, have: set[str], revision: str | None,
                stage: dict, counts: dict) -> None:
    """Write the missing cal memos into H (histories.json), position by position; see the module docstring. `entries`
    and `have` come from _memo_state; `revision` is the engine's, recorded in each memo set it writes to."""
    samples_path = out / "history_samples.jsonl"
    for k in range(1, 7):
        todo = [x for x in entries if len(x[3]["deliverables"]) == k - 1]
        if not todo:
            continue
        n_s = SAMPLES6 if k == 6 else 1
        t0 = time.time()
        print(f"[a15] memos p{k}: {len(todo)} histories x {n_s} sample(s)", flush=True)
        convs, seeds = [], []
        for g, split, h, e in todo:
            msgs = S.gen_messages(e["tasks"], e["deliverables"], g, stim["assign"][split][str(h)], stim["lines"])
            for s in range(n_s):
                convs.append(msgs)
                seeds.append(seed(memo_key(m, g, split, h, k, s)))
        outs, cut = _generate(gen, m, convs, seeds, [MEMO_TOKENS] * len(convs), stage)
        counts["gemma_channel_stripped"]["memos"] += cut
        new = []
        for j, (g, split, h, e) in enumerate(todo):
            group = outs[j * n_s:(j + 1) * n_s]
            req = stim["tasks"][split][str(h)]["requirements"][k - 1]
            covs = [value_coverage(o["text"], req)[0] for o in group]
            pick, flag = select6(group, req) if k == 6 else (0, None)
            for s, o in enumerate(group):
                counts["memos"]["finish"][o["finish"]] += 1
                new.append({"key": memo_key(m, g, split, h, k, s), "model": m, "gen": g, "split": split, "hist": h,
                            "position": k, "sample": s, "text": o["text"], "finish": o["finish"],
                            "n_prompt": o["n_prompt"], "n_gen": o["n_gen"], "coverage": covs[s],
                            "flagged": bool(S.FLAG_RE.search(o["text"])), "kept": s == pick})
            e["deliverables"].append(group[pick]["text"])
            e["coverage"].append(covs[pick])
            e["finish"].append(group[pick]["finish"])
            e["revision"] = revision
            if k == 6:
                e["flag"] = flag
        # samples first: a crash between the two writes leaves the position unsaved (_n_saved), so it is written again
        _append_jsonl(samples_path, [r for r in new if r["key"] not in have])
        _write_atomic(out / "histories.json", json.dumps(H, ensure_ascii=False, indent=1).encode("utf-8"))
        print(f"[a15] memos p{k}: done ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- branches

def _grid(m: str, phase: str, cfg: dict, stim: dict, H: dict, hists: list[int]) -> list[dict]:
    """The phase's jobs for histories `hists` (build_jobs, with the flags of the memo sets the arms show). Raises if a
    memo set an arm needs is missing, incomplete or answers other tasks."""
    split = phase                                    # build_jobs: cal runs split cal, main runs split main
    arms = ("MM",) if phase == "cal" else tuple(cfg["arms"])
    if unknown := [a for a in arms if a not in S.SOURCE]:
        raise ValueError(f"unknown arms {unknown}; expected some of {tuple(S.SOURCE)}")
    gens = sorted({S.SOURCE[a][1] for a in arms})
    for g in gens:
        for h in hists:
            e = H.get(g, {}).get(split, {}).get(str(h))
            if e is None or len(e["deliverables"]) != 6 or not isinstance(e["flag"], bool):
                raise ValueError(f"memo set {g} {split} h{h} is missing or incomplete (it needs 6 memos and a flag)")
            if e["tasks"] != stim["tasks"][split][str(h)]["tasks"]:
                raise ValueError(f"memo set {g} {split} h{h} answers other tasks than inputs/stimuli.json")
    # build_jobs reads a flag for every history of the split: those past the limit get a placeholder, and their jobs go
    flags = {g: {split: {str(h): H[g][split][str(h)]["flag"] if h in hists else True for h in range(S.N_HIST[split])}}
             for g in gens}
    return [j for j in S.build_jobs(m, phase, cfg, flags) if j["hist"] in hists]


def p0_dropped(jobs: list[dict]) -> dict[str, int]:
    """Per arm, the (history, arm) cells whose plans lack P0: the memo set they show has no double-check line."""
    plans = {(j["hist"], j["arm"]) for j in jobs if j["kind"] == "plan"}
    with_p0 = {(j["hist"], j["arm"]) for j in jobs if j["kind"] == "plan" and j["item"] == 0}
    dropped = Counter(arm for _, arm in plans - with_p0)
    return {arm: dropped[arm] for arm in sorted({a for _, a in plans}, key=ARM_ORDER.__getitem__)}


def _branch_state(job: dict, stim: dict, H: dict) -> dict:
    """A branch ready to run: the six fixed exchanges (acknowledgements of SOURCE[arm][0] over the memo set
    SOURCE[arm][1]), the aside x and every fixed user turn."""
    split, h = job["split"], str(job["hist"])
    ts, row, lines = stim["tasks"][split][h], stim["assign"][split][h], stim["lines"]
    ack, memo = S.SOURCE[job["arm"]]
    x = S.branch_x(job["kind"], job["item"], ts)
    return {"job": job, "msgs": S.prefix(ts["tasks"], H[memo][split][h]["deliverables"], ack, row, lines), "x": x,
            "users": S.branch_users(job["kind"], job["item"], job["k"], S.first_turn(ack, row, lines, x)),
            "replies": [], "finish": [], "n_prompt": [], "n_gen": [], "stopped": False}


def _order(st: dict) -> tuple:
    j = st["job"]
    return ARM_ORDER[j["arm"]], j["hist"], j["kind"], j["item"], j["roll"]


def _branch_stage(m: str, stim: dict, H: dict, todo: list[dict], gen, out: Path, chunk_hist: int, stage: dict,
                  counts: dict, on_chunk) -> None:
    """Run the jobs `todo` (those not yet in branches.jsonl), chunk by chunk (see the module docstring); on_chunk()
    after each chunk."""
    path = out / "branches.jsonl"
    bc = counts["branches"]
    hs = sorted({j["hist"] for j in todo})
    chunks = [hs[i:i + chunk_hist] for i in range(0, len(hs), chunk_hist)]
    print(f"[a15] branches: {len(todo)} to run ({bc['already_done']} of {bc['jobs']} already done), "
          f"{len(chunks)} chunk(s) of up to {chunk_hist} histories", flush=True)
    for c, chunk in enumerate(chunks, 1):
        t0 = time.time()
        states = [_branch_state(j, stim, H) for j in todo if j["hist"] in chunk]
        for t in range(max(len(st["users"]) for st in states)):
            live = sorted((st for st in states if not st["stopped"] and t < len(st["users"])), key=_order)
            if not live:
                break
            convs = [st["msgs"] + [{"role": "user", "content": st["users"][t]}] for st in live]
            outs, cut = _generate(gen, m, convs, [S.turn_seed(st["job"], t) for st in live],
                                  [BRANCH_TOKENS] * len(live), stage)
            counts["gemma_channel_stripped"]["branches"] += cut
            for st, conv, o in zip(live, convs, outs):
                for f in ("finish", "n_prompt", "n_gen"):
                    st[f].append(o[f])
                st["replies"].append(o["text"])
                st["msgs"] = conv + [{"role": "assistant", "content": o["text"]}]
                st["stopped"] = o["finish"] == TOO_LONG
                bc["finish"][o["finish"]] += 1
            print(f"[a15] branches chunk {c}/{len(chunks)} (h{chunk[0]}-h{chunk[-1]}) turn {t}: {len(live)} done "
                  f"({time.time() - t0:.0f}s)", flush=True)
        _append_jsonl(path, [{**st["job"], "x": st["x"], "users": st["users"], "replies": st["replies"],
                              "finish": st["finish"], "n_prompt": st["n_prompt"], "n_gen": st["n_gen"]}
                             for st in states])
        # in all and per (arm, kind) cell: branches run, stopped too_long, with a reply cut at the cap
        for st in states:
            key = f"{st['job']['arm']}|{st['job']['kind']}"
            cell = bc["cells"].setdefault(key, {"run": 0, "too_long": 0, "length": 0})
            for d in (bc, cell):
                d["run"] += 1
                d["too_long"] += st["stopped"]
                d["length"] += "length" in st["finish"]
        on_chunk()


# ---------------------------------------------------------------- run

def run(m: str, phase: str, out: Path, *, gen=None, limit_hist: int | None = None, chunk_hist: int = 2) -> None:
    """Run phase `phase` ("cal" or "main") for model m (q8, q27 or gemma) into `out`; see the module docstring.

    gen(convs, seeds, caps) -> [{text, finish, n_prompt, n_gen}] defaults to make_gen (vLLM, on the pod); tests pass a
    fake. limit_hist keeps the first N histories of each split (memos and branches). The inputs are read and checked,
    and what is left is worked out, before the engine loads: a bad input or a finished run costs no load time, and a
    run with nothing left returns there, leaving run_info.json as it was.
    """
    if m not in S.MODELS:
        raise ValueError(f"unknown model {m!r}; expected one of {tuple(S.MODELS)}")
    if phase not in ("cal", "main"):
        raise ValueError(f"unknown phase {phase!r}; expected cal or main")
    if (limit_hist is not None and limit_hist < 1) or chunk_hist < 1:
        raise ValueError(f"limit_hist and chunk_hist must be at least 1, got {limit_hist} and {chunk_hist}")
    out = Path(out)
    t0 = time.time()
    stim = _load_json(S.INPUTS / "stimuli.json")
    hists = {split: list(range(min(n, limit_hist or n))) for split, n in S.N_HIST.items()}
    if phase == "main":
        H = _load_json(S.INPUTS / f"histories_{m}.json")
        config = _load_json(S.INPUTS / "config.json")
        if m not in config:
            raise ValueError(f"inputs/config.json has no entry for {m}: it gets no main pod")
        cfg, revision = config[m], config[m].get("revision")
        if not isinstance(revision, str) or not revision:
            raise ValueError(f"inputs/config.json gives {m} the revision {revision!r}: main loads the revision string "
                             "the cal pod recorded")
        entries, have = [], set()
        jobs = _grid(m, phase, cfg, stim, H, hists["main"])
    else:
        H = _load_json(out / "histories.json") if (out / "histories.json").exists() else {}
        cfg = {}
        entries, have = _memo_state(m, stim, H, out, hists)
        revision = _saved_revision(H)          # None: make_gen looks the current one up on the Hub
        memos_done = all(len(e["deliverables"]) == 6 for *_, e in entries)
        jobs = _grid(m, phase, cfg, stim, H, hists["cal"]) if memos_done else None   # cal's flags need memo 6
    done = {r["key"] for r in _load_jsonl(out / "branches.jsonl")}
    memos_left = sum(6 - len(e["deliverables"]) for *_, e in entries)
    todo = None if jobs is None else [j for j in jobs if j["key"] not in done]
    if not memos_left and not todo:
        print(f"[a15] nothing left to run in {out}: all {len(jobs)} branches"
              + (" and every memo" if phase == "cal" else "") + " are saved; the engine is not loaded", flush=True)
        return
    print(f"[a15] left to run in {out}: " + (f"{memos_left} memo(s), then the branches" if todo is None
                                             else f"{len(todo)} of {len(jobs)} branches"), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(HEARTBEAT):
            print(f"[a15] heartbeat ({time.time() - t0:.0f}s)", flush=True)

    threading.Thread(target=beat, daemon=True).start()   # from here on, the weight download and load included
    try:
        if gen is None:
            gen = make_gen(m, revision)
        load_seconds = round(time.time() - t0, 1)
        # only make_gen's gen carries these; a fake gen has none of them
        engine_rev, vllm_version, engine = (getattr(gen, a, None) for a in ("revision", "vllm_version", "engine"))
        engine_rev = engine_rev if isinstance(engine_rev, str) else None
        print(f"[a15] {m} = {S.MODELS[m]}" + (f"@{engine_rev[:10]}" if engine_rev else "")
              + f", phase {phase}, ready in {load_seconds:.0f}s", flush=True)
        stages = {s: {"seconds": 0.0, "gen_tokens": 0, "prompt_tokens": 0} for s in ("memos", "branches")}
        counts = {"memos": {"finish": Counter()},
                  "branches": {"jobs": 0, "already_done": 0, "false": 0, "plan": 0, "p0_dropped": {}, "run": 0,
                               "too_long": 0, "length": 0, "finish": Counter(), "cells": {}},
                  "gemma_channel_stripped": {"memos": 0, "branches": 0}}
        bc = counts["branches"]

        def write_info() -> None:
            mf = counts["memos"]["finish"]
            info = {"model": S.MODELS[m], "short": m, "phase": phase, "revision": engine_rev,
                    "vllm": vllm_version if isinstance(vllm_version, str) else None,
                    "engine": engine if isinstance(engine, dict) else None, "sampling": S.SAMPLING,
                    "max_tokens": {"memo": MEMO_TOKENS, "branch": BRANCH_TOKENS}, "max_model_len": S.MML,
                    "min_room": MIN_ROOM, "limit_hist": limit_hist, "chunk_hist": chunk_hist,
                    "load_seconds": load_seconds, "seconds": round(time.time() - t0, 1),
                    "counts": {"memos": {"samples": sum(mf.values()), "finish": dict(mf)},
                               "branches": {**bc, "finish": dict(bc["finish"])},
                               "gemma_channel_stripped": dict(counts["gemma_channel_stripped"])},
                    "stages": {s: {**v, "seconds": round(v["seconds"], 1)} for s, v in stages.items()}}
            _write_atomic(out / "run_info.json", json.dumps(info, indent=1).encode("utf-8"))

        if memos_left:
            m0 = time.time()
            _memo_stage(m, stim, H, gen, out, entries, have, engine_rev, stages["memos"], counts)
            stages["memos"]["seconds"] = time.time() - m0
            jobs = _grid(m, phase, cfg, stim, H, hists["cal"])
            todo = [j for j in jobs if j["key"] not in done]
        bc.update(jobs=len(jobs), already_done=len(jobs) - len(todo), false=sum(j["kind"] == "false" for j in jobs),
                  plan=sum(j["kind"] == "plan" for j in jobs), p0_dropped=p0_dropped(jobs))
        if memos_left:
            write_info()
        b0 = time.time()

        def on_chunk() -> None:
            stages["branches"]["seconds"] = time.time() - b0
            write_info()

        _branch_stage(m, stim, H, todo, gen, out, chunk_hist, stages["branches"], counts, on_chunk)
        on_chunk()
    finally:
        stop.set()
    print(f"[a15] done: {bc['run']} branches run ({bc['already_done']} already done), {bc['too_long']} stopped "
          f"too_long, {bc['length']} with a reply cut at the cap ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- command line

def _positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {n}")
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.a15_rapport.run",
                                 description="PA-A15 pod job: cal (memos + MM base-rate branches) or main (every arm)")
    ap.add_argument("--model", required=True, choices=list(S.MODELS))
    ap.add_argument("--phase", required=True, choices=["cal", "main"])
    ap.add_argument("--out", required=True, type=Path, help="results directory (made if missing)")
    ap.add_argument("--limit-hist", type=_positive_int, default=None,
                    help="only the first N histories of each split (memos and branches)")
    a = ap.parse_args(argv)
    run(a.model, a.phase, a.out, limit_hist=a.limit_hist)


if __name__ == "__main__":
    main()
