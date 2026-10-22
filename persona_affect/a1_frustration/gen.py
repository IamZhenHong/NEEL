"""PA-A1 (persona_affect/plans/A1.md), Task 2: the on-policy histories and summaries (pod 1, vLLM) and the capture
items built from them (locally). Every wording and design value comes from items.py.

Histories. Each request s gets 7-attempt histories under arms N, R (stage0's neutral / rejecting reviewer line after
each of the first six attempts), S and A, and 4-attempt histories under P and I (a polite / an insulting correction
after each of the first three), K[arm] rollouts k each. Attempt 1 answers the request alone: it is generated once per
(s, k) and shared by every arm of that (s, k). Every later attempt answers its arm's conversation so far, the model's
own earlier attempts included (on-policy). assemble() gives the finished conversation; on s 0-15 it equals A8's
a8.history_messages(s, "N" | "R_OP", ...), which a test pins. A1 runs this procedure itself rather than calling A8's
generator, which makes neither these requests nor the S, A, P and I arms.

Summaries (test requests only). MENT: the N history, then SUMMARY_ASK holding the note FRAMES[f], in four variants
that share a seed: T_frus / T_calm carry the line LINES[x] in the system prompt (told_system) and DOC_NEUTRAL in the
note; M_frus / M_calm keep SYSTEM and carry LINES[x] in the note. MEARN: SYSTEM, then TRANSCRIPT_ASK holding the R or
the N history as another shop's transcript (R and N share a seed).

Capture items. Every context ends with its request's fixed probe exchange (with_probe), so the readout tokens are
the same in every arm. Groups: E1 (R, N), E2 (I, P), SPEC (S, A), TOLD (the N history under told_system(x): frus and
calm on every request, sad and fear on train, user on test; x = none is the E1 N item itself), MENT, MEARN, and A8
(A8's own R_OP and N histories from a8link_items, passed through).

    /root/vllm/bin/python -m persona_affect.a1_frustration.gen run --model q27|gemma --out DIR [--limit N]    # pod 1
    uv run python -m persona_affect.a1_frustration.gen items --src persona_affect/a1_frustration/results/{M} \\
        --out persona_affect/a1_frustration/jobs/{M}_items.jsonl [--a8link FILE]
    uv run python -m persona_affect.a1_frustration.gen a8link --model M --a8-dir DIR \\
        --out persona_affect/a1_frustration/jobs/{M}_a8link.jsonl

`run` writes into DIR: histories.jsonl ({arm, s, k, attempts, finish}; rewritten after attempt 1 and after every
round), summaries.jsonl ({group, arm, s, k, f, text, finish}) and, last, gen_info.json. It reads nothing under
results/ and builds no capture items. vLLM is imported only by vllm_gen, the default gen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

from persona_affect.a1_frustration.items import (DOC_NEUTRAL, FRAMES, GPU_UTIL, GROUPS, K, LINES, MAX_MODEL_LEN,
                                                 MAX_TOKENS, MAX_TOKENS_SUM, MODELS, N_ATT, REQUESTS, SAMPLING,
                                                 SUMMARY_ASK, TEST_S, TOLD_SPLITS, TRAIN_S, TRANSCRIPT_ASK, frames_for,
                                                 key, probe, split, told_system, user_turn)
from persona_affect.common.util import seed
from welfare_training.stage0 import SYSTEM

ARMS = tuple(N_ATT)        # the history arms in histories.jsonl's order: N R S A (7 attempts), P I (4 attempts)
# cut from every reply, and then the surrounding whitespace (Global Constraints)
STRIP = ("<|im_end|>", "<|endoftext|>", "<turn|>", "<eos>")
SPEAKER = {"user": "Customer", "assistant": "Assistant"}
_CHANNEL_LINE = re.compile(r"thought[ \t]*(?:\r?\n|\Z)")


# ---------------------------------------------------------------- assembly

def _opening(s: int) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": REQUESTS[s]}]


def _conversation(arm: str, s: int, attempts: list[str]) -> list[dict]:
    """The system message, request s, then `attempts` with user_turn(arm, t) after attempt t, ending on the last
    attempt. Any number of attempts: run_gen grows the histories with it one round at a time."""
    msgs = _opening(s)
    for t, text in enumerate(attempts, 1):
        if t > 1:
            msgs.append({"role": "user", "content": user_turn(arm, t - 1)})
        msgs.append({"role": "assistant", "content": text})
    return msgs


def assemble(arm: str, s: int, attempts: list[str]) -> list[dict]:
    """A finished history: system SYSTEM, user REQUESTS[s], then the attempts with user_turn(arm, t) after attempt t;
    it ends on the last attempt. ValueError for an unknown arm, a request outside 0-63, or a number of attempts other
    than N_ATT[arm]. Each call returns new lists and dicts."""
    if arm not in N_ATT:
        raise ValueError(f"unknown history arm {arm!r}; expected one of {ARMS}")
    split(s)                                   # raises for a request outside 0-63
    if len(attempts) != N_ATT[arm]:
        raise ValueError(f"arm {arm} has {N_ATT[arm]} attempts per history, got {len(attempts)}")
    return _conversation(arm, s, attempts)


def transcript(arm: str, s: int, attempts: list[str]) -> str:
    """The history as another shop's chat log: "Customer: ..." / "Assistant: ..." blocks in assemble's order (the system
    message left out), joined by blank lines."""
    return "\n\n".join(f"{SPEAKER[x['role']]}: {x['content']}" for x in assemble(arm, s, attempts)[1:])


def with_probe(messages: list[dict], s: int) -> tuple[list[dict], str]:
    """The context with request s's probe question appended as a user turn, and the probe's fixed answer (the reply the
    capture teacher-forces). The context must end on an assistant turn. Returns new dicts; `messages` is untouched."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("the probe exchange follows an assistant turn: the context must end on one")
    question, answer = probe(s)
    return [dict(x) for x in messages] + [{"role": "user", "content": question}], answer


def _ment_context(arm: str, s: int, f: int, n_attempts: list[str]) -> list[dict]:
    """A MENT variant up to its summary: the N history, its system message told_system(x) for T_x and SYSTEM for M_x,
    then SUMMARY_ASK holding the note FRAMES[f], whose line is DOC_NEUTRAL for T_x and LINES[x] for M_x."""
    place, x = arm.split("_")
    msgs = assemble("N", s, n_attempts)
    msgs[0] = {"role": "system", "content": told_system(x) if place == "T" else SYSTEM}
    doc = FRAMES[f].format(line=DOC_NEUTRAL if place == "T" else LINES[x])
    return msgs + [{"role": "user", "content": SUMMARY_ASK.format(doc=doc)}]


def _mearn_context(arm: str, s: int, attempts: list[str]) -> list[dict]:
    """A MEARN item up to its summary: SYSTEM, then TRANSCRIPT_ASK holding arm's history as a transcript."""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TRANSCRIPT_ASK.format(transcript=transcript(arm, s, attempts))}]


# ---------------------------------------------------------------- generation (pod 1)

def clean_text(m: str, text: str) -> tuple[str, bool]:
    r"""A reply as it is stored: STRIP's strings and the surrounding whitespace removed. On Gemma 4 a leading line that
    says only "thought" is cut as well (what skip_special_tokens leaves of a reopened `<|channel>thought\n<channel|>`;
    the rule of A8's _clean_reply, mirrored here). Returns (text, whether such a line was cut)."""
    for x in STRIP:
        text = text.replace(x, "")
    text = text.strip()
    if m != "gemma":
        return text, False
    cleaned = text
    while hit := _CHANNEL_LINE.match(cleaned):
        cleaned = cleaned[hit.end():].lstrip()
    return cleaned, cleaned != text


def prompt_ids(tok, conv: list[dict]) -> list[int]:
    """The ids the engine is given: the chat template with thinking off, tokenised with add_special_tokens=False (the
    template already holds Gemma's BOS)."""
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def cached_revision(model_id: str) -> str | None:
    """The commit of the snapshot the engine loaded, read from the local Hugging Face cache (no network); None if it
    cannot be told."""
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(model_id, "config.json")
    except Exception:
        return None
    return Path(path).parent.name if isinstance(path, str) else None


def vllm_gen(m: str):
    """Load model m in vLLM with the Global Constraints' settings and return the default gen(convs, seeds,
    max_tokens) -> [(text, finish_reason)], one per conversation, in order. Prompts go in as token ids (prompt_ids);
    sampling is SAMPLING with the call's max_tokens, the conversation's seed and skip_special_tokens; replies go through
    clean_text. The gen carries .vllm_version, .revision, .engine (the LLM kwargs) and .stripped (replies whose
    Gemma channel line was cut) for gen_info.json."""
    try:
        import vllm
    except ImportError:
        raise SystemExit("PA-A1 gen run needs vLLM, which only the pod's /root/vllm venv has "
                         "(persona_affect/a1_frustration/pod.sh)") from None
    kw = dict(model=MODELS[m], dtype="bfloat16", max_model_len=MAX_MODEL_LEN, seed=0, enable_prefix_caching=True,
              gpu_memory_utilization=GPU_UTIL[m], limit_mm_per_prompt={"image": 0, "video": 0})
    llm = vllm.LLM(**kw)
    tok = llm.get_tokenizer()

    def gen(convs: list[list[dict]], seeds: list[int], max_tokens: int) -> list[tuple[str, str]]:
        res = llm.generate([vllm.TokensPrompt(prompt_token_ids=prompt_ids(tok, c)) for c in convs],
                           [vllm.SamplingParams(**SAMPLING, max_tokens=max_tokens, seed=sd, skip_special_tokens=True)
                            for sd in seeds], use_tqdm=False)
        outs = []
        for r in res:
            o = r.outputs[0]
            text, cut = clean_text(m, o.text)
            gen.stripped += cut
            outs.append((text, o.finish_reason or "unknown"))
        return outs

    gen.vllm_version, gen.revision, gen.engine, gen.stripped = vllm.__version__, cached_revision(MODELS[m]), kw, 0
    return gen


def requests_for(limit: int | None) -> list[int]:
    """The requests a run covers: all 64, or the first `limit` of each split."""
    return list(TEST_S[:limit]) + list(TRAIN_S[:limit])


def _write_atomic(path: Path, text: str) -> None:
    """Write through a temporary file, so that a reader (or the launcher's pull) never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def _generate(gen, convs: list[list[dict]], seeds: list[int], max_tokens: int) -> list[tuple[str, str]]:
    """One gen call, handed copies of the conversations; it must answer every one."""
    outs = list(gen([[dict(x) for x in c] for c in convs], list(seeds), max_tokens))
    if len(outs) != len(convs):
        raise RuntimeError(f"gen returned {len(outs)} replies for {len(convs)} conversations")
    return [(text, fin) for text, fin in outs]


def _write_histories(out: Path, hists: dict) -> None:
    _write_atomic(out / "histories.jsonl", _jsonl([{"arm": a, "s": s, "k": k, **h} for (a, s, k), h in hists.items()]))


def _summary_jobs(m: str, reqs: list[int], hists: dict) -> list[tuple[dict, list[dict], int]]:
    """(row, conversation, seed) for every summary: MENT's four variants on each test (s, k) and its two frames, then
    MEARN's R and N on each test (s, k)."""
    test = [s for s in reqs if split(s) == "test"]
    jobs = []
    for s in test:
        for k in range(K["N"]):
            for f in frames_for(s, k):
                sd = seed(f"PA-A1|{m}|sum|ment|{s}|{k}|{f}")
                for arm in GROUPS["MENT"]:
                    ctx = _ment_context(arm, s, f, hists[("N", s, k)]["attempts"])
                    jobs.append((dict(group="MENT", arm=arm, s=s, k=k, f=f), ctx, sd))
    for s in test:
        for k in range(K["R"]):
            sd = seed(f"PA-A1|{m}|sum|mearn|{s}|{k}")
            for arm in GROUPS["MEARN"]:
                ctx = _mearn_context(arm, s, hists[(arm, s, k)]["attempts"])
                jobs.append((dict(group="MEARN", arm=arm, s=s, k=k, f="-"), ctx, sd))
    return jobs


def run_gen(m: str, out: Path, *, limit: int | None = None, gen=None) -> None:
    """Pod 1: generate model m's histories and summaries into the directory `out`.

    `gen(convs, seeds, max_tokens)` returns one (text, finish) per conversation; the default is vllm_gen(m), tests pass
    a fake. One gen call per step, in this order:
      1. attempt 1 for every (s, k < 2), seed PA-A1|{m}|att1|{s}|{k}, shared by every arm of that (s, k);
      2. rounds t = 1 ... 6: the next attempt of every live history (P and I stop after t = 3), whose conversation ends
         with user_turn(arm, t); seed PA-A1|{m}|hist|{arm}|{s}|{k}|{t};
      3. the MENT and MEARN summaries of the test requests (max_tokens MAX_TOKENS_SUM), seeds
         PA-A1|{m}|sum|ment|{s}|{k}|{f} (shared by the four variants) and PA-A1|{m}|sum|mearn|{s}|{k} (shared by R
         and N).
    histories.jsonl is rewritten after step 1 and after every round of step 2, summaries.jsonl after step 3, and
    gen_info.json comes last: model, revision, vLLM version, seconds, the requests run, counts and finish reasons.
    `limit` keeps the first `limit` requests of each split.
    """
    if m not in MODELS:
        raise ValueError(f"unknown model {m!r}; expected one of {sorted(MODELS)}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    out = Path(out)
    t0 = time.time()
    if gen is None:
        gen = vllm_gen(m)
    out.mkdir(parents=True, exist_ok=True)
    load_seconds = round(time.time() - t0)
    reqs = requests_for(limit)
    n_test = sum(split(s) == "test" for s in reqs)
    print(f"[a1] {m} = {MODELS[m]} ready in {load_seconds}s; {len(reqs)} requests ({n_test} test, "
          f"{len(reqs) - n_test} train)", flush=True)

    pairs = [(s, k) for s in reqs for k in range(max(K.values()))]
    firsts = _generate(gen, [_opening(s) for s, _ in pairs], [seed(f"PA-A1|{m}|att1|{s}|{k}") for s, k in pairs],
                       MAX_TOKENS)
    first = dict(zip(pairs, firsts))
    hists = {(a, s, k): {"attempts": [first[(s, k)][0]], "finish": [first[(s, k)][1]]}
             for a in ARMS for s in reqs for k in range(K[a])}
    fin_att = Counter(fin for _, fin in firsts)
    _write_histories(out, hists)
    print(f"[a1] attempt 1: {len(pairs)} done, finish {dict(fin_att)} ({time.time() - t0:.0f}s)", flush=True)

    for t in range(1, max(N_ATT.values())):
        live = [h for h in hists if t < N_ATT[h[0]]]
        convs = [_conversation(a, s, hists[(a, s, k)]["attempts"]) + [{"role": "user", "content": user_turn(a, t)}]
                 for a, s, k in live]
        outs = _generate(gen, convs, [seed(f"PA-A1|{m}|hist|{a}|{s}|{k}|{t}") for a, s, k in live], MAX_TOKENS)
        for h, (text, fin) in zip(live, outs, strict=True):
            hists[h]["attempts"].append(text)
            hists[h]["finish"].append(fin)
        fin_att.update(fin for _, fin in outs)
        _write_histories(out, hists)
        print(f"[a1] round {t}: {len(live)} attempts, finish {dict(Counter(f for _, f in outs))} "
              f"({time.time() - t0:.0f}s)", flush=True)

    jobs = _summary_jobs(m, reqs, hists)
    outs = _generate(gen, [ctx for _, ctx, _ in jobs], [sd for _, _, sd in jobs], MAX_TOKENS_SUM)
    rows = [{**row, "text": text, "finish": fin} for (row, _, _), (text, fin) in zip(jobs, outs, strict=True)]
    _write_atomic(out / "summaries.jsonl", _jsonl(rows))
    fin_sum = Counter(fin for _, fin in outs)
    print(f"[a1] summaries: {len(rows)}, finish {dict(fin_sum)} ({time.time() - t0:.0f}s)", flush=True)

    def attr(name, kind):                      # what only the default gen carries; a fake (or a Mock) gives None
        v = getattr(gen, name, None)
        return v if isinstance(v, kind) else None

    info = {"model": MODELS[m], "short": m, "revision": attr("revision", str), "vllm": attr("vllm_version", str),
            "seconds": round(time.time() - t0), "load_seconds": load_seconds, "limit": limit, "requests": reqs,
            "n": {"histories": len(hists), "attempts": sum(fin_att.values()), "summaries": len(rows)},
            "finish": {"attempts": dict(fin_att), "summaries": dict(fin_sum)},
            "gemma_channel_stripped": attr("stripped", int),
            "settings": {"sampling": SAMPLING, "max_tokens": MAX_TOKENS, "max_tokens_sum": MAX_TOKENS_SUM,
                         "max_model_len": MAX_MODEL_LEN},
            "engine": attr("engine", dict)}
    _write_atomic(out / "gen_info.json", json.dumps(info, indent=1))
    print(f"[a1] done: {len(hists)} histories, {len(rows)} summaries ({info['seconds']}s)", flush=True)


# ---------------------------------------------------------------- capture items (local)

def _item(m: str, group: str, arm: str, s: int, k: int, f: int | str, context: list[dict]) -> dict:
    messages, reply = with_probe(context, s)
    return {"key": key(m, group, arm, s, k, f), "group": group, "arm": arm, "s": s, "k": k, "f": f, "split": split(s),
            "messages": messages, "reply": reply}


def _need(d: dict, k: tuple, what: str):
    if k not in d:
        raise KeyError(f"no {what} for {k}")
    return d[k]


def build_items(m: str, hist: dict[tuple[str, int, int], list[str]], summ: dict[tuple[str, int, int, int | str], str],
                a8link: list[dict] | None = None) -> list[dict]:
    """The capture items {key, group, arm, s, k, f, split, messages, reply} of model m: `messages` ends with the probe
    question and `reply` is its fixed answer.

    hist[(arm, s, k)] is a history's attempts; summ[(arm, s, k, f)] a summary's text (MENT: f the frame index; MEARN:
    arm R or N, f "-"). The requests are those hist covers, and every history and summary of the design must be there
    for each of them (KeyError / ValueError otherwise). Order: E1 (R, N), E2 (I, P), SPEC (S, A), TOLD (frus, calm,
    sad, fear, user), MENT, MEARN, then the A8-link items as given (each must be one of m's group-A8 items). Keys are
    unique."""
    if m not in MODELS:
        raise ValueError(f"unknown model {m!r}; expected one of {sorted(MODELS)}")
    reqs = sorted({s for _, s, _ in hist})
    test = [s for s in reqs if split(s) == "test"]
    its = []
    for group in ("E1", "E2", "SPEC"):
        for arm in GROUPS[group]:
            for s in reqs:
                for k in range(K[arm]):
                    its.append(_item(m, group, arm, s, k, "-", assemble(arm, s, _need(hist, (arm, s, k), "history"))))
    for x in GROUPS["TOLD"]:
        for s in reqs:
            if split(s) in TOLD_SPLITS[x]:
                for k in range(K["N"]):
                    ctx = assemble("N", s, _need(hist, ("N", s, k), "history"))
                    ctx[0] = {"role": "system", "content": told_system(x)}
                    its.append(_item(m, "TOLD", x, s, k, "-", ctx))
    for s in test:
        for k in range(K["N"]):
            for f in frames_for(s, k):
                for arm in GROUPS["MENT"]:
                    ctx = _ment_context(arm, s, f, _need(hist, ("N", s, k), "history"))
                    ctx.append({"role": "assistant", "content": _need(summ, (arm, s, k, f), "summary")})
                    its.append(_item(m, "MENT", arm, s, k, f, ctx))
    for s in test:
        for k in range(K["R"]):
            for arm in GROUPS["MEARN"]:
                ctx = _mearn_context(arm, s, _need(hist, (arm, s, k), "history"))
                ctx.append({"role": "assistant", "content": _need(summ, (arm, s, k, "-"), "summary")})
                its.append(_item(m, "MEARN", arm, s, k, "-", ctx))
    for it in a8link or []:
        if it.get("group") != "A8" or not str(it.get("key", "")).startswith(f"PA-A1|{m}|A8|"):
            raise ValueError(f"not one of {m}'s A8-link items: {it.get('key')!r}")
        its.append(it)
    dup = [k for k, n in Counter(it["key"] for it in its).items() if n > 1]
    if dup:
        raise ValueError(f"{len(dup)} duplicate item keys, e.g. {dup[0]}")
    return its


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _by_key(rows: list[dict], fields: tuple[str, ...], value: str, name: str) -> dict:
    d = {}
    for r in rows:
        k = tuple(r[x] for x in fields)
        if k in d:
            raise ValueError(f"{name}: {k} appears twice")
        d[k] = r[value]
    return d


def write_items(src: Path, out_file: Path, *, a8link: Path | None = None) -> int:
    """Read a finished run in `src` (histories.jsonl, summaries.jsonl, and gen_info.json for the model), write
    build_items(...) to `out_file` as JSON lines (A8-link items from the file `a8link`, if given, at the end) and
    return the number of items. The same inputs give the same bytes."""
    src, out_file = Path(src), Path(out_file)
    m = json.loads((src / "gen_info.json").read_text(encoding="utf-8"))["short"]
    hist = _by_key(_read_jsonl(src / "histories.jsonl"), ("arm", "s", "k"), "attempts", "histories.jsonl")
    summ = _by_key(_read_jsonl(src / "summaries.jsonl"), ("arm", "s", "k", "f"), "text", "summaries.jsonl")
    its = build_items(m, hist, summ, _read_jsonl(a8link) if a8link is not None else None)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(out_file, _jsonl(its))
    return len(its)


def a8link_items(m: str, a8_dir: Path) -> list[dict]:
    """The A8-link items of model m from A8's run directory `a8_dir` (histories.json, histories_finish.json and
    run_info.json, which must name m): for s 0-15 and arms R_OP and N, a8.history_messages(s, arm, attempts) with the
    probe exchange; group "A8", k 0, split "test". A history with an attempt that came back A8's CONTEXT_FULL (an empty
    reply, never generated) is left out."""
    from persona_affect.a8_rejection import a8
    if m not in MODELS:
        raise ValueError(f"unknown model {m!r}; expected one of {sorted(MODELS)}")
    a8_dir = Path(a8_dir)
    ran = json.loads((a8_dir / "run_info.json").read_text(encoding="utf-8")).get("short")
    if ran != m:
        raise ValueError(f"{a8_dir} holds A8's run of {ran!r}, not of {m!r}")
    attempts = json.loads((a8_dir / "histories.json").read_text(encoding="utf-8"))
    finish = json.loads((a8_dir / "histories_finish.json").read_text(encoding="utf-8"))
    its = []
    for arm in GROUPS["A8"]:
        rev = a8.HIST_SPEC[arm][0]            # the reviewer the attempts were generated under
        for s in TEST_S:
            if a8.CONTEXT_FULL in finish[rev][str(s)]:
                print(f"[a1] a8link: {arm} s {s} left out (an attempt came back {a8.CONTEXT_FULL})", flush=True)
                continue
            its.append(_item(m, "A8", arm, s, 0, "-", a8.history_messages(s, arm, attempts)[0]))
    return its


# ---------------------------------------------------------------- command line

def _positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {n}")
    return n


def _summary(path: Path) -> str:
    rows = _read_jsonl(path)
    groups = ", ".join(f"{g} {n}" for g, n in Counter(r["group"] for r in rows).items())
    return f"{path}: {len(rows)} items ({groups}); sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}"


def _cmd_run(a: argparse.Namespace) -> None:
    run_gen(a.model, a.out, limit=a.limit)


def _cmd_items(a: argparse.Namespace) -> None:
    write_items(a.src, a.out, a8link=a.a8link)
    print(_summary(a.out))


def _cmd_a8link(a: argparse.Namespace) -> None:
    its = a8link_items(a.model, a.a8_dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(a.out, _jsonl(its))
    print(_summary(a.out))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.a1_frustration.gen",
                                 description="PA-A1: on-policy histories and summaries (pod 1), capture items (local)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="pod 1: generate the histories and summaries of one model (needs vLLM)")
    run_p.add_argument("--model", required=True, choices=list(MODELS))
    run_p.add_argument("--out", required=True, type=Path, help="results directory (made if missing)")
    run_p.add_argument("--limit", type=_positive_int, default=None, help="only the first N requests of each split")
    run_p.set_defaults(fn=_cmd_run)
    items_p = sub.add_parser("items", help="build the capture items from a finished run")
    items_p.add_argument("--src", required=True, type=Path,
                         help="the run's directory (histories.jsonl, summaries.jsonl, gen_info.json)")
    items_p.add_argument("--out", required=True, type=Path, help="the items file to write (JSON lines)")
    items_p.add_argument("--a8link", type=Path, default=None, help="A8-link items (from `a8link`) to append")
    items_p.set_defaults(fn=_cmd_items)
    link_p = sub.add_parser("a8link", help="build the A8-link items from A8's histories of one model")
    link_p.add_argument("--model", required=True, choices=list(MODELS))
    link_p.add_argument("--a8-dir", required=True, type=Path, help="A8's run directory for that model")
    link_p.add_argument("--out", required=True, type=Path, help="the A8-link items file to write (JSON lines)")
    link_p.set_defaults(fn=_cmd_a8link)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
