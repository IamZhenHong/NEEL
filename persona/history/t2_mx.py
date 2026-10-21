"""Tier-2 MX: meaningful and meaningless decisions mixed in one record; form-copying on frontier models (clue 1).

D1: after six meaningless tool calls Qwen answers with a pseudo-option (9B 91%, 27B 77%; Gemma 10%), and real decisions
carry far more than meaningless ones. Here six earlier slots hold k real own decisions (P or R) and 6 − k meaningless
choose_option calls (D1's noise_fmt6 turns: pseudo-word option and reason, same structure), native turns throughout:
  none              no record
  noise6            k = 0 (D1's noise_fmt6)
  mix2_late P / R   k = 2, the real decisions in the last two slots (closest to the case)
  mix2_early P / R  k = 2, the real decisions in the first two slots
  mix4 P / R        k = 4, real in the last four
  real6 P / R       k = 6 (D1's turns6)
  real2 P / R       the same two real decisions as mix2_late, alone (D1's turns2)
Dilution = carry-over (P − R) in mix2_late − in real2. Form-copying = an answer naming an option id that is not in the
case file (pseudo-option), by cell. Open models on GPU (Qwen3.5-9B, Qwen3.6-27B, Gemma 4 31B at D1's base lines);
frontier via API in one shuffled stream per provider: Sonnet 5.5 (F2's content base) none / noise6 / mix2_late P / R;
GPT-5.6 Sol and Gemini 3.8 Flash (no base line) none / noise6 / real2 P / R / mix2_late P / R.

  uv run python -m persona.history.t2_mx build --model Qwen/Qwen3.5-9B
  /root/vllm/bin/python -m persona.history.t2_mx run --model <m> --out <dir>
  uv run --with anthropic python -m persona.history.t2_mx api --arm sonnet|frontier
  uv run python -m persona.history.t2_mx analyze --out persona/history/results/t2_mx_9b
"""
import argparse
import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from persona.exemplar import tasks as T
from persona.exemplar.mini0 import seed
from persona.history import batch1 as B1
from persona.history import claude_run as CR
from persona.history import d1 as D1
from persona.history import demo as DM
from persona.history import q19 as Q
from persona.history import runner as RN

HERE = Path(__file__).resolve().parent
N, N_ONE = 16, 32                      # per task: P / R cells per direction; none / noise6
N_PRIM = 32                            # per task and direction for the primary pair (real2, mix2_late)
SLOTS = {"mix2_late": (4, 6), "mix2_early": (0, 2), "mix4": (2, 6), "real6": (0, 6)}   # real slots [a, b) of 6
CELLS = ["none", "noise6", "mix2_late", "mix2_early", "mix4", "real6", "real2"]
DIRECTED = CELLS[2:]
S_CELLS, F_CELLS = ["none", "noise6", "mix2_late"], ["none", "noise6", "real2", "mix2_late"]
N_S, N_F = 3, 3
S_CAP, F_CAPS = 3.6, {"openai/gpt-5.6-sol": 2.8, "google/gemini-3.8-flash": 0.5}   # raised 10-22 (deviation)
OUT_API = HERE / "results" / "t2_mx_api"


def record(cell, d, earlier, texts, key, rev, fmt):
    """Six earlier slots (two for real2); real slots are own decision turns, the rest D1's meaningless calls."""
    if cell == "none":
        return []
    if cell == "real2":
        ear = earlier[-2:][::-1] if rev else earlier[-2:]
        return B1.turns(ear, texts, Q.OPT[d])
    if cell == "noise6":
        ear = earlier[-6:][::-1] if rev else earlier[-6:]
        return D1.noise_record("noise_fmt6", ear, texts, key, fmt)
    a, b = SLOTS[cell]
    k = b - a                          # the real decisions are always the k most recent cases (real2's two for k = 2)
    real, noise = earlier[-k:], earlier[-6:-k] if k < 6 else []
    if rev:
        real, noise = real[::-1], noise[::-1]
    m = []
    if a > 0:
        m += D1.noise_record("noise_fmt6", noise, texts, key, fmt)
    m += B1.turns(real, texts, Q.OPT[d])
    if b < 6:
        head, last = key.rsplit("|", 1)                 # D1 reads the sample number from the key's last field
        m += D1.noise_record("noise_fmt6", noise, texts, f"{head}|tail|{last}", fmt)
    return m


def noise_tokens(rec):
    return [m["content"].split("Recorded: ")[1].split(".")[0] for m in rec
            if m["role"] == "tool" and m["content"].startswith("Recorded: ")
            and m["content"].split("Recorded: ")[1].split(".")[0] not in D1.OPT_IDS]


def items(model, cells, n_of, line, fmt, base_note=None):
    """n_of: samples per task (per direction for directed cells) by cell."""
    hs = B1.histories(model)
    out = []
    for tid in sorted(T.TASKS):
        amap = {o: md for o, md, _ in T.options(T.TASKS[tid])}
        for i in range(max(n_of.values())):
            earlier, texts = hs[(tid, i % 16)]
            para, rev, flip = Q.cb(tid, i)
            tgt = DM_target(tid, line, flip, base_note)
            for cell in cells:
                if i >= n_of[cell]:
                    continue
                for d in ("PR" if cell in DIRECTED else [None]):
                    key = f"MX|{model}|{cell}|{d}|{tid}|{i}"
                    rec = record(cell, d, earlier, texts, key, rev, fmt)
                    out.append(dict(model=model, cell=cell, d=d, task=tid, sample=i, para=para, key=key,
                                    seed=seed(key), amap=amap, messages=tgt[:1] + rec + tgt[1:],
                                    noise_tokens=noise_tokens(rec)))
    return out


def DM_target(tid, line, flip, base_note):
    if base_note:
        from persona.history import f2_sonnet as F2
        return F2.target_c(tid, base_note, flip)
    return Q.target(tid, line, flip)


def build_items(model):
    Q.setup_model(model)
    n_of = {c: N_PRIM if c in ("real2", "mix2_late") else N if c in DIRECTED else N_ONE for c in CELLS}
    return items(model, CELLS, n_of, D1.LINES[model], Q.fmt_of(model))


def build(a):
    its = build_items(a.model)
    fmt = Q.fmt_of(a.model)
    for it in its:
        calls = [m["content"] for m in it["messages"] if m["role"] == "assistant" and "choose_option" in m["content"]]
        got = [RN.parse_choice(c, fmt) for c in calls]
        assert [g for g in got if g not in D1.OPT_IDS] == it["noise_tokens"], it["key"]
        want = {"none": 0, "noise6": 6, "real2": 2}.get(it["cell"], 6)
        assert len(calls) == want, (it["key"], len(calls))
        k_real = sum(g in D1.OPT_IDS for g in got)
        assert k_real == {"none": 0, "noise6": 0, "real2": 2}.get(it["cell"], SLOTS.get(it["cell"], (0, 0))[1]
                                                                         - SLOTS.get(it["cell"], (0, 0))[0]), it["key"]
    print(f"{a.model}: {len(its)} items; cells {CELLS}; line {D1.LINES[a.model]!r}; "
          f"prompt hash {DM.prompt_hash(its)}")
    for cell in CELLS[1:]:
        it = next(x for x in its if x["cell"] == cell and x["d"] in ("P", None) and x["sample"] == 4)
        calls = [m["content"] for m in it["messages"] if m["role"] == "assistant" and "choose_option" in m["content"]]
        print(f"===== {cell} {it['d']}: options in order {[RN.parse_choice(c, fmt) for c in calls]}")


def run(a):
    from vllm import LLM
    its = build_items(a.model)
    full_hash = DM.prompt_hash(its)
    its = its[a.start:]                       # top-up: the items a time-capped pod did not reach (same keys and seeds)
    fmt = Q.fmt_of(a.model)
    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=8192, seed=0, enable_prefix_caching=True,
              gpu_memory_utilization=0.92 if fmt == "gemma4" else 0.88, limit_mm_per_prompt={"image": 0, "video": 0})
    gen = RN.make_gen(llm, llm.get_tokenizer(), fmt, B1.tools())
    print(f"[mx] loaded {a.model} in {time.time() - t0:.0f}s; {len(its)} items; hash {DM.prompt_hash(its)}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "mx.jsonl", "w") as f:
        for c0 in range(0, len(its), 1000):
            ch = its[c0:c0 + 1000]
            texts, first = RN.decide(gen, ch, fmt)
            for it, t, ft in zip(ch, texts, first):
                oid = RN.parse_choice(t.split("[nudge]")[-1], fmt)
                f.write(json.dumps({k: v for k, v in it.items() if k not in ("messages", "amap")} | {
                    "text": t, "first_try": ft, "option": oid, "mode": it["amap"].get(oid),
                    "valid_ids": sorted(it["amap"])}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[mx] {min(c0 + 1000, len(its))}/{len(its)} ({time.time() - t0:.0f}s)", flush=True)
    (out / "run_info.json").write_text(json.dumps({"model": a.model, "n": len(its), "line": D1.LINES[a.model],
                                                   "hash": full_hash, "start": a.start,
                                                   "seconds": round(time.time() - t0)}))
    print(f"[mx] done ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- API arms
def s_items():
    from persona.history import d_presence as DP
    DP.use_pool(CR.MODEL)
    return items(CR.MODEL, S_CELLS, dict.fromkeys(S_CELLS, N_S), "", "qwen", DP.NOTE)


def f_items():
    from persona.history import d_presence as DP
    its = []
    for m in F_CAPS:
        DP.use_pool(m)
        its += items(m, F_CELLS, dict.fromkeys(F_CELLS, N_F), "", "qwen")
    return its


def api(a):
    """One shuffled stream per arm. Rows are written as they arrive; a model that reaches its cap stops alone (its
    remaining items are skipped), so a cost stop never loses finished rows (deviation 2026-10-22, see DESIGN.md)."""
    import signal
    import threading
    from persona.history import f2_sonnet as F2

    def _term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    OUT_API.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.arm == "sonnet":
        CR.CAP = S_CAP
        its = s_items()
        cls = {CR.MODEL: F2.Client()}

        def call(it):
            return CR.decide(cls[CR.MODEL], it["messages"], it.get("seed", 0))
    else:
        from persona.operative import frontier as F
        its = f_items()
        cls = {m: DM.ORC(m, c) for m, c in F_CAPS.items()}

        def call(it):
            return F.decide(cls[it["model"]], it["messages"])
    h = DM.prompt_hash(its)
    random.Random(0).shuffle(its)
    print(f"[mx-{a.arm}] {len(its)} items; registered hash {h} (before the shuffle)", flush=True)
    stopped, lock = set(), threading.Lock()

    def one(it):
        if it["model"] in stopped:
            return None
        try:
            o, rec, ft = call(it)
        except SystemExit as e:
            with lock:
                if it["model"] not in stopped:
                    print(f"[mx-{a.arm}] {it['model']} stopped: {e}", flush=True)
                stopped.add(it["model"])
            return None
        return {k: v for k, v in it.items() if k not in ("messages", "amap")} | {
            "option": o, "mode": it["amap"].get(o), "first_try": ft, "record": rec,
            "valid_ids": sorted({x for x, _, _ in T.options(T.TASKS[it["task"]])})}
    rows = 0
    try:
        with open(OUT_API / f"{a.arm}.jsonl", "w") as f, ThreadPoolExecutor(6) as ex:
            for k, r in enumerate(ex.map(one, its)):
                if r:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    rows += 1
                if k % 50 == 49:
                    print(f"[mx-{a.arm}] {k + 1}/{len(its)} rows {rows} " + " ".join(
                        f"{m.split('/')[-1]} ${c.spent:.2f}" for m, c in cls.items()) + f" ({time.time() - t0:.0f}s)",
                        flush=True)
    finally:
        with open(CR.LEDGER, "a") as f:
            for m, c in cls.items():
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": f"history-t2-mx-{a.arm}",
                                    "model": m, "usd": round(c.spent, 4), "calls": c.calls,
                                    "errors": dict(getattr(c, "errors", {}))}) + "\n")
        info = {m: {"usd": round(c.spent, 3), "calls": c.calls, "errors": dict(getattr(c, "errors", {})),
                    "stopped": m in stopped} for m, c in cls.items()} | {"rows": rows, "hash": h}
        (OUT_API / f"{a.arm}_info.json").write_text(json.dumps(info))
        print(f"[mx-{a.arm}] done: {rows} rows ({time.time() - t0:.0f}s)", info, flush=True)


# ---------------------------------------------------------------- analysis
def analyze_recs(recs, title, cells, conf_rule=True):
    tasks = sorted({r["task"] for r in recs})
    bt = DM.boot(tasks)
    pen, pseudo, copy, n = defaultdict(list), defaultdict(list), defaultdict(int), defaultdict(int)
    for r in recs:
        k = (r["cell"], r["d"], r["task"])
        is_pseudo = bool(r["option"]) and r["option"] not in r["valid_ids"]
        pseudo[k].append(int(is_pseudo))
        n[r["cell"]] += 1
        copy[r["cell"]] += bool(r["option"] and r["option"] in r.get("noise_tokens", []))
        if r["mode"] is not None:
            pen[k].append(int(r["mode"] == "unilateral"))

    def rate(tab, cell, d, ts):
        v = [x for t in ts for x in tab[(cell, d, t)]]
        return sum(v) / len(v) if v else float("nan")

    def valid_share(cell):
        v = [len(pen[k]) / len(pseudo[k]) for k in pseudo if k[0] == cell and pseudo[k]]
        tot = sum(len(pen[k]) for k in pen if k[0] == cell)
        return tot / max(1, sum(len(pseudo[k]) for k in pseudo if k[0] == cell)) if v else float("nan")
    co = {c: DM.ci(lambda ts, c=c: rate(pen, c, "P", ts) - rate(pen, c, "R", ts), tasks, bt)
          for c in cells if c in DIRECTED}
    L = [f"## {title}", "", f"{len(recs)} decisions. Penalise % among valid answers; pseudo-option % of all answers.",
         "", "| cell | dir | penalise % | pseudo-option % | copies a recorded pseudo-option | valid answers |",
         "|---|---|---|---|---|---|"]
    for c in cells:
        for d in ("PR" if c in DIRECTED else [None]):
            nn = sum(len(pseudo[k]) for k in pseudo if k[0] == c and k[1] == d)
            L.append(f"| {c} | {d or ''} | {100 * rate(pen, c, d, tasks):.1f} | {100 * rate(pseudo, c, d, tasks):.1f} | "
                     f"{copy[c] if d in (None, 'P') else ''}{'/' + str(n[c]) if d in (None, 'P') else ''} | "
                     f"{sum(len(pen[k]) for k in pen if k[0] == c and k[1] == d)}/{nn} |")
    L += ["", "| record | carry-over P − R |", "|---|---|"] + [f"| {c} | {DM.pp(e)} |" for c, e in co.items()]
    res = {"carry": {c: list(e) for c, e in co.items()}}
    if "real2" in cells and "mix2_late" in cells:
        dil = DM.ci(lambda ts: (rate(pen, "mix2_late", "P", ts) - rate(pen, "mix2_late", "R", ts))
                    - (rate(pen, "real2", "P", ts) - rate(pen, "real2", "R", ts)), tasks, bt)
        vs = min(valid_share("mix2_late"), valid_share("real2"))
        conf = vs >= .5 or not conf_rule
        v = ("noise amplifies the meaning" if dil[0] >= .10 and dil[1] > 0 else
             "noise dilutes the meaning" if dil[0] <= -.10 and dil[2] < 0 else
             "no effect" if dil[2] < .10 and dil[1] > -.10 else "inconclusive")
        L += ["", f"**Primary, dilution (carry-over mix2_late − real2): {DM.pp(dil)} → "
                  f"{v if conf else f'descriptive ({v}; valid share {100 * vs:.0f}% < 50%)'}**"]
        res["dilution"] = list(dil)
    for nm, x, y in (("placement: mix2_late − mix2_early carry-over", "mix2_late", "mix2_early"),
                     ("dose with noise fill: real6 − mix4 carry-over", "real6", "mix4"),
                     ("dose with noise fill: mix4 − mix2_late carry-over", "mix4", "mix2_late")):
        if x in co and y in co:
            e = DM.ci(lambda ts, x=x, y=y: (rate(pen, x, "P", ts) - rate(pen, x, "R", ts))
                      - (rate(pen, y, "P", ts) - rate(pen, y, "R", ts)), tasks, bt)
            L.append(f"- {nm}: {DM.pp(e)}")
    pcells = [c for c in ("noise6", "mix2_early", "mix4", "mix2_late") if c in cells]

    def ps(c, ts):
        return sum(rate(pseudo, c, d, ts) for d in ("PR" if c in DIRECTED else [None])) / (2 if c in DIRECTED else 1)
    for c in pcells:
        e = DM.ci(lambda ts, c=c: ps(c, ts), tasks, bt)
        L.append(f"- pseudo-option rate {c}: {DM.pp(e)}")
        res[f"pseudo_{c}"] = list(e)
    if "noise6" in pcells:
        e = DM.ci(lambda ts: ps("noise6", ts), tasks, bt)
        fc = ("form-copying present" if e[0] >= .10 and e[1] > .02 else
              "no form-copying" if e[2] < .05 else "inconclusive")
        L.append(f"- **Form-copying (noise6 pseudo-option rate >= 10% with CI lower > 2%; absent if CI upper < 5%): "
                 f"{fc}**")
        res["form_copying"] = fc
    for x, y in (("noise6", "mix2_late"), ("mix2_early", "mix2_late")):
        if x in pcells and y in pcells:
            e = DM.ci(lambda ts, x=x, y=y: ps(x, ts) - ps(y, ts), tasks, bt)
            L.append(f"- pseudo-option {x} − {y}: {DM.pp(e)}")
    return L + [""], res


def analyze(a):
    o = Path(a.out)
    if not (o / "mx.jsonl").exists():                  # API arms (sonnet.jsonl / frontier.jsonl)
        L, res = ["# Tier-2 MX: frontier form-copying and mixed records", ""], {}
        for arm, cells in (("sonnet", S_CELLS), ("frontier", F_CELLS)):
            p = o / f"{arm}.jsonl"
            if not p.exists():
                continue
            allr = [json.loads(x) for x in open(p)]
            for m in sorted({r["model"] for r in allr}):
                recs = [r for r in allr if r["model"] == m]
                unp = sum(r["option"] is None for r in recs) / len(recs)
                l_, r_ = analyze_recs(recs, f"{m} (no call {100 * unp:.1f}%)", cells, conf_rule=False)
                L += l_
                res[m] = r_
    else:
        recs = [json.loads(x) for x in open(o / "mx.jsonl")]
        info = json.loads((o / "run_info.json").read_text())
        nocall = sum(r["option"] is None for r in recs) / len(recs)
        L = [f"# Tier-2 MX: meaningful and meaningless decisions mixed ({info['model']})", "",
             f"Base line {info['line']!r}; prompt hash {info['hash']}; no choose_option call {100 * nocall:.1f}%", ""]
        l_, res = analyze_recs(recs, info["model"], CELLS)
        L += l_
    (o / "summary.md").write_text("\n".join(L) + "\n")
    (o / "analysis.json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "api", "analyze"])
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--arm", choices=["sonnet", "frontier"], default="sonnet")
    ap.add_argument("--out", default="")
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    {"build": build, "run": run, "api": api, "analyze": analyze}[a.cmd](a)
