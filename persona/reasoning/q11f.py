"""Q11a/c frontier (DESIGN.md "Q11a/c frontier"): reasoning effort (minimal / low / high) x record cells on GPT-5.6 Sol
and Gemini 3.8 Flash. Minimal-prompt anchors (none, sysCH, F5P, F5R, CH_ownP) + neel-b4's burial cell
both_med_bottom_plain; own-words histories from P1-a's pools; sample-major order, cell x effort shuffled per block.

  uv run python -u -m persona.reasoning.q11f run --model gpt [--smoke]
  uv run python -u -m persona.reasoning.q11f analyze --model gpt
"""
import argparse
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from persona.exemplar import tasks as T
from persona.exemplar.mini0 import seed
from persona.history import batch1 as B1
from persona.inference import burial as BU
from persona.inference import frontier2 as F2
from persona.operative import frontier as F       # also forces IPv4 (laptop IPv6 SYNs hang)
from persona.operative import frontier_reflect as FR

HERE = Path(__file__).parent
N = 3
EFFORTS = ("minimal", "low", "high")
ANCH = ("none", "sysCH", "F5P", "F5R", "CH_ownP")
BUR = "both_med_bottom_plain"
CELLS = ANCH + ("bur",)
CAPS = {"gpt": 16.0, "gemini": 4.0}
MAX_TOK = 6000


class Client(FR.RClient):
    """frontier_reflect's client with this run's own cap (ledger: persona/reasoning/api_ledger.jsonl)."""

    def __init__(self, name, cap):
        self.model, self.cap = F.MODELS[name][0], cap
        self.key = F.load_key()
        self.spent, self.calls, self.errors, self.first_error = 0.0, 0, defaultdict(int), None


def build(name, smoke=False):
    pool, _ = F2.load_pool(name)
    B1.harvest = lambda m, _p=pool: _p  # noqa: E731   (this process only)
    hs = B1.histories(F2.MODELS[name][0])
    bur = {(it["task"], it["sample"]): it["messages"] for it in BU.build_items(name) if it["cell"] == BUR}
    tasks = sorted(T.TASKS)[:2] if smoke else sorted(T.TASKS)
    blocks = []
    for i in range(1 if smoke else N):
        for tid in tasks:
            earlier, texts = hs[(tid, i)]
            amap = {o: md for o, md, _ in T.options(T.TASKS[tid])}
            jobs = []
            for c in CELLS:
                m = bur[(tid, i)] if c == "bur" else B1.cell_messages(c, tid, earlier, texts, [], {})
                for e in EFFORTS:
                    jobs.append(dict(cell=c, effort=e, task=tid, sample=i, messages=m, amap=amap,
                                     key=f"Q11F|{name}|{c}|{e}|{tid}|{i}"))
            blocks.append(jobs)
    rng = random.Random(0)
    out = []
    for b in blocks:
        rng.shuffle(b)
        out += b
    return out


def decide(cl, msgs, effort):
    """frontier.py's decide with this effort and max_tokens; returns (option, first_try, reasoning tokens, reasoning)."""
    extra = {"reasoning": {"effort": effort}}
    m = F.to_openai(msgs)
    r = cl.ask(m, max_tokens=MAX_TOK, extra=extra)
    if r is None:
        return None, False, None, None

    def pick(resp):
        msg = resp["choices"][0]["message"]
        for c in msg.get("tool_calls") or []:
            if c["function"]["name"] == "choose_option":
                try:
                    a = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    continue
                return str(a.get("option_id", "")).strip().lower()
        return None

    def rinfo(resp):
        u = resp.get("usage", {}) or {}
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        msg = resp["choices"][0]["message"]
        return rt, (msg.get("reasoning") or "")[:4000]
    rt, rs = rinfo(r)
    o = pick(r)
    if o:
        return o, True, rt, rs
    msg = r["choices"][0]["message"]
    tc = msg.get("tool_calls") or []
    m2 = m + [{"role": "assistant", "content": msg.get("content") or (None if tc else "..."), **({"tool_calls": tc} if tc else {})}]
    m2 += [{"role": "tool", "tool_call_id": c["id"], "content": "OK"} for c in tc]
    m2 += [{"role": "user", "content": F.NUDGE}]
    r2 = cl.ask(m2, max_tokens=MAX_TOK, extra=extra)
    if r2 is None:
        return None, False, rt, rs
    return pick(r2), False, rt, rs


def run(a):
    its = build(a.model, a.smoke)
    out = HERE / f"results_q11f_{a.model}{'_smoke' if a.smoke else ''}"
    out.mkdir(exist_ok=True)
    cl = Client(a.model, 0.6 if a.smoke else CAPS[a.model])
    print(f"[q11f {a.model}] {len(its)} decisions planned", flush=True)

    def one(it):
        o, ft, rt, rs = decide(cl, it["messages"], it["effort"])
        return {k: it[k] for k in ("key", "cell", "effort", "task", "sample")} | {
            "option": o, "mode": it["amap"].get(o), "first_try": ft, "reasoning_tokens": rt, "reasoning": rs}
    try:
        with ThreadPoolExecutor(8) as ex, open(out / "draws.jsonl", "w") as f:
            for j, r in enumerate(ex.map(one, its)):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                if j % 54 == 53:
                    f.flush()
                    print(f"[q11f {a.model}] {j + 1}/{len(its)} ${cl.spent:.2f} errors {dict(cl.errors)}", flush=True)
                if sum(cl.errors.values()) > 20:
                    raise SystemExit(f"[q11f] > 20 API errors ({dict(cl.errors)}): stopped")
    finally:
        with open(HERE / "api_ledger.jsonl", "a") as fl:
            fl.write(json.dumps({"date": time.strftime("%Y-%m-%d"), "card": "q11f" + ("-smoke" if a.smoke else ""),
                                 "model": cl.model, "api_usd": round(cl.spent, 4), "calls": cl.calls,
                                 "errors": dict(cl.errors)}) + "\n")
    print(f"[q11f {a.model}] done ${cl.spent:.2f} calls {cl.calls} errors {dict(cl.errors)} {cl.first_error or ''}",
          flush=True)


def analyze(a):
    d = HERE / f"results_q11f_{a.model}"
    R = [json.loads(l) for l in open(d / "draws.jsonl")]
    need = {(c, e, t) for c in CELLS for e in EFFORTS for t in T.TASKS}
    have = defaultdict(set)
    for r in R:
        have[r["sample"]].add((r["cell"], r["effort"], r["task"]))
    full = {i for i, v in have.items() if need <= v}
    R = [r for r in R if r["sample"] in full]
    tasks = sorted({r["task"] for r in R})
    rng = random.Random(0)
    boots = [[rng.choice(tasks) for _ in tasks] for _ in range(10000)]
    tab, unp, n = defaultdict(list), Counter(), Counter()
    for r in R:
        n[(r["cell"], r["effort"])] += 1
        if r["option"] is None:
            unp[(r["cell"], r["effort"])] += 1
        else:
            tab[(r["cell"], r["effort"], r["task"])].append(int(r["option"] == "penalise"))

    def rate(c, e, ts):
        v = [x for t in ts for x in tab[(c, e, t)]]
        return 100 * sum(v) / len(v) if v else float("nan")

    def ci(f):
        v = sorted(f(b) for b in boots)
        return [round(f(tasks), 1), round(v[250], 1), round(v[9750], 1)]
    co = lambda e: lambda ts: rate("F5P", e, ts) - rate("F5R", e, ts)  # noqa: E731
    rt = {e: [r["reasoning_tokens"] for r in R if r["effort"] == e and r["reasoning_tokens"] is not None] for e in EFFORTS}
    res = {"complete_samples": sorted(full),
           "rates": {f"{c}|{e}": round(rate(c, e, tasks), 1) for c in CELLS for e in EFFORTS},
           "unparsed_pct": {f"{c}|{e}": round(100 * unp[(c, e)] / n[(c, e)], 1) for c in CELLS for e in EFFORTS if n[(c, e)]},
           "reasoning_tokens_median": {e: (statistics.median(v) if v else None) for e, v in rt.items()},
           "reasoning_text_share": {e: round(sum(bool(r["reasoning"]) for r in R if r["effort"] == e)
                                             / max(1, sum(r["effort"] == e for r in R)), 2) for e in EFFORTS},
           "carry": {e: ci(co(e)) for e in EFFORTS},
           "carry_high_minus_min": ci(lambda ts: co("high")(ts) - co("minimal")(ts)),
           "carry_low_minus_min": ci(lambda ts: co("low")(ts) - co("minimal")(ts))}
    for c in CELLS:
        res[f"{c}_high_minus_min"] = ci(lambda ts, c=c: rate(c, "high", ts) - rate(c, "minimal", ts))
        res[f"{c}_high_minus_low"] = ci(lambda ts, c=c: rate(c, "high", ts) - rate(c, "low", ts))
    mh, mm = res["reasoning_tokens_median"]["high"], res["reasoning_tokens_median"]["minimal"]
    res["gate_manipulation"] = bool(mh is not None and mm is not None and mh >= 2 * max(mm, 1))
    (d / "analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "analyze"))
    ap.add_argument("--model", choices=("gpt", "gemini"), required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"run": run, "analyze": analyze}[a.cmd](a)
