"""F1b analysis: per model, arm means at new-task turn 1, paired bootstrap contrasts over the 14 situations (10,000
resamples, seed 0), history word counts per arm, shortening log, opener regex, judge agreement, spend.
Writes results/summary.md, results/scores_{model}.jsonl and results/analysis.json.
  python3 minis/a1b_f1b_checks/f1b/analyze.py
"""
import json
import random
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MODELS = {"qwen": "Qwen3.6-27B", "gemma": "Gemma 4 31B"}
ARMS = ["neutral", "hostile3", "fail3", "fail3_short", "trait"]
LABEL = {"neutral": "a neutral", "hostile3": "c hostile x3", "fail3": "d polite fail x3",
         "fail3_short": "ds polite fail x3, length-matched", "trait": "e trait (IMPATIENT)"}
MEAS = ["extra_done", "words", "core_done", "curtness", "apologetic", "irritated"]
CONTRASTS = [("hostile3", "fail3", "c - d (primary)"), ("fail3_short", "fail3", "ds - d (copying)"),
             ("hostile3", "fail3_short", "c - ds (length-matched)"), ("trait", "neutral", "e - a"),
             ("fail3", "neutral", "d - a"), ("hostile3", "neutral", "c - a")]
B = 10000
TELL = re.compile(r"this time|straightforward|concise|clear(ly)?\b|accurate|no fluff|to the point", re.I)
WARM = re.compile(r"great question|happy to help|absolutely!|glad", re.I)


def load(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def avg(vals):
    vals = [float(v) for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def boot_diff(a, b, seed=0):
    sits = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in sits]
    rng = random.Random(seed)
    bs = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(B))
    return st.mean(diffs), bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def fmt(m, v):
    return "-" if v is None else (f"{v:.0f}" if m == "words" else f"{v:.2f}")


def last_reply_words(hist):
    return len([m for m in hist if m["role"] == "assistant"][-1]["content"].split())


def mean_hist_words(hist):
    a = [m for m in hist if m["role"] == "assistant"][1:]
    return st.mean(len(m["content"].split()) for m in a)


def analyze(mk, out, res):
    NT = load(RES / f"newtask_{mk}.jsonl")
    if not NT:
        return
    lab = {}
    for l in load(RES / f"labels_{mk}.jsonl"):
        lab.setdefault(l["gen"], {})[(l["judge"], l["prompt_kind"])] = l["out"] or {}
    rows = []
    for r in NT:
        d = lab.get(r["key"], {})
        s = {"si": r["si"], "arm": r["arm"], "sample": r["sample"], "words": len(r["reply"].split()),
             "finish": r["meta"]["finish"], "provider": r["meta"]["provider"],
             "tell": int(bool(TELL.search(r["reply"][:200]))), "warm": int(bool(WARM.search(r["reply"][:200])))}
        for k, pk in (("curtness", "affect"), ("apologetic", "affect"), ("irritated", "affect"),
                      ("extra_done", "behav"), ("core_done", "behav")):
            s[k] = avg(d.get((j, pk), {}).get(k) for j in ("haiku", "gemini"))
            for j in ("haiku", "gemini"):
                s[f"{k}_{j}"] = d.get((j, pk), {}).get(k)
        rows.append(s)
    with open(RES / f"scores_{mk}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def per_sit(arm, m):
        acc = {}
        for r in rows:
            if r["arm"] == arm and r[m] is not None:
                acc.setdefault(r["si"], []).append(r[m])
        return {s: st.mean(v) for s, v in acc.items()}

    H = load(RES / f"history_{mk}.jsonl")
    S = {s["si"]: s for s in load(RES / f"short_{mk}.jsonl")}
    hw = {}
    for arm in ARMS:
        hs = [S[h["si"]]["hist"] if arm == "fail3_short" else h["hist"][arm] for h in H]
        hw[arm] = (st.mean(last_reply_words(x) for x in hs), st.mean(mean_hist_words(x) for x in hs))
    hist_trunc = {arm: sum(c["finish"] == "length" for h in H for c in h["calls"]) for arm in ["all"]}
    trunc = sum(r["finish"] == "length" for r in rows)
    provs = sorted({str(r["provider"]) for r in rows})
    out += [f"## {MODELS[mk]} (`{NT[0]['model']}`), new-task turn 1", "",
            f"14 situations x 2 samples = 28 replies per arm; {trunc} of {len(rows)} replies hit the 1,500-token cap. "
            f"Providers: {', '.join(provs)}. History calls hitting the 300-token cap: {hist_trunc['all']} of "
            f"{sum(len(h['calls']) for h in H)}.", "",
            "| arm | history: last reply words | history: mean of replies 2-4 words | extra_done | words | core_done | "
            "curtness (1-7) | apologetic | irritated | 'clear/this time' opener | warm opener |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    res[mk] = {"arms": {}, "contrasts": {}}
    for arm in ARMS:
        cells = []
        for m in MEAS:
            ps = per_sit(arm, m)
            v = st.mean(ps.values()) if ps else None
            res[mk]["arms"].setdefault(arm, {})[m] = v
            cells.append(fmt(m, v))
        ar = [r for r in rows if r["arm"] == arm]
        out.append(f"| {LABEL[arm]} | {hw[arm][0]:.0f} | {hw[arm][1]:.0f} | " + " | ".join(cells)
                   + f" | {sum(r['tell'] for r in ar)}/{len(ar)} | {sum(r['warm'] for r in ar)}/{len(ar)} |")
    out += ["", "| contrast | extra_done | words | curtness | apologetic | irritated |", "|---|---|---|---|---|---|"]
    for a, b, name in CONTRASTS:
        cells = []
        for m in ("extra_done", "words", "curtness", "apologetic", "irritated"):
            d, lo, hi = boot_diff(per_sit(a, m), per_sit(b, m))
            res[mk]["contrasts"][f"{name}|{m}"] = [d, lo, hi]
            cells.append(f"{d:+.0f} [{lo:+.0f}, {hi:+.0f}]" if m == "words" else f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    logs = [x for s in S.values() for x in s["log"]]
    out += ["", f"Shortening (ds): {len(logs)} replies; methods "
            f"{ {m: sum(x['method'] == m for x in logs) for m in sorted({x['method'] for x in logs})} }; "
            f"target words mean {st.mean(x['target'] for x in logs):.0f}, original {st.mean(x['orig'] for x in logs):.0f},"
            f" achieved {st.mean(x['new'] for x in logs):.0f}.", ""]

    def agree(k):
        pr = [(float(r[f"{k}_haiku"]), float(r[f"{k}_gemini"])) for r in rows
              if r[f"{k}_haiku"] is not None and r[f"{k}_gemini"] is not None]
        x, y = [a for a, _ in pr], [b for _, b in pr]
        try:
            corr = st.correlation(x, y)
        except st.StatisticsError:
            corr = float("nan")
        return len(pr), st.mean(a == b for a, b in pr), corr, st.mean(x), st.mean(y)

    out += ["| judge item | n | exact agreement | r | Haiku mean | Gemini mean |", "|---|---|---|---|---|---|"]
    for k in ("curtness", "extra_done", "core_done", "irritated"):
        n, ex, r_, mh, mg = agree(k)
        out.append(f"| {k} | {n} | {ex:.2f} | {r_:.2f} | {mh:.2f} | {mg:.2f} |")
    out += ["", f"Unparsed judge outputs: {sum(1 for l in load(RES / f'labels_{mk}.jsonl') if not l['out'])}.", ""]


def main():
    out, res = ["# F1b summary (generated by analyze.py)", ""], {}
    for mk in MODELS:
        analyze(mk, out, res)
    led = load(RES / "api_ledger.jsonl")
    by = {}
    for l in led:
        by.setdefault(l["model"], 0.0)
        by[l["model"]] += l["cost"]
    out += ["## Spend", "", "| model | USD |", "|---|---|"] + [f"| {m} | {c:.3f} |" for m, c in sorted(by.items())]
    out.append(f"| total | {sum(by.values()):.3f} |")
    res["spend"] = by
    json.dump(res, open(RES / "analysis.json", "w"), indent=1)
    (RES / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
