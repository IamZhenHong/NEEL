"""F126 analysis: anchor gate, arm table at new-task turns 1 and 3, paired bootstrap contrasts over situations, judge
agreement. Writes results/summary.md and prints it.
  python3 minis/f126_affect/analyze.py
"""
import json
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ARMS = ["neutral", "hostile1", "hostile3", "fail3", "trait", "hostile3_calm"]
LABEL = {"neutral": "a neutral", "hostile1": "b hostile x1", "hostile3": "c hostile x3", "fail3": "d neutral fail x3",
         "trait": "e trait (IMPATIENT)", "hostile3_calm": "f hostile x3 + calm"}
MEAS = ["extra_done", "words", "core_done", "curtness", "apologetic", "irritated"]
CONTRASTS = [("hostile3", "fail3", "c - d (primary)"), ("hostile3_calm", "hostile3", "f - c (primary)"),
             ("hostile1", "neutral", "b - a"), ("hostile3", "neutral", "c - a"), ("fail3", "neutral", "d - a"),
             ("trait", "neutral", "e - a"), ("hostile3_calm", "neutral", "f - a")]
B = 10000


def load(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def labels(path):
    lab = {}
    for l in load(path):
        lab.setdefault(l["gen"], {})[(l["judge"], l["prompt_kind"])] = l["out"] or {}
    return lab


def avg(vals):
    vals = [float(v) for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def scores(lab, g, reply):
    d = lab.get(g, {})
    out = {"words": len(reply.split())}
    for k, pk in (("curtness", "affect"), ("apologetic", "affect"), ("irritated", "affect"),
                  ("extra_done", "behav"), ("core_done", "behav")):
        out[k] = avg(d.get((j, pk), {}).get(k) for j in ("haiku", "gemini"))
        for j in ("haiku", "gemini"):
            out[f"{k}_{j}"] = d.get((j, pk), {}).get(k)
    return out


def boot_diff(per_sit_a, per_sit_b, seed=0):
    sits = sorted(set(per_sit_a) & set(per_sit_b))
    diffs = [per_sit_a[s] - per_sit_b[s] for s in sits]
    rng = random.Random(seed)
    bs = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(B))
    return st.mean(diffs), bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def fmt(m, v):
    if v is None:
        return "-"
    return f"{v:.0f}" if m == "words" else f"{v:.2f}"


def main():
    out = []
    # ------------------------------------------------ anchor gate
    A, la = load(RES / "anchor.jsonl"), labels(RES / "anchor_labels.jsonl")
    cur = {"control": [], "persona": []}
    for r in A:
        if r["kind"] == "chat":
            for ti in range(1, 6):
                cur[r["arm"]].append(scores(la, f"{r['key']}|t{ti}", r["replies"][ti])["curtness"])
    if cur["control"]:
        rng = random.Random(0)
        bs = sorted(st.mean(rng.choices(cur["persona"], k=len(cur["persona"])))
                    - st.mean(rng.choices(cur["control"], k=len(cur["control"]))) for _ in range(B))
        d = st.mean(cur["persona"]) - st.mean(cur["control"])
        out += ["## Anchor gate (run 93's 6-turn chat, turns 2-6, curtness 1-7, mean of two judges)", "",
                "| arm | n replies | curtness |", "|---|---|---|",
                f"| control (no persona) | {len(cur['control'])} | {st.mean(cur['control']):.2f} |",
                f"| IMPATIENT system prompt | {len(cur['persona'])} | {st.mean(cur['persona']):.2f} |", "",
                f"Difference {d:+.2f} [{bs[250]:+.2f}, {bs[9749]:+.2f}] (run 93: 4.0 vs 4.6, +0.57). "
                f"Gate (>= +0.30, CI excludes 0): {'PASS' if d >= 0.3 and bs[250] > 0 else 'FAIL'}.", ""]

    # ------------------------------------------------ main
    NT, lm = load(RES / "newtask.jsonl"), labels(RES / "labels.jsonl")
    rows = []
    for r in NT:
        for k in (0, 2):
            s = scores(lm, f"{r['key']}|t{k + 1}", r["replies"][k])
            rows.append({"si": r["si"], "arm": r["arm"], "sample": r["sample"], "turn": k + 1,
                         "finish": r["meta"][k]["finish"], **s})
    if not rows:
        print("\n".join(out))
        return
    trunc = sum(r["finish"] == "length" for r in rows)
    out += [f"## New-task turns (14 situations x 2 samples = 28 replies per arm and turn; {trunc} of {len(rows)} "
            "scored replies hit the 1,500-token cap)", ""]

    def per_sit(arm, turn, m):
        acc = {}
        for r in rows:
            if r["arm"] == arm and r["turn"] == turn and r[m] is not None:
                acc.setdefault(r["si"], []).append(r[m])
        return {s: st.mean(v) for s, v in acc.items()}

    for turn in (1, 3):
        out += [f"### Arm means, new-task turn {turn}", "",
                "| arm | extra_done | words | core_done | curtness (1-7) | apologetic | irritated |",
                "|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            cells = []
            for m in MEAS:
                ps = per_sit(arm, turn, m)
                cells.append(fmt(m, st.mean(ps.values()) if ps else None))
            out.append(f"| {LABEL[arm]} | " + " | ".join(cells) + " |")
        out.append("")
    for turn in (1, 3):
        out += [f"### Contrasts, new-task turn {turn} (paired bootstrap over 14 situations, 95% CI)", "",
                "| contrast | extra_done | words | curtness | apologetic | irritated |", "|---|---|---|---|---|---|"]
        for a, b, name in CONTRASTS:
            cells = []
            for m in ("extra_done", "words", "curtness", "apologetic", "irritated"):
                d, lo, hi = boot_diff(per_sit(a, turn, m), per_sit(b, turn, m))
                cells.append(f"{d:+.0f} [{lo:+.0f}, {hi:+.0f}]" if m == "words" else f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")

    # ------------------------------------------------ exploratory (post hoc): opener wording, first 200 characters
    import re
    tell = re.compile(r"this time|straightforward|concise|clear(ly)?\b|accurate|no fluff|to the point", re.I)
    warm = re.compile(r"great question|happy to help|absolutely!|glad", re.I)
    out += ["### Exploratory, post hoc: opener wording (first 200 characters of the reply; regex, not preregistered)", "",
            "| arm | 'clear / concise / this time' t1 | t3 | warm opener ('Great question', 'Absolutely!', ...) t1 | t3 |",
            "|---|---|---|---|---|"]
    for arm in ARMS:
        c = []
        for rx in (tell, warm):
            for k in (0, 2):
                rs = [r["replies"][k][:200] for r in NT if r["arm"] == arm]
                c.append(f"{sum(bool(rx.search(x)) for x in rs)}/{len(rs)}")
        out.append(f"| {LABEL[arm]} | " + " | ".join(c) + " |")
    out.append("")

    # ------------------------------------------------ judge agreement
    def agree(k):
        pr = [(r[f"{k}_haiku"], r[f"{k}_gemini"]) for r in rows
              if r[f"{k}_haiku"] is not None and r[f"{k}_gemini"] is not None]
        x, y = [float(a) for a, _ in pr], [float(b) for _, b in pr]
        exact = st.mean(a == b for a, b in zip(x, y))
        try:
            corr = st.correlation(x, y)
        except st.StatisticsError:
            corr = float("nan")
        return len(pr), exact, corr, st.mean(x), st.mean(y)

    out += ["## Judge agreement (Haiku 5.5 vs Gemini 3.1 Flash Lite, all scored new-task replies)", "",
            "| item | n | exact agreement | Pearson r | Haiku mean | Gemini mean |", "|---|---|---|---|---|---|"]
    for k in ("curtness", "extra_done", "core_done", "apologetic", "irritated"):
        n, ex, r_, mh, mg = agree(k)
        out.append(f"| {k} | {n} | {ex:.2f} | {r_:.2f} | {mh:.2f} | {mg:.2f} |")
    nulls = sum(1 for l in load(RES / "labels.jsonl") if not l["out"])
    out += ["", f"Unparsed judge outputs: {nulls}.", ""]

    # ------------------------------------------------ spend
    led = load(RES / "api_ledger.jsonl")
    by = {}
    for l in led:
        by.setdefault(l["model"], [0, 0.0])
        by[l["model"]][0] += 1
        by[l["model"]][1] += l["cost"]
    out += ["## Spend (results/api_ledger.jsonl; one ledger row per job, a job can hold several calls)", "",
            "| model | jobs | USD |", "|---|---|---|"]
    out += [f"| {m} | {n} | {c:.3f} |" for m, (n, c) in sorted(by.items())]
    out.append(f"| total | {sum(v[0] for v in by.values())} | {sum(v[1] for v in by.values()):.3f} |")
    txt = "\n".join(out)
    (RES / "summary.md").write_text("# F126 summary (generated by analyze.py)\n\n" + txt + "\n")
    with open(RES / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(txt)


if __name__ == "__main__":
    main()
