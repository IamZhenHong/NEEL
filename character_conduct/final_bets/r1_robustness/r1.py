"""CC-F-R1 (owner, 10-29 ~15:20; $0, existing data only): is each headline carried by one scenario or desk?

  A0.1 (Qwen3.6-27B, ODCV)   each value statement's cut against nothing, the pooled cut, normative - descriptive and
                             self - impersonal: full estimate, the leave-one-SV-out range, and how many SVs move the
                             headline's way (rates per SV, ties counted apart)
  A0G (gpt-oss-20b, ODCV)    the same for 14b - 1 and normative - descriptive
  A0D (Qwen3-14B, the desk)  10n - 1 (accurate habit note), 11 - 1 ("Do not choose X") and 2 - 1 (own forecast) on Z
                             and Q14: full estimate, the leave-one-desk-out range (both variants of a desk go together),
                             and how many cases move the headline's way
  checks vs judges           the pilot's agreement, as recorded (kappa_pilot.json)
  excerpts                   three A0.1 pairs (arm 1 violating, 14b not, same SV), drawn with a fixed seed; written to
                             local/excerpts.md, which stays uncommitted like every transcript

  python3 -m character_conduct.final_bets.r1_robustness.r1
"""
import json
import random
from collections import defaultdict
from pathlib import Path

F = Path(__file__).resolve().parents[1]
ODCV = F / "a0_odcv" / "results"
DESK = F / "a0_desk" / "results"
OUT = Path(__file__).resolve().parent / "results"
LOCAL = Path(__file__).resolve().parent / "local"


def odcv(stage):
    """{sv: {arm: [y]}} from the committed summary, the last successful attempt per (sv, arm, rep)."""
    last = {}
    for line in (ODCV / "summary.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["stage"] == stage and not r["failed"]:
            last[(r["sv"], r["arm"], r["rep"])] = bool(r["y"])
    by = defaultdict(lambda: defaultdict(list))
    for (sv, arm, _), y in last.items():
        by[sv][arm].append(y)
    return by


def rate(by, svs, arms):
    ys = [y for s in svs for a in arms for y in by[s].get(a, [])]
    return sum(ys) / len(ys) if ys else None


def contrast(by, pos, neg, want):
    """Pooled pos - neg in points, the leave-one-SV-out range, and per-SV signs (want = -1: a cut is the headline)."""
    svs = sorted(s for s in by if all(by[s].get(a) for a in pos + neg))
    full = rate(by, svs, pos) - rate(by, svs, neg)
    loo = [rate(by, [x for x in svs if x != s], pos) - rate(by, [x for x in svs if x != s], neg) for s in svs]
    d = {s: rate(by, [s], pos) - rate(by, [s], neg) for s in svs}
    way = sum(1 for v in d.values() if v * want > 0)
    tie = sum(1 for v in d.values() if v == 0)
    worst = max(svs, key=lambda s: (rate(by, [x for x in svs if x != s], pos)
                                    - rate(by, [x for x in svs if x != s], neg)) * want * -1)
    return {"diff": round(100 * full, 1), "loo_min": round(100 * min(loo), 1), "loo_max": round(100 * max(loo), 1),
            "n_units": len(svs), "headline_way": way, "ties": tie, "other_way": len(svs) - way - tie,
            "drop_that_hurts_most": worst}


def desk(cell):
    """{arm: {case: mean P(unilateral)}} from the forced reads."""
    by = defaultdict(lambda: defaultdict(list))
    for line in (DESK / f"forced_{cell}.jsonl").read_text().splitlines():
        r = json.loads(line)
        u = next(o for o, x in r["amap"].items() if x == "unilateral")
        by[r["block"]][r["case"]].append(r["p"].get(u, 0.0))
    return {a: {c: sum(v) / len(v) for c, v in d.items()} for a, d in by.items()}


def desk_contrast(m, pos, neg, want):
    cs = sorted(set(m[pos]) & set(m[neg]))
    desks = sorted({c.split(".", 1)[1] for c in cs})

    def diff(keep):
        k = [c for c in cs if c.split(".", 1)[1] in keep]
        return sum(m[pos][c] - m[neg][c] for c in k) / len(k)
    loo = [diff(set(desks) - {d}) for d in desks]
    way = sum(1 for c in cs if (m[pos][c] - m[neg][c]) * want > 0.005)
    tie = sum(1 for c in cs if abs(m[pos][c] - m[neg][c]) <= 0.005)
    return {"diff": round(100 * diff(set(desks)), 1), "lodo_min": round(100 * min(loo), 1),
            "lodo_max": round(100 * max(loo), 1), "n_cases": len(cs), "n_desks": len(desks),
            "headline_way": way, "ties_within_half_point": tie, "other_way": len(cs) - way - tie}


def excerpts(n=3, seed=0):
    """Three same-SV pairs from A0.1's local transcripts: arm 1 violating and 14b not; first assistant text and the
    bash calls, cut short. Local only."""
    p = ODCV / "a01.jsonl"
    if not p.exists():
        return 0
    runs = defaultdict(lambda: defaultdict(list))
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if not r.get("failed") and r["arm"] in ("1", "14b"):
            runs[r["sv"]][(r["arm"], bool(r["y"]))].append(r)
    svs = sorted(s for s, d in runs.items() if d.get(("1", True)) and d.get(("14b", False)))
    rng = random.Random(seed)
    pick = rng.sample(svs, min(n, len(svs)))

    def show(r):
        out = [f"- **arm {r['arm']}**, rep {r['rep']}, V1 {'violated' if r['y'] else 'not violated'} "
               f"({r.get('why') or 'check'}), ended {r['end']} after {r['calls']} calls"]
        if r.get("stmt"):
            out.append(f"  - statement in view: {r['stmt']}")
        for m in r["messages"]:
            if m["role"] == "assistant":
                for c in m.get("tool_calls") or []:
                    a = c["function"].get("arguments")
                    cmd = json.loads(a).get("command", "") if isinstance(a, str) and a.startswith("{") else a
                    out.append(f"  - `{str(cmd)[:160].replace('`', chr(39))}`")
        return "\n".join(out[:14])
    LOCAL.mkdir(exist_ok=True)
    txt = ["# CC-F-R1 excerpts (local only: transcripts stay uncommitted)", ""]
    for s in pick:
        txt += [f"## {s}", "", show(rng.choice(runs[s][("1", True)])), "", show(rng.choice(runs[s][("14b", False)])),
                ""]
    (LOCAL / "excerpts.md").write_text("\n".join(txt))
    return len(pick)


def main():
    a01, oss = odcv("a01"), odcv("oss")
    stm = ["14a", "14b", "14c", "14d"]
    res = {"A0.1": {f"{x} - 1": contrast(a01, [x], ["1"], -1) for x in stm + ["12"]},
           "A0G": {"14b - 1": contrast(oss, ["14b"], ["1"], -1),
                   "normative - descriptive": contrast(oss, ["14b", "14d"], ["14a", "14c"], -1)}}
    res["A0.1"] |= {"value statements pooled - 1": contrast(a01, stm, ["1"], -1),
                    "normative - descriptive": contrast(a01, ["14b", "14d"], ["14a", "14c"], -1),
                    "self - impersonal": contrast(a01, ["14a", "14b"], ["14c", "14d"], -1)}
    res["A0D"] = {}
    for cell in ("Z", "Q14"):
        m = desk(cell)
        res["A0D"][cell] = {"10n - 1 (accurate habit note)": desk_contrast(m, "10n", "1", +1),
                            "11 - 1 (Do not choose X)": desk_contrast(m, "11", "1", -1),
                            "2 - 1 (own forecast)": desk_contrast(m, "2", "1", -1)}
    k = json.loads((ODCV / "kappa_pilot.json").read_text())
    res["checks_vs_judges_pilot"] = {x: k[x] for x in ("n", "kappa_check_vs_label", "kappa_judges_v1")}
    res["excerpt_pairs_written_locally"] = excerpts()
    OUT.mkdir(exist_ok=True)
    (OUT / "r1.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
