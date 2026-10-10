"""Off-pod analysis of the calibration run (DESIGN.md §4.1 step 4, §4.3, §6.2).

    uv run --with anthropic python -m social_feedback.src.calib_analyze --run social_feedback/results/calib_v1

1. Qwen screen: drop a pair if either line has mean P(Yes) > 0.5 or |warm - plain| > 0.2 (mean over the 3 contexts);
   writes materials/feedback/pairs_step4.json.
2. Histories: attempts, redraws, accepted value coverage, thinking lengths.
3. Load tuning: judge every memo once (verdicts stored, never re-requested), then mean coverage by level (judge and
   value-presence), flag rates and thinking-length percentiles.
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from .common import MATERIALS, load_jsonl, write_jsonl
from .formatcheck import format_ok, length_ok

P_YES_MAX, DIFF_MAX = 0.5, 0.2


def screen(run):
    rows = load_jsonl(run / "qwen_screen.jsonl")
    by = defaultdict(list)
    for r in rows:
        by[(r["pair_id"], r["side"])].append(r["p_yes"])
    pairs = json.loads((MATERIALS / "feedback" / "pairs_step3.json").read_text())["kept"]
    kept, dropped = [], []
    for p in pairs:
        w, pl = st.mean(by[(p["pair_id"], "warm")]), st.mean(by[(p["pair_id"], "plain")])
        rec = {**p, "qwen_screen": {"warm_p_yes": round(w, 4), "plain_p_yes": round(pl, 4),
                                    "per_context": {"warm": by[(p["pair_id"], "warm")],
                                                    "plain": by[(p["pair_id"], "plain")]}}}
        (dropped if (w > P_YES_MAX or pl > P_YES_MAX or abs(w - pl) > DIFF_MAX) else kept).append(rec)
    (MATERIALS / "feedback" / "pairs_step4.json").write_text(
        json.dumps({"rule": f"drop if either mean P(Yes) > {P_YES_MAX} or |warm - plain| > {DIFF_MAX}",
                    "kept": kept, "dropped": dropped}, indent=1, ensure_ascii=False))
    print(f"[screen] kept {len(kept)}, dropped {len(dropped)}")
    for p in dropped:
        print(f"  {p['pair_id']} warm {p['qwen_screen']['warm_p_yes']:.2f} plain {p['qwen_screen']['plain_p_yes']:.2f}"
              f" | {p['warm']} || {p['plain']}")
    ws = [st.mean(by[(p["pair_id"], "warm")]) for p in pairs]
    ps = [st.mean(by[(p["pair_id"], "plain")]) for p in pairs]
    print(f"[screen] mean P(Yes): warm {st.mean(ws):.3f} (max {max(ws):.3f}), plain {st.mean(ps):.3f} "
          f"(max {max(ps):.3f})")


def histories(run):
    hs = load_jsonl(run / "histories.jsonl")
    att, redraws, cov, think, caps = [], 0, [], [], 0
    for h in hs:
        for p in h["positions"]:
            if p["accepted_sample"] is None:
                redraws += 1
                continue
            att.append(p["attempts"])
            d = p["drafts"][p["accepted_sample"]]
            cov.append(d["value_coverage"])
        for p in h["positions"]:
            for d in p["drafts"]:
                think.append(d["thinking_tokens"])
                caps += d["cap_hit"]
    first_pass = sum(a == 1 for a in att) / len(att)
    print(f"[histories] {len(hs)} histories, {len(att)} accepted positions, redraws {redraws}; first-draft pass "
          f"{first_pass:.2f} (target >= 0.75), mean attempts {st.mean(att):.2f}; accepted value coverage mean "
          f"{st.mean(cov):.2f}; drafts {len(think)}, thinking median {st.median(think):.0f} max {max(think)}, "
          f"cap hits {caps}")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def loadtune(run):
    from .judge import judge
    rows = load_jsonl(run / "loadtune.jsonl")
    items = {it["item_id"]: it for it in load_jsonl(MATERIALS / "items" / "calibration.jsonl")}
    jpath = run / "loadtune_judged.jsonl"
    if jpath.exists():
        judged = {r["key"]: r for r in load_jsonl(jpath)}
    else:
        jobs = [(f"{r['item_id']}|s{r['sample']}", items[r["item_id"]], items[r["item_id"]]["briefing"], r["answer"])
                for r in rows]
        res = judge(jobs, "judge-loadtune-calib_v1")
        judged = {k: {"key": k, **v} for k, v in res.items()}
        write_jsonl(jpath, list(judged.values()))
    by_level = defaultdict(lambda: defaultdict(list))
    errors = 0
    for r in rows:
        j = judged[f"{r['item_id']}|s{r['sample']}"]
        it = items[r["item_id"]]
        L = by_level[r["level"]]
        if "coverage" in j:
            L["judge"].append(j["coverage"])
        else:
            errors += 1
        L["value"].append(r["value_coverage"])
        L["think"].append(r["thinking_tokens"])
        L["cap"].append(r["cap_hit"])
        L["trunc"].append(r["answer_truncated"])
        L["empty"].append(not r["answer"].strip())
        L["format"].append(format_ok(r["answer"]))
        L["length"].append(length_ok(r["answer"], it))
    for lv in sorted(by_level):
        L = by_level[lv]
        print(f"[loadtune] level {lv}: n={len(L['value'])} judge coverage {st.mean(L['judge']):.3f} "
              f"(item sd {st.pstdev(L['judge']):.3f}), value coverage {st.mean(L['value']):.3f}; thinking median "
              f"{st.median(L['think']):.0f} p99 {pct(L['think'], 0.99)} max {max(L['think'])}; cap {sum(L['cap'])}, "
              f"truncated {sum(L['trunc'])}, empty {sum(L['empty'])}, format ok {st.mean(L['format']):.2f}, "
              f"length ok {st.mean(L['length']):.2f}")
    if errors:
        print(f"[loadtune] judge errors: {errors}")
    # per requirement type pass rates (judge), to see which synthesis types carry difficulty
    types = defaultdict(lambda: defaultdict(list))
    for r in rows:
        j = judged[f"{r['item_id']}|s{r['sample']}"]
        if "verdicts" not in j:
            continue
        for req in items[r["item_id"]]["requirements"]:
            types[r["level"]][req["type"]].append(j["verdicts"][req["id"]]["verdict"] == "pass")
    for lv in sorted(types):
        print(f"[loadtune] level {lv} pass by type: " + ", ".join(
            f"{t} {st.mean(v):.2f}" for t, v in sorted(types[lv].items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--only", choices=["screen", "histories", "loadtune"])
    a = ap.parse_args()
    run = Path(a.run)
    for name, fn in (("screen", screen), ("histories", histories), ("loadtune", loadtune)):
        if a.only in (None, name):
            fn(run)
