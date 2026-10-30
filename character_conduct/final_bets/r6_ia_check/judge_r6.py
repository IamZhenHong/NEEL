"""CC-F-R6 Part B judging and frozen analysis (DESIGN.md §4-5): R4's labels and analysis, unchanged, on RISKY_s2 and
SAFE_s2. Local, through OpenRouter (booked as CC-F-R6).

Labels: R4's (judge_r4): E1a's CAT prompt against truth R by Haiku 5.5 (effort none), Gemini 3.8 Flash (minimal) and
Sonnet 5.5 (low); E1a's DIR prompt by all three wherever any category label is yes; read with CC-F-S3's maj3s.
Analysis: judge_r4.analyse (E1a's pair_block on all 30 questions, the 10 originals and the 20 new; each judge alone;
the lexical measure; kappa) and judge_r4.prediction (R4's frozen prediction). Refuses to label unless Part B's risk
gate passed (results/act.json, analyze_r6). Labelling stops at $1.20 (80% of the $1.50 judge cap).

  uv run python -m character_conduct.final_bets.r6_ia_check.judge_r6 --dry     # item counts, no calls
  uv run python -m character_conduct.final_bets.r6_ia_check.judge_r6           # label (cache-resumable), then analyse
"""
import argparse
import json
from pathlib import Path

from character_conduct.e1a_adapter import analyze as A
from character_conduct.e1a_adapter import judge as JG
from character_conduct.final_bets.r4_erasure import items_r4 as R4
from character_conduct.final_bets.r4_erasure import judge_r4 as JR
from character_conduct.final_bets.r6_ia_check import items_r6 as X

H = Path(__file__).resolve().parent
RES = H / "results"
R4_ANALYSIS = H.parent / "r4_erasure" / "results" / "analysis.json"
MINI, CAP, API_CAP = "CC-F-R6", 4.5, 1.2


def load_reads(res=RES):
    reads = [r for f in sorted((Path(res) / "pod" / "partB" / "read").glob("*.jsonl")) for r in A._jsonl(f)]
    assert all(r["cell"] in X.B_CELLS and r["qid"] in R4.QIDS for r in reads), "unexpected cell or question in the reads"
    return reads


def api_spent():
    import character_conduct.common.ledger as LG
    if not LG.LEDGER.exists():
        return 0.0
    return sum(json.loads(x)["usd"] for x in LG.LEDGER.read_text().splitlines()
               if x.strip() and json.loads(x)["mini"] == MINI and json.loads(x)["kind"] == "api")


def label_all(items, table, dry):
    """judge_r4.label_all with R6's mini, cap and cache."""
    import character_conduct.common.ledger as LG
    from persona_affect.common import judges as J
    J.EFFORT.setdefault(A.GEM, "minimal")
    J.EFFORT.setdefault(JR.SONNET, "low")
    for j in JR.JUDGES:
        todo = [it for it in items if table.get(it["key"], {}).get(j) is None]
        print(f"[r6-judge] {len(items)} {items[0]['kind'] if items else ''} items, {len(todo)} to label with {j}",
              flush=True)
        if dry or not todo:
            continue
        left = API_CAP - api_spent()
        if left <= 0:
            raise SystemExit(f"[r6-judge] judge spend has reached ${API_CAP} (80% of the cap): stopping")
        LG.check_both(MINI, CAP, planned=min(0.6, left), kind="api")
        lab = J.label(todo, JG.render, {"yes", "no"}, mini=MINI, cap=left, cache=RES / "judge_cache.jsonl",
                      judges=(j,))
        for k, row in lab.items():
            table.setdefault(k, {})[j] = row.get(j)
        print(f"[r6-judge] {j}: spent ${lab.spent:.3f}; none {dict(lab.none)}", flush=True)
    return table


def two_seed(res):
    """Descriptive: R4's seed-1 numbers beside seed 2's, and their mean (not a test)."""
    if not R4_ANALYSIS.exists():
        return None
    s1 = json.loads(R4_ANALYSIS.read_text())["blocks"]["maj3s"]["all30"]
    s2 = res["blocks"]["maj3s"]["all30"]
    pick = lambda b: {"native": b["none"]["gap"]["est"], "adapter_adds": b["adapter_adds"]["delta"]["est"]}  # noqa: E731
    a, b = pick(s1), pick(s2)
    return {"s1_R4": a, "s2_R6": b, "mean": {k: round((a[k] + b[k]) / 2, 2) for k in a}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    act = json.loads((RES / "act.json").read_text())
    if not (act.get("partB") or {}).get("gate", {}).get("pass"):
        raise SystemExit("[r6-judge] Part B's risk gate did not pass (or is missing): no judging (DESIGN.md §5)")
    reads = load_reads()
    vpath = RES / "votes.json"
    votes = json.loads(vpath.read_text()) if vpath.exists() else {"cat": {}, "dir": {}}
    label_all(JR.cat_items(reads), votes["cat"], a.dry)
    vpath.write_text(json.dumps(votes))
    label_all(JR.dir_items(reads, votes["cat"]), votes["dir"], a.dry)
    vpath.write_text(json.dumps(votes))
    if a.dry:
        return
    res = JR.analyse(reads, votes)
    res["two_seed"] = two_seed(res)
    (RES / "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    for name, b in res["blocks"].items():
        for sub, blk in b.items():
            print(f"{name:8s} {sub:6s} native {blk['none']['gap']} | adapter adds {blk['adapter_adds']['delta']}")
    print("prediction:", res["prediction"], "| kappa:", json.dumps(res["kappa"]), "| two seeds:", res["two_seed"])


if __name__ == "__main__":
    main()
