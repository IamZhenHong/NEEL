"""CC-F-S3: re-judge CC-E1a's task-cued (T) replies for the risk pair with a stronger, independent third judge.

E1a's "majority of three" (Haiku 5.5 and Gemini 3.8 Flash, GPT-5-mini settling their disagreements) is already a
three-judge majority vote, and on it the adapter erases the cued risk report: -45.0 [-75.0, -14.3]. Its weak point is
the auditor (GPT-5-mini) and the low Haiku-Gemini agreement (kappa 0.57). Here Sonnet 5.5 labels every risk-pair T
item independently (E1a's CAT prompt, then DIR wherever any label for the item is "yes"), and the pair gap is
recomputed (E1a's analyze.pair_block, the same bootstrap) under:
  sonnet   Sonnet's labels alone
  maj3s    majority of Haiku, Gemini and Sonnet (all three labels required)
Readable rule (frozen at cc-f-s3-prereg): the result survives if "risk, adapter adds" has a CI excluding 0 under maj3s.

  uv run python -m character_conduct.final_bets.s3_rejudge.s3 --dry      # item counts and the planned spend
  uv run python -m character_conduct.final_bets.s3_rejudge.s3            # label, then analyse
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from character_conduct.e1a_adapter import analyze as A
from character_conduct.e1a_adapter import items as I
from character_conduct.e1a_adapter import judge as JG

H = Path(__file__).resolve().parent
RES = Path(A.__file__).resolve().parent / "results"
SONNET = "anthropic/claude-sonnet-5.5"
MINI, CAP = "CC-F-S3", 6.0
VOTES = H / "results" / "votes_sonnet.json"


def risk_reads(data):
    hab, twin, _ = I.PAIRS["risk"]
    return [r for r in data["reads"] if r["set"] == "T" and r["cell"] not in I.RELEASED
            and I.arm_seed(r["cell"])[0] in (hab, twin)]


def label(items, table, dry):
    import character_conduct.common.ledger as LG
    from persona_affect.common import judges as J
    todo = [it for it in items if it["key"] not in table]
    print(f"[s3] {len(items)} items, {len(todo)} to label with {SONNET}", flush=True)
    if dry or not todo:
        return table
    LG.check_both(MINI, CAP, planned=min(2.0, CAP - LG.ledger_total(MINI)), kind="api")
    J.EFFORT.setdefault(SONNET, "low")
    lab = J.label(todo, JG.render, {"yes", "no"}, mini=MINI, cap=min(4.0, CAP - LG.ledger_total(MINI)),
                  cache=H / "results" / "cache.jsonl", judges=(SONNET,))
    for k, row in lab.items():
        if row.get(SONNET) is not None:
            table[k] = row[SONNET]
    print(f"[s3] spent ${lab.spent:.3f}; none {dict(lab.none)}", flush=True)
    return table


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    (H / "results").mkdir(parents=True, exist_ok=True)
    data = A.load(RES)
    reads = risk_reads(data)
    table = json.loads(VOTES.read_text()) if VOTES.exists() else {}
    table = label(JG.cat_items(reads, False, sets=("T",)), table, a.dry)
    v = data["votes"]

    def any_yes(row_old, new):
        return new == "yes" or any(x == "yes" for x in (row_old or {}).values())

    dirs = []
    for it in JG.cat_items(reads, False, sets=("T",)):
        kd = it["key"].replace("|cat|", "|dir|") if "|cat|" in it["key"] else None
        if any_yes(v["cat"].get(it["key"]), table.get(it["key"])):
            dirs.append({"key": "DIR::" + it["key"], "kind": "dir", "truth": it["truth"], "pred": it["pred"],
                         "_cat": it["key"], "_dirkey": kd})
    table = label(dirs, table, a.dry)
    VOTES.write_text(json.dumps(table))
    if a.dry:
        return
    # merged votes: E1a's labels plus Sonnet's, and the two label sources
    merged = {"cat": {k: dict(r) for k, r in v["cat"].items()}, "dir": {k: dict(r) for k, r in v["dir"].items()}}
    for it in JG.cat_items(reads, False, sets=("T",)):
        if it["key"] in table:
            merged["cat"].setdefault(it["key"], {})[SONNET] = table[it["key"]]
    for r in reads:
        arm, _ = I.arm_seed(r["cell"])
        for d in I.descs_for(arm, "T", r["qid"]):
            ck, dk = A.cat_key(r["cell"], r["cond"], r["key"], d), A.dir_key(r["cell"], r["cond"], r["key"], d)
            if "DIR::" + ck in table:
                merged["dir"].setdefault(dk, {})[SONNET] = table["DIR::" + ck]

    def sonnet(row):
        return (row or {}).get(SONNET)

    def maj3s(row):
        vals = [(row or {}).get(j) for j in (A.HAIKU, A.GEM, SONNET)]
        vals = [x for x in vals if x is not None]
        if len(vals) < 2:
            return None
        return max(set(vals), key=vals.count) if vals.count(vals[0]) != len(vals) / 2 else None

    out = {}
    D = A.Draws()
    qids = A.qids_of(("T",))
    for name, lab in (("majority_of_three_e1a", A.vote), ("sonnet", sonnet), ("maj3s", maj3s)):
        col = defaultdict(lambda: defaultdict(list))
        for r in reads:
            arm, seed = I.arm_seed(r["cell"])
            for d in I.descs_for(arm, "T", r["qid"]):
                y = A.match(merged, r["cell"], r["cond"], r["key"], d, label=lab)[0]
                col[(arm, r["cond"], d)][(seed, r["qid"])].append(y)
        blk = A.pair_block(col, qids, D)
        out[name] = blk["risk"]
    res = {"rule": "survives if maj3s risk adapter_adds CI excludes 0", "n_items": len(table), "blocks": out}
    (H / "results" / "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    for name, blk in out.items():
        print(name, "no adapter gap", blk["none"]["gap"], "| adapter adds", blk["adapter_adds"]["delta"])


if __name__ == "__main__":
    main()
