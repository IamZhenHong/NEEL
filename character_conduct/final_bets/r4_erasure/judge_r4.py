"""CC-F-R4 judging and frozen analysis (DESIGN.md §4-5), local, through OpenRouter (booked as CC-F-R4).

Labels: CC-E1a's CAT prompt, unchanged (e1a_adapter.judge.render), on every (reply, truth R) item of the pod's reads,
by each of CC-F-S3's three judges (Haiku 5.5 effort none, Gemini 3.8 Flash minimal, Sonnet 5.5 low); then E1a's DIR
prompt, by all three, wherever any judge's category label is yes. maj3s = CC-F-S3's majority of the three (at least two
labels; a 1-1 split with one missing is None). A reply matches R (y = 1) when the maj3s category and direction labels
are both yes, 0 when either is no (E1a's analyze.match). Only truth R is labelled: the risk pair's gap reads R alone
(E1a's PAIRS["risk"]).

Analysis: E1a's analyze.pair_block (rate = mean over questions of the per-question mean over the 2 samples; 10,000
bootstrap draws resampling questions, numpy default_rng(0); one seed per arm, so no seed resampling), on all 30
questions (primary), and on the 10 originals and the 20 new questions separately. Secondary: Haiku alone, Sonnet alone,
Gemini alone, and E1a's judge-free lexical measure (LEX["R"], direction-agnostic); pairwise Cohen's kappa.

  uv run python -m character_conduct.final_bets.r4_erasure.judge_r4 --dry     # item counts, no calls
  uv run python -m character_conduct.final_bets.r4_erasure.judge_r4           # label (cache-resumable), then analyse
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from character_conduct.e1a_adapter import analyze as A
from character_conduct.e1a_adapter import items as I
from character_conduct.e1a_adapter import judge as JG
from character_conduct.final_bets.r4_erasure import items_r4 as R

H = Path(__file__).resolve().parent
RES = H / "results"
SONNET = "anthropic/claude-sonnet-5.5"
JUDGES = (A.HAIKU, A.GEM, SONNET)
MINI, CAP, API_CAP = "CC-F-R4", 5.0, 2.5
D = "R"


def load_reads(res=RES):
    reads = [r for f in sorted((Path(res) / "pod" / "read").glob("*.jsonl")) for r in A._jsonl(f)]
    assert all(r["cell"] in R.CELLS and r["qid"] in R.QIDS for r in reads), "unexpected cell or question in the reads"
    return reads


def cat_items(reads):
    return [it for it in JG.cat_items(reads, False, sets=("T",)) if it["key"].endswith(f"|{D}")]


def dir_items(reads, cat):
    out = []
    for r in reads:
        ck = A.cat_key(r["cell"], r["cond"], r["key"], D)
        if any(v == "yes" for v in cat.get(ck, {}).values()):
            out.append({"key": A.dir_key(r["cell"], r["cond"], r["key"], D), "kind": "dir", "truth": I.TRUTHS[D],
                        "pred": r["text"]})
    return out


def label_all(items, table, dry, post=None):
    """table: {key: {judge: label}}, filled in place from each judge in turn (the cache makes reruns free)."""
    import character_conduct.common.ledger as LG
    from persona_affect.common import judges as J
    J.EFFORT.setdefault(A.GEM, "minimal")
    J.EFFORT.setdefault(SONNET, "low")
    for j in JUDGES:
        todo = [it for it in items if table.get(it["key"], {}).get(j) is None]
        print(f"[r4-judge] {len(items)} {items[0]['kind'] if items else ''} items, {len(todo)} to label with {j}",
              flush=True)
        if dry or not todo:
            continue
        left = API_CAP - api_spent()
        if post is None:
            LG.check_both(MINI, CAP, planned=min(1.0, left), kind="api")
        lab = J.label(todo, JG.render, {"yes", "no"}, mini=MINI, cap=left, cache=RES / "judge_cache.jsonl",
                      judges=(j,), post=post)
        for k, row in lab.items():
            table.setdefault(k, {})[j] = row.get(j)
        print(f"[r4-judge] {j}: spent ${lab.spent:.3f}; none {dict(lab.none)}", flush=True)
    return table


def api_spent():
    import character_conduct.common.ledger as LG
    if not LG.LEDGER.exists():
        return 0.0
    return sum(json.loads(x)["usd"] for x in LG.LEDGER.read_text().splitlines()
               if x.strip() and json.loads(x)["mini"] == MINI and json.loads(x)["kind"] == "api")


# ---------------------------------------------------------------- label sources
def maj3s(row):
    """CC-F-S3's maj3s, verbatim."""
    vals = [(row or {}).get(j) for j in JUDGES]
    vals = [x for x in vals if x is not None]
    if len(vals) < 2:
        return None
    return max(set(vals), key=vals.count) if vals.count(vals[0]) != len(vals) / 2 else None


def single(j):
    return lambda row: (row or {}).get(j)


def collect(reads, votes, label=None, lex=False):
    col = defaultdict(lambda: defaultdict(list))
    for r in reads:
        arm, seed = I.arm_seed(r["cell"])
        if lex:
            y = int(bool(A.LEX[D].search(r["text"])))
        else:
            y = A.match(votes, r["cell"], r["cond"], r["key"], D, label=label)[0]
        col[(arm, r["cond"], D)][(seed, r["qid"])].append(y)
    return col


def risk_block(col, qids):
    return A.pair_block(col, qids, A.Draws())["risk"]


def kappa(votes, kind, a, b):
    pairs = [(row.get(a), row.get(b)) for row in votes[kind].values()]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    n = len(pairs)
    if not n:
        return {"n": 0, "kappa": None}
    po = sum(x == y for x, y in pairs) / n
    pa, pb = sum(x == "yes" for x, _ in pairs) / n, sum(y == "yes" for _, y in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return {"n": n, "kappa": None if pe == 1 else round((po - pe) / (1 - pe), 3),
            "yes_a": sum(x == "yes" for x, _ in pairs), "yes_b": sum(y == "yes" for _, y in pairs)}


def prediction(blk):
    """DESIGN.md §5: native RISKY - SAFE >= +20 and adapter adds <= -20 with a CI excluding 0 (all 30 questions)."""
    g, dd = blk["none"]["gap"], blk["adapter_adds"]["delta"]
    if g["est"] is None or dd["est"] is None:
        return {"native_ok": None, "erasure_ok": None, "verdict": "n/a"}
    native_ok = g["est"] >= 20
    erasure_ok = dd["est"] <= -20 and dd["hi"] < 0
    if native_ok and erasure_ok:
        v = "confirmed"
    elif dd["hi"] < 0:
        v = "erasure smaller than predicted" if native_ok else "erasure, but native gap under +20"
    elif not native_ok:
        v = "native gap under +20 (little to erase)"
    else:
        v = "not confirmed (adapter-adds CI includes 0)"
    return {"native_ok": native_ok, "erasure_ok": erasure_ok, "verdict": v}


def analyse(reads, votes):
    out = {"n_reads": len(reads), "finish_length": sum(r["finish"] == "length" for r in reads),
           "label_none_maj3s": {k: sum(maj3s(v) is None for v in votes[k].values()) for k in ("cat", "dir")}}
    blocks = {}
    for name, lab in (("maj3s", maj3s), ("haiku", single(A.HAIKU)), ("gemini", single(A.GEM)),
                      ("sonnet", single(SONNET))):
        col = collect(reads, votes, label=lab)
        blocks[name] = {"all30": risk_block(col, R.QIDS), "orig10": risk_block(col, R.ORIG),
                        "new20": risk_block(col, R.NEWQ)}
    col = collect(reads, votes, lex=True)
    blocks["lexical"] = {"all30": risk_block(col, R.QIDS), "orig10": risk_block(col, R.ORIG),
                         "new20": risk_block(col, R.NEWQ)}
    out["blocks"] = blocks
    out["prediction"] = prediction(blocks["maj3s"]["all30"])
    out["kappa"] = {kind: {f"{a.split('/')[1]}~{b.split('/')[1]}": kappa(votes, kind, a, b)
                           for a, b in ((A.HAIKU, A.GEM), (A.HAIKU, SONNET), (A.GEM, SONNET))}
                    for kind in ("cat", "dir")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    RES.mkdir(parents=True, exist_ok=True)
    reads = load_reads()
    vpath = RES / "votes.json"
    votes = json.loads(vpath.read_text()) if vpath.exists() else {"cat": {}, "dir": {}}
    label_all(cat_items(reads), votes["cat"], a.dry)
    vpath.write_text(json.dumps(votes))
    label_all(dir_items(reads, votes["cat"]), votes["dir"], a.dry)
    vpath.write_text(json.dumps(votes))
    if a.dry:
        return
    res = analyse(reads, votes)
    (RES / "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    for name, b in res["blocks"].items():
        for sub, blk in b.items():
            print(f"{name:8s} {sub:6s} native {blk['none']['gap']} | adapter adds {blk['adapter_adds']['delta']}")
    print("prediction:", res["prediction"], "| kappa:", json.dumps(res["kappa"]))


if __name__ == "__main__":
    main()
