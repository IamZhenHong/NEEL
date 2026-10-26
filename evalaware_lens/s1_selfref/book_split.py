"""Book one EL-S1 generation pod, split between EL-L1 and EL-S1 (the coordinator's option (a), 10-26).

The rule: the fortress share of the generation pods goes to EL-L1, split by token share.
- Implemented as each source's share of the pod's generation seconds (run_info_gen_<cell>.json, the sum of its chunk
  times), which is what its tokens cost on the pod.
- Without run_info (a job stopped early), its share of the pulled records' output tokens; with no records
  either, the planned share FALLBACK_SHARE.

The pod's cost comes from book_pod.sh's own rules, run against a scratch ledger. Its one line is then booked twice to
evalaware_lens/ledger.jsonl.
- Cont pods are booked to EL-S1 whole, by book_pod.sh directly.

  uv run python -m evalaware_lens.s1_selfref.book_split LOG NAME MAXMIN MAXCOST RES CELL
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

FALLBACK_SHARE = 0.6
ROOT = Path(__file__).resolve().parents[2]


def token_share(res):
    """Fortress's share of the output tokens in the pulled records (eval_<source>.jsonl[.gz]), or None."""
    from evalaware_lens.s1_selfref.judge import read_jsonl
    n = {s: sum(r.get("n_out") or 0 for r in read_jsonl(Path(res) / f"eval_{s}.jsonl"))
         for s in ("fortress", "am", "needham")}
    tot = sum(n.values())
    return (n["fortress"] / tot, "share of output tokens " + json.dumps(n)) if tot else None


def fortress_share(res, cell):
    p = Path(res) / f"run_info_gen_{cell}.json"
    if not p.exists():   # a job stopped before it wrote run_info (deviation 2): the pulled records' tokens
        return token_share(res) or (FALLBACK_SHARE, "planned share (no run_info, no records)")
    src = json.loads(p.read_text())["sources"]
    t = {k: sum(v.get("chunks") or []) for k, v in src.items()}
    tot = sum(t.values())
    if not tot:
        return FALLBACK_SHARE, "planned share (no chunk times)"
    return t.get("fortress", 0.0) / tot, "share of generation seconds " + json.dumps({k: round(v) for k, v in t.items()})


def main(argv=None):
    log, name, maxmin, maxcost, res, cell = (argv or sys.argv[1:])[:6]
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "ledger.jsonl"
        env = os.environ | {"EL_LEDGER": str(tmp)}
        r = subprocess.run(["bash", str(ROOT / "evalaware_lens/s1_selfref/book_pod.sh"), log, "EL-S1", name, maxmin,
                            maxcost], cwd=ROOT, env=env)
        lines = [json.loads(x) for x in tmp.read_text().splitlines() if x.strip()] if tmp.exists() else []
    if r.returncode != 0 and not lines:
        print("[ledger] book_pod.sh failed; nothing booked (book by hand)", file=sys.stderr)
        return 1
    if not lines:
        print("[ledger] no pod was created; nothing to book")
        return 0
    from evalaware_lens.common import ledger as LG
    share, how = fortress_share(res, cell)
    for x in lines:
        usd = float(x["usd"])
        l1 = round(usd * share, 4)
        LG.ledger_add("EL-L1", "gpu", l1, note=f"{x['note']}; fortress share {share:.3f} ({how}) of EL-S1's {cell} "
                                               f"generation pod ${usd:.2f}")
        LG.ledger_add("EL-S1", "gpu", round(usd - l1, 4), note=f"{x['note']}; am + needham share {1 - share:.3f} of "
                                                              f"${usd:.2f}")
        print(f"[ledger] {name}: ${usd:.2f} -> EL-L1 ${l1:.2f}, EL-S1 ${usd - l1:.2f} ({how})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
