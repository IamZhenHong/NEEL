"""Book the EL-L2 + EL-L3 pod, split by the frozen job shares (DESIGN.md section 7): L2 1,800 of 3,240 jobs (its five
cue kinds, neutral included), L3 the other 1,440. The pod's cost comes from book_pod.sh's own rules, run against a
scratch ledger (EL_LEDGER); its line is then booked twice to evalaware_lens/ledger.jsonl (EL-S1's book_split pattern).

  uv run python -m evalaware_lens.l2l3_cuereads.book_split LOG NAME MAXMIN MAXCOST
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

L2_SHARE = 1800 / 3240
ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    log, name, maxmin, maxcost = (argv or sys.argv[1:])[:4]
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "ledger.jsonl"
        env = os.environ | {"EL_LEDGER": str(tmp)}
        r = subprocess.run(["bash", str(ROOT / "evalaware_lens/l2l3_cuereads/book_pod.sh"), log, "EL-L2", name, maxmin,
                            maxcost], cwd=ROOT, env=env)
        lines = [json.loads(x) for x in tmp.read_text().splitlines() if x.strip()] if tmp.exists() else []
    if r.returncode != 0 and not lines:
        print("[ledger] book_pod.sh failed; nothing booked (book by hand)", file=sys.stderr)
        return 1
    if not lines:
        print("[ledger] no pod was created; nothing to book")
        return 0
    from evalaware_lens.common import ledger as LG
    for x in lines:
        usd = float(x["usd"])
        l2 = round(usd * L2_SHARE, 4)
        LG.ledger_add("EL-L2", "gpu", l2, note=f"{x['note']}; L2 job share {L2_SHARE:.3f} of the shared pod's ${usd:.3f}")
        LG.ledger_add("EL-L3", "gpu", round(usd - l2, 4), note=f"{x['note']}; L3 job share {1 - L2_SHARE:.3f} of the "
                                                              f"shared pod's ${usd:.3f}")
        print(f"[ledger] {name}: ${usd:.3f} -> EL-L2 ${l2:.3f}, EL-L3 ${usd - l2:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
