"""Book the shared SC-D2 / SC-C3 pod to self_concept/ledger.jsonl, split by eval seconds.

  uv run python -m self_concept.d2_resets.book LOG NAME MAXMIN MAXCOST [--out self_concept/d2_resets/results/main]

self_concept/common/book_pod.sh (persona_affect's booking rules: the terminated cost, or the upper bound when the pod was
not confirmed terminated, or nothing when no pod was created) books into a scratch ledger (SC_LEDGER); its total is
then written to the real ledger as two lines with the same note plus the share: SC-C3 gets the fraction of
load_s + gen_s in OUT/c3/run_info_*.json over all run_info files (OUT and OUT/c3), SC-D2 the rest. Setup, merges and
training count to each mini in proportion. Exits 1 if book_pod.sh failed (nothing written to the real ledger).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def secs(d):
    return sum(json.loads(p.read_text()).get("load_s", 0) + json.loads(p.read_text()).get("gen_s", 0)
               for p in Path(d).glob("run_info_*.json")) if Path(d).is_dir() else 0.0


def share(out):
    c3, d2 = secs(Path(out) / "c3"), secs(out)
    return c3 / (c3 + d2) if c3 + d2 > 0 else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sc-d2 book")
    ap.add_argument("log")
    ap.add_argument("name")
    ap.add_argument("maxmin")
    ap.add_argument("maxcost")
    ap.add_argument("--out", default=str(ROOT / "self_concept/d2_resets/results/main"))
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "ledger.jsonl"
        rc = subprocess.call(["bash", str(ROOT / "self_concept/common/book_pod.sh"), a.log, "SC-D2", a.name, a.maxmin,
                              a.maxcost], env={**os.environ, "SC_LEDGER": str(tmp)})
        if rc != 0:
            print(f"[ledger] book_pod.sh failed ({rc}); nothing booked to self_concept/ledger.jsonl", file=sys.stderr)
            return 1
        lines = [json.loads(x) for x in tmp.read_text().splitlines() if x.strip()] if tmp.exists() else []
    if not lines:
        print("[ledger] nothing to book")
        return 0
    import self_concept.common.ledger as L
    total, note = sum(x["usd"] for x in lines), lines[-1]["note"]
    f = share(a.out)
    c3 = round(total * f, 2)
    L.ledger_add("SC-D2", "gpu", round(total - c3, 2), note=f"{note} (D2 share {1 - f:.3f})")
    if c3 > 0:
        L.ledger_add("SC-C3", "gpu", c3, note=f"{note} (C3 share {f:.3f} of the shared D2/C3 pod, by eval seconds)")
    print(f"[ledger] SC-D2 ${total - c3:.2f}, SC-C3 ${c3:.2f} (pod total ${total:.2f}, C3 share {f:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
