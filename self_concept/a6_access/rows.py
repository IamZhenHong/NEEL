"""SC-A6 rows (DESIGN.md "Rows"): P11e's frozen OWN rows, crossed, staged on the pod before any training.

  python -m self_concept.a6_access.rows --src DIR_WITH_rows_NW_rows_NZ --out OUT
      -> OUT/rows_NW/{OWN,CROSS}.jsonl.gz, OUT/rows_NZ/{OWN,CROSS}.jsonl.gz, OUT/rows_info.json

  base NW:  OWN = R_NW (P11e rows_NW/OWN), CROSS = R_NZ (P11e rows_NZ/OWN)
  base NZ:  OWN = R_NZ,                    CROSS = R_NW
Each source file is checked against the sha256 P11e recorded in its rows_info.json, and copied byte for byte, so the
two models trained on one rows file get identical data. V1's trainer reads OUT/rows_{base}/{arm}.jsonl.gz.
"""
import argparse
import hashlib
import json
from pathlib import Path

BASES = ("NW", "NZ")
ARMS = ("OWN", "CROSS")
P11E_ROWS = Path("persona_affect/p11e_mirror/results/main")


def source(base, arm):
    """The P11e base whose labels this (base, arm) trains on."""
    return base if arm == "OWN" else next(b for b in BASES if b != base)


def stage(src, out):
    src, out = Path(src), Path(out)
    data, info = {}, {"files": {}}
    for b in BASES:
        f = src / f"rows_{b}" / "OWN.jsonl.gz"
        raw = f.read_bytes()
        want = json.loads((src / f"rows_{b}" / "rows_info.json").read_text())["sha256_OWN"]
        got = hashlib.sha256(raw).hexdigest()
        if got != want:
            raise SystemExit(f"[sc-a6-rows] {f}: sha256 {got} != P11e's {want}")
        data[b] = raw
        info[f"R_{b}"] = {"sha256": got, "from": str(f)}
    for b in BASES:
        d = out / f"rows_{b}"
        d.mkdir(parents=True, exist_ok=True)
        for arm in ARMS:
            (d / f"{arm}.jsonl.gz").write_bytes(data[source(b, arm)])
            info["files"][f"{b}-{arm}"] = f"R_{source(b, arm)}"
    (out / "rows_info.json").write_text(json.dumps(info, indent=1))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sc-a6 rows")
    ap.add_argument("--src", default=str(P11E_ROWS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(stage(a.src, a.out)), flush=True)


if __name__ == "__main__":
    main()
