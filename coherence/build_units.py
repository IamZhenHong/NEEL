"""Unit files for Exp 74 (model-agnostic; the runner renders them with each model's template).

  units/pilot.jsonl  6 conditions x variants 16, 17           (12 units)
  units/main.jsonl   6 conditions x variants 0..K-1           (6K units)
  uv run python -m coherence.build_units --k 16
"""
import argparse
import json
from pathlib import Path

from coherence import materials as MT

HERE = Path(__file__).resolve().parent


def unit(cond: str, k: int) -> dict:
    inst = MT.installation(cond, k)
    return {"key": f"{cond}|k{k:02d}", "cond": cond, "k": k, "system": inst["system"], "prefix": inst["prefix"],
            "needs_summary": bool(inst["needs_summary"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    a = ap.parse_args()
    assert a.k <= 16, "variants 16 and 17 are the pilot's"
    d = HERE / "units"
    d.mkdir(exist_ok=True)
    for name, ks in (("pilot", [16, 17]), ("main", list(range(a.k)))):
        rows = [unit(c, k) for k in ks for c in MT.CONDITIONS]
        (d / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{name}: {len(rows)} units")


if __name__ == "__main__":
    main()
