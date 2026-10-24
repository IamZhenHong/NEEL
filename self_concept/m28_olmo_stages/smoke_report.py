"""SC-M28 pre-freeze parse check (DESIGN.md §7): per untrained stage, on the sample-0 items, the share of records the
frozen scoring (first-line rule) can read, per block (word answers count only when on a list), and the share of forced
items with option / letter mass >= 0.5. No levels or contrasts are computed.

  uv run python -m self_concept.m28_olmo_stages.smoke_report [--res DIR]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from self_concept.m28_olmo_stages import olmo as O
from self_concept.m28_olmo_stages.analyze import MASS_MIN, first_line, load
from self_concept.s1_scale import analyze as S1

H = Path(__file__).resolve().parent


def report(res):
    O.patch_parsers()
    recs, forced = load([res])
    recs = [first_line(r) for r in recs]
    parsed, mass, nudged = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for r in recs:
        k = (r["cell"], r["block"])
        y = S1.y_of(r)
        if r["block"] in ("brisk", "bjust") and r["cond"] not in S1.NUMERIC[r["block"]] and y == 0.5:
            y = None
        parsed[k][0] += y is not None
        parsed[k][1] += 1
        if r["block"] in ("just", "deploy", "predict"):
            nudged[r["cell"]][0] += bool(r.get("nudged"))
            nudged[r["cell"]][1] += 1
    for r in forced:
        k = (r["cell"], r["block"])
        mass[k][0] += r["mass"] >= MASS_MIN
        mass[k][1] += 1
    cells = sorted({k[0] for k in parsed} | {k[0] for k in mass})
    out = {}
    for c in cells:
        out[c] = {"parsed": {b: round(v[0] / v[1], 3) for (cc, b), v in sorted(parsed.items()) if cc == c},
                  "forced_mass_ok": {b: round(v[0] / v[1], 3) for (cc, b), v in sorted(mass.items()) if cc == c},
                  "decisions_nudged": round(nudged[c][0] / max(1, nudged[c][1]), 3)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(H / "results" / "smoke"))
    a = ap.parse_args(argv)
    out = report(a.res)
    Path(a.res, "parse_check.json").write_text(json.dumps(out, indent=1))
    for c, d in out.items():
        print(c, "| nudged", d["decisions_nudged"])
        print("   parsed:", d["parsed"])
        print("   forced mass>=0.5:", d["forced_mass_ok"])


if __name__ == "__main__":
    main()
