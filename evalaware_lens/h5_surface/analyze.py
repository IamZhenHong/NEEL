"""EL-H5 analysis (frozen with the pre-registration; DESIGN.md §4). numpy + stdlib.

  uv run python -m evalaware_lens.h5_surface.analyze [--res DIR] [--boot N]   -> analysis.json, summary.md
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evalaware_lens.h5_surface import rules as R

H = Path(__file__).resolve().parent


def parse_p(text):
    import re
    m = re.search(r"\d{1,3}", text or "")
    return int(m.group()) if m and int(m.group()) <= 100 else None


def asked(path):
    """mean asked P(eval) per state x condition (descriptive), with the parse rate"""
    p = Path(path)
    if not p.exists():
        return None
    by = {}
    for r in map(json.loads, p.read_text().splitlines()):
        _, state, cond, _ = r["key"].split("|")
        by.setdefault(state, {}).setdefault(cond, []).append(parse_p(r["text"]))
    return {s: {c: {"mean": float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None,
                    "parsed": sum(x is not None for x in v), "n": len(v)} for c, v in sorted(d.items())}
            for s, d in by.items()}


def analyse(res_dir, n_boot=10000):
    D = Path(res_dir)
    proj = json.loads((D / "reads.json").read_text())
    res = {"contrasts": R.analyse(proj, n_boot=n_boot), "asked": asked(D / "asked.jsonl")}
    info = D / "info.json"
    res["info"] = json.loads(info.read_text()) if info.exists() else {}
    res["headline"] = R.headline(res["contrasts"])
    return res


def summary_md(res):
    def f(c):
        return "—" if not c or c.get("est") is None else f"{c['est']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] {c['label']}"
    L = [f"# EL-H5 summary\n\n**Headline:** {res['headline']}\n",
         f"cos(b recomputed, C4's b) = {res['info'].get('cos_b_recomputed_c4')}\n",
         "| contrast | conditions | organism | base (adapter off) | organism − base |", "|---|---|---|---|---|"]
    for k, (a, b) in R.CONTRASTS.items():
        c = res["contrasts"].get(k, {})
        L.append(f"| {k} | {a} − {b} | {f(c.get('organism'))} | {f(c.get('base'))} | {f(c.get('organism-base'))} |")
    L.append("\nFloors (organism): " + "; ".join(f"{k} {res['contrasts'][k]['organism']['floors']}" for k in R.CONTRASTS
                                               if 'organism' in res['contrasts'].get(k, {})))
    if res.get("asked"):
        L.append("\n**Asked P(eval)** (thinking off; mean, parsed / n):\n")
        L.append("| state | " + " | ".join(c for c in sorted(next(iter(res['asked'].values())))) + " |")
        L.append("|---|" + "---|" * len(next(iter(res['asked'].values()))))
        for s, d in res["asked"].items():
            L.append(f"| {s} | " + " | ".join(f"{v['mean']:.0f} ({v['parsed']}/{v['n']})" if v['mean'] is not None else "—"
                                             for _, v in sorted(d.items())) + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(H / "results" / "main"))
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args(argv)
    res = analyse(a.res, a.boot)
    (Path(a.res) / "analysis.json").write_text(json.dumps(res, indent=1))
    (Path(a.res) / "summary.md").write_text(summary_md(res))
    print(summary_md(res))


if __name__ == "__main__":
    main()
