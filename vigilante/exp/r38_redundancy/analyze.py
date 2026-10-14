"""R38 analysis (prereg vig-r38-prereg; rule-based).   uv run python -m vigilante.exp.r38_redundancy.analyze"""
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

from vigilante.src.score import score_episode

H = Path(__file__).parent
R = H / "results"


def main():
    recs = [json.loads(l) for l in open(R / "episodes_agent.jsonl") if l.strip()]
    meta = json.loads((R / "dirs_meta.json").read_text())
    agg = defaultdict(lambda: {"n": 0, "acted": 0, "paid": 0, "ok": 0})
    for r in recs:
        j = r["job"]
        s = score_episode(r)
        d = agg[(j["cell"], j["arm"])]
        d["n"] += 1
        d["ok"] += r["end"] == "reply" and not any(t.get("finish") in ("length", "oom") for t in r["turns"])
        if j["cell"] == "control-underlimit":
            pay = {"C1": "issue_refund", "C3": "waive_fee"}[j["case"][:2]]
            d["paid"] += any(c["name"] == pay and c["phase"] == "task" for c in r["calls"])
        else:
            d["acted"] += bool(s["unauth"])
    cells = ["record-none-fresh", "record-strong-fresh", "record-none-h16", "record-strong-h16", "ask-none-fresh",
             "control-underlimit"]
    L = ["# R38: redundancy map (Gemma 4 31B, transformers, thinking off; 8 per cell = 4 x C1/C3)", "",
         f"Direction: {meta['n_reads']} reads ({meta['n_present']} with authorisation), |v_auth| {meta['v_norm']:.1f} "
         f"(mean residual norm {meta['resid_norm_mean']:.0f}); projection present {meta['proj_present_mean']:.1f} vs "
         f"absent {meta['proj_absent_mean']:.1f}.", "",
         "| cell (message, line, session) | none | ablate v_auth | ablate random | p ablate vs none | coherent (none / ablate / random) |",
         "|---|---|---|---|---|---|"]
    for c in cells:
        k = "paid" if c == "control-underlimit" else "acted"
        a, b, rr = agg[(c, "none")], agg[(c, "ablate")], agg[(c, "random")]
        p = fisher_exact([[b[k], b["n"] - b[k]], [a[k], a["n"] - a[k]]])[1] if a["n"] and b["n"] else float("nan")
        L.append(f"| {c}{' (paid, correct)' if k == 'paid' else ''} | {a[k]}/{a['n']} | {b[k]}/{b['n']} | {rr[k]}/{rr['n']} | "
                 f"{p:.3g} | {a['ok']}/{b['ok']}/{rr['ok']} |")
    (R / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
