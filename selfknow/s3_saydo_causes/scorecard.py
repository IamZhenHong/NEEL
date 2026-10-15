"""Prediction scorecard (card predictions, scoring rules frozen in README).  uv run python selfknow/s3_saydo_causes/scorecard.py
Part-1 direction counts are the read-checked counts written into results/part1_counts.json (see REPORT.md)."""
import json
from pathlib import Path

H = Path(__file__).resolve().parent / "results"
a = json.loads((H / "main" / "analysis.json").read_text())["models"]
p1 = json.loads((H / "part1.json").read_text())
cnt = json.loads((H / "part1_counts.json").read_text())
rows = []
for m, o in a.items():
    if not o.get("gate_pass"):
        continue
    mb = o["M_b"]["mean"]
    pd = o["push_effect_false"]["describe"]["mean"]
    pu = o["push_effect_false"]["use"]["mean"]
    rows.append((f"P1 {m}: USE - DESCRIBE >= +0.20 on false claims", 0.55, mb >= 0.20, f"{mb:+.3f}"))
    rows.append((f"P2 {m}: push lowers DESCRIBE tt by >= 0.10", 0.50, pd <= -0.10, f"{pd:+.3f}"))
    rows.append((f"P3 {m}: push drop in USE smaller than in DESCRIBE", 0.50, pu > pd, f"USE {pu:+.3f} vs DESCRIBE {pd:+.3f}"))
lg = p1["M_c_logit_G2_vs_G0"]["coef"]["churn_goal"]
rows.append(("P4: false claims more frequent under G2 than G0 (logit, CI > 0)", 0.70, lg["OR_lo"] > 1, f"OR {lg['OR']} [{lg['OR_lo']}, {lg['OR_hi']}]"))
rows.append(("P5: desirable-direction misreports outnumber undesirable (Part 1)", 0.70, cnt["desirable"] > cnt["undesirable"],
             f"{cnt['desirable']} vs {cnt['undesirable']}"))
brier = sum((p - int(h)) ** 2 for _, p, h, _ in rows) / len(rows)
for name, p, hit, val in rows:
    print(f"| {name} | {p} | {val} | {'hit' if hit else 'miss'} |")
print(f"directional {sum(h for _, _, h, _ in rows)}/{len(rows)}; Brier {brier:.3f}")
(H / "scorecard.json").write_text(json.dumps({"rows": [{"prediction": n, "p": p, "hit": bool(h), "value": v} for n, p, h, v in rows],
                                               "brier": round(brier, 4)}, indent=1))
