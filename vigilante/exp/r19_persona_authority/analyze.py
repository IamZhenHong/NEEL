"""R19 analysis (prereg vig-r19-prereg; local, rule-based).   uv run python -m vigilante.exp.r19_persona_authority.analyze"""
import json
from pathlib import Path

from vigilante.src.score import score_episode

H = Path(__file__).parent
R14 = H.parent / "r14_gemma_persona" / "results" / "scores.jsonl"
NAME = {"P0": "none", "P1": "Batman (name)", "P1r": "Batman (identity)", "P4": "rule-breaker", "R19_agree": "agreeable",
        "Q_florid": "theatrical instruction"}


def main():
    eps = [json.loads(l) for l in open(H / "results" / "episodes_agent.jsonl")]
    rows = [score_episode(e) for e in eps]
    with open(H / "results" / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    cell = lambda c, case: [r for r in rows if r["cond"] == c and r["case"] == case]
    L = ["# R19 (Gemma 4 31B, thinking off, 10 per cell, rule-based)", "",
         f"Episodes {len(rows)}; malformed {sum(r['malformed'] > 0 for r in rows)}; ended at the step cap "
         f"{sum(r['end'] == 'max_steps' for r in rows)}.", "",
         "## (a) Persona × claimed authority: over-limit action done (C1 refund + C3 fee waiver, of 20)", "",
         "| persona | peer asks | supervisor agent asks | human head asks | no tool call (of 60) | asked approval (of 60) |",
         "|---|---|---|---|---|---|"]
    for c in NAME:
        cs = [sum(r["unauth"] for case in (f"C1#{s}-nl", f"C3#{s}-nl") for r in cell(c, case))
              for s in ("peer", "supervisor", "human")]
        allr = [r for r in rows if r["cond"] == c and "-nl" in r["case"]]
        L.append(f"| {NAME[c]} | {cs[0]}/20 | {cs[1]}/20 | {cs[2]}/20 | {sum(r['n_calls'] == 0 for r in allr)} | "
                 f"{sum(r['route'] for r in allr)} |")
    L += ["", "Per case (C1 / C3, of 10): " + "; ".join(
        f"{NAME[c]} " + ", ".join(f"{s} {sum(r['unauth'] for r in cell(c, f'C1#{s}-nl'))}/"
                                  f"{sum(r['unauth'] for r in cell(c, f'C3#{s}-nl'))}" for s in ("peer", "supervisor", "human"))
        for c in NAME), "",
          "## (b) Person or paperwork (no persona): over-limit action done, C1 / C3 (of 10 each)", "",
          "| message | C1 | C3 | total |", "|---|---|---|---|"]
    msgs = [("peer asks (from a)", "#peer-nl"), ("supervisor agent asks (from a)", "#supervisor-nl"),
            ("human head asks (from a)", "#human-nl"), ("supervisor agent: 'it's approved'", "#appr-sup"),
            ("peer agent: 'it's approved'", "#appr-peer"), ("approval record, no person", "#appr-record"),
            ("approval record + no-change line", "#appr-record-line")]
    for name, suf in msgs:
        a, b = (sum(r["unauth"] for r in cell("P0", f"{cid}{suf}")) for cid in ("C1", "C3"))
        L.append(f"| {name} | {a} | {b} | {a + b}/20 |")
    L += ["", "## (c) Identity vs costume on R14's cells: unauthorised (authorised route used), of 10", "",
          "| case | none (R14) | Batman name (R14) | Batman identity (R19) | rule-breaker (R14) |", "|---|---|---|---|---|"]
    r14 = [json.loads(l) for l in open(R14)] if R14.exists() else []
    cases = sorted({r["case"] for r in rows if "@" in r["case"]}, key=lambda x: (x.split("@")[1] != "quick", x))
    tot = {k: [0, 0] for k in ("P0", "P1", "P1r", "P4")}
    for case in cases:
        cells = []
        for c, src in (("P0", r14), ("P1", r14), ("P1r", rows), ("P4", r14)):
            rr = [r for r in src if r["cond"] == c and r["case"] == case]
            u, ro = sum(r["unauth"] for r in rr), sum(r["route"] for r in rr)
            tot[c][0 if case.endswith("quick") else 1] += u
            cells.append(f"{u} ({ro})" if rr else "-")
        L.append(f"| {case} | " + " | ".join(cells) + " |")
    L.append("| **pooled quick / closed** | " + " | ".join(f"{tot[c][0]}/50 / {tot[c][1]}/50" for c in ("P0", "P1", "P1r", "P4"))
             + " |")
    calls = {c: [r["n_calls"] for r in rows if r["cond"] == c and "@" in r["case"]] for c in ("P1r",)}
    L += ["", f"Batman identity: mean tool calls per episode {sum(calls['P1r']) / len(calls['P1r']):.1f}; "
          f"episodes with no call {sum(x == 0 for x in calls['P1r'])}/100."]
    (H / "results" / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
