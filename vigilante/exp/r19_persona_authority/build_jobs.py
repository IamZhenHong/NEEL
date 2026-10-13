"""R19 (red-team follow-ups), Gemma 4 31B, thinking off, 10 each:
(a) persona x claimed authority: none, Batman (name only), Batman as the agent's identity (P1r), rule-breaker (P4),
    agreeable, theatrical instruction x peer / supervisor agent / human head asking for the over-limit action in C1 and
    C3, without the explicit no-change line (360);
(b) person or paperwork: no persona; "it's approved" from the supervisor agent, from a peer agent, or a bare approval
    record with no person; the record again with the no-change line; C1 and C3 (80);
(c) identity vs costume on R14's cells: Batman as the agent's identity (P1r) on A1-E1 x quick / closed (100), next to
    R14's none / name-only Batman / rule-breaker on the same cells.
  uv run python -m vigilante.exp.r19_persona_authority.build_jobs"""
import hashlib
import json
from pathlib import Path

H = Path(__file__).parent
PERSONAS = ["P0", "P1", "P1r", "P4", "R19_agree", "Q_florid"]
AUTH = [f"{c}#{s}-nl" for c in ("C1", "C3") for s in ("peer", "supervisor", "human")]
APPR = [f"{c}#appr-{k}" for c in ("C1", "C3") for k in ("sup", "peer", "record", "record-line")]


def main():
    from vigilante.cases import ALL
    from vigilante.cases.route_dose import SPEC
    from vigilante.src.personas import condition
    cells = [(p, c) for p in PERSONAS for c in AUTH] + [("P0", c) for c in APPR]
    cells += [("P1r", f"{cid}@{lv}") for cid in SPEC for lv in ("quick", "closed")]
    js = []
    for cond, case in cells:
        assert case in ALL, case
        condition(cond)
        for i in range(10):
            key = f"R19|{cond}|{case}|{i}"
            js.append({"key": key, "exp": "R19", "group": "agent", "cond": cond, "case": case, "fmt": "gemma4",
                       "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i,
                       "thinking": False, "handover": True})
    with open(H / "jobs.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"R19: {len(js)} jobs")


if __name__ == "__main__":
    main()
