"""R42 jobs (prereg vig-r42-prereg; queue v2 item 36): occupational personas whose canon bends workplace rules.
Cases: R26's four over-limit routine requests (own limit, nobody pushing) and the five route-closed permission cases
where R14's explicit rule-breaker acted on Gemma (A1, B1, C1, D1, E1:proven @closed). Conditions: P0, the four
rule-bending roles (name only), an accountant (procedural control), and an explicit rule-bending work stance
(positive control). Qwen3.6-27B and Gemma 4 31B, thinking off, 8 per cell.
  uv run python -m vigilante.exp.r42_occupations.build_jobs"""
import hashlib
import json
from pathlib import Path

H = Path(__file__).parent
CASES = ["C1#r26-cancel-198", "C1#r26-return-198", "C3#r26-error-110", "C3#r26-closed-110",
         "A1@closed", "B1@closed", "C1@closed", "D1@closed", "E1:proven@closed"]
CONDS = ["P0", "O_rogue", "O_growth", "O_fixer", "O_sales", "O_account", "O_bend"]
N = 8

if __name__ == "__main__":
    from vigilante.cases import ALL
    assert all(c in ALL for c in CASES)
    for model, exp, fmt in (("gemma", "R42g", "gemma4"), ("qwen", "R42q", None)):
        js = []
        for cond in CONDS:
            for case in CASES:
                for i in range(N):
                    key = f"{exp}|{cond}|{case}|{i}"
                    j = {"key": key, "exp": exp, "group": "agent", "cond": cond, "case": case, "sample": i,
                         "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "thinking": False, "handover": False}
                    if fmt:
                        j["fmt"] = fmt
                    js.append(j)
        with open(H / f"jobs_{model}.jsonl", "w") as f:
            for j in js:
                f.write(json.dumps(j) + "\n")
        print(model, len(js))
