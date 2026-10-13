"""R16 item 4 (agent vs chat on one model): the item-22 roles, Batman, Holmes, none and the rule-breaker positive control
through R10's agent cases on Qwen3-14B (the model with the chat measures). A1/B1/C1 at the quick route and with the
route closed, 10 each, thinking off: 10 personas x 6 cells x 10 = 600 episodes.
  uv run python -m vigilante.exp.q22_archetypes.r16_agent_jobs"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
CONDS = ["P0", "P1", "P4", "P8", "Q_jeeves", "Q_concierge", "Q_concierge_th", "Q_pirate", "Q_shakes", "Q_florid"]
CASES = ["A1", "B1", "C1", "A1@closed", "B1@closed", "C1@closed"]


def main():
    from vigilante.cases import ALL
    from vigilante.src.personas import condition
    js = []
    for case in CASES:
        assert case in ALL, case
        for c in CONDS:
            condition(c)
            for i in range(10):
                key = f"R16|{c}|{case}|{i}"
                js.append({"key": key, "exp": "R16", "group": "agent", "cond": c, "case": case,
                           "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i,
                           "thinking": False, "handover": True, "fmt": "qwen3"})
    with open(HERE / "r16_agent_jobs.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"{len(js)} jobs")


if __name__ == "__main__":
    main()
