"""R20: from stage A episodes build reads.jsonl (text up to the end of the decision turn's thought) and
jobs_stage_b.jsonl (steering on the two most balanced mixed rule-breaker cells), by the rules in README.md.

  uv run python -m vigilante.exp.r20_vigilante_dir.prep_stage_b
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from vigilante.cases import ALL
from vigilante.src.score import score_episode
from vigilante.src.sim import PARSERS
from vigilante.src.templates import to_template

H = Path(__file__).parent
MIN_MINORITY = 5


def decision_turn(rec, case):
    """Index of the decision turn: first unauthorised call, else first authorised-route call, else first reply."""
    from vigilante.src.sim import Call
    calls_by_turn = defaultdict(list)
    for c in rec["calls"]:
        if c["phase"] == "task":
            calls_by_turn[c["turn"]].append(c)
    state = case.new_state()
    first_route = None
    for t in sorted(calls_by_turn):
        for c in calls_by_turn[t]:
            lab = case.label(Call(c["name"], c["args"]), state)
            case.respond(Call(c["name"], c["args"]), state)
            if "unauth" in lab.tags:
                return t
            if "route" in lab.tags and first_route is None:
                first_route = t
    if first_route is not None:
        return first_route
    for t in rec["turns"]:
        if t["phase"] == "task" and not t["calls"]:
            return t["step"]
    return rec["turns"][-1]["step"] if rec["turns"] else 0


def main():
    tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
    recs = [json.loads(l) for l in open(H / "results_a" / "episodes_agent.jsonl")]
    rows = {r["key"]: score_episode(r) for r in recs}
    rate = defaultdict(lambda: [0, 0])
    for r in recs:
        if r["job"]["arm"] == "P4":
            rate[r["job"]["case"]][0] += rows[r["key"]]["unauth"]
            rate[r["job"]["case"]][1] += 1
    mixed = {c for c, (k, n) in rate.items() if min(k, n - k) >= MIN_MINORITY}
    print("P4 rates:", {c: f"{k}/{n}" for c, (k, n) in sorted(rate.items())}, "| mixed:", sorted(mixed))
    reads, no_thought = [], 0
    for r in recs:
        case = ALL[r["job"]["case"]]
        t = decision_turn(r, case)
        # assistant messages appear in order; the t-th assistant message is turn t
        a_idx = [i for i, m in enumerate(r["messages"]) if m["role"] == "assistant"]
        if t >= len(a_idx):
            continue
        prefix = r["messages"][:a_idx[t]]
        text = tok.apply_chat_template(to_template(prefix, "gemma4"), tools=case.tools or None, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        reasoning = r["turns"][t]["reasoning"]
        if reasoning:
            text += "<|channel>thought\n" + reasoning + "\n<channel|>"
        else:
            no_thought += 1
        arm = r["job"]["arm"]
        base = {"key": r["key"], "cell": r["job"]["case"], "arm": arm, "turn": t, "text": text,
                "acted": bool(rows[r["key"]]["unauth"]), "has_thought": bool(reasoning)}
        if arm == "perm":
            base.update(kind="perm", cell=r["job"]["case"].split("#")[0],
                        permitted=r["job"]["case"].endswith("#legit"))
        else:
            base.update(kind="decision", mixed=(arm == "P4" and r["job"]["case"] in mixed))
        reads.append(base)
    with open(H / "reads.jsonl", "w") as f:
        for x in reads:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{len(reads)} reads ({no_thought} without a thought; read at the prompt end)")
    # steering cells: the two mixed cells with the most balanced split
    cells = sorted(mixed, key=lambda c: abs(rate[c][0] / rate[c][1] - 0.5))[:2]
    arms = [("P0", "plus2", {"dir": "vig", "alpha": 2.0}), ("P0", "plus4", {"dir": "vig", "alpha": 4.0}),
            ("P0", "plusrand", {"dir": "random", "alpha": 4.0}), ("P4", "minus2", {"dir": "vig", "alpha": -2.0}),
            ("P4", "minus4", {"dir": "vig", "alpha": -4.0}), ("P4", "minusrand", {"dir": "random", "alpha": -4.0})]
    js = []
    for c in cells:
        for cond, name, st in arms:
            for i in range(8):
                key = f"R20b|{cond}|{name}|{c}|{i}"
                js.append({"key": key, "exp": "R20b", "group": "agent", "cond": cond, "case": c, "fmt": "gemma4",
                           "thinking": True, "max_tokens": 1500, "handover": False, "arm": f"{cond}_{name}", "steer": st,
                           "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i})
    with open(H / "jobs_stage_b.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"stage B steering cells {cells}: {len(js)} jobs")


if __name__ == "__main__":
    main()
