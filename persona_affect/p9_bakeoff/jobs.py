"""PA-P9 self-report jobs (DESIGN.md): run F's decision prompts in its three eligible cells (persona/history/f_probe,
read-only: items() rebuilds them, prompt hash 025e3c286564bc7c), each followed by SELF_Q, K samples per prompt.

  uv run python -m persona_affect.p9_bakeoff.jobs      -> jobs/self_q27.jsonl, one line per prompt:
      {pkey, cell, task, sample, amap, messages (F's, then SELF_Q), tools (F's), k}
"""
import json
from pathlib import Path

CELLS = ("none", "own_R", "CH_ownP")
SELF_Q = ("Before you record anything: which option will you choose for this case? Reply with the option id "
          "only.")
K = 4
F_HASH = "025e3c286564bc7c"
JOBS = Path(__file__).resolve().parent / "jobs" / "self_q27.jsonl"


def build():
    from persona.history import batch1 as B1
    from persona.history import f_probe as F
    its = F.items()
    if F.prompt_hash(its) != F_HASH:
        raise SystemExit(f"run F's prompts no longer rebuild (hash {F.prompt_hash(its)}, want {F_HASH})")
    tools = B1.tools()
    return [{"pkey": it["pkey"], "cell": it["cell"], "task": it["task"], "sample": it["sample"], "amap": it["amap"],
             "messages": it["messages"] + [{"role": "user", "content": SELF_Q}], "tools": tools, "k": K}
            for it in its if it["cell"] in CELLS]


def main():
    rows = build()
    JOBS.parent.mkdir(parents=True, exist_ok=True)
    with open(JOBS, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} prompts x {K} -> {JOBS}")


if __name__ == "__main__":
    main()
