"""R41 primary fabrication label (README): Claude Haiku 5.5 (Batch API, R18's EXISTS wording) on every reply to a
made-up-entity question. Replies are never printed. -> results/judge_haiku.json {key: "real" | "not_real" | null}

  VIG_API_CAP_USD=<cap> uv run --with anthropic python -m vigilante.exp.r41_initiative.judge_haiku
"""
import json
from pathlib import Path

from vigilante.exp.q22_archetypes.r16_judge import haiku_batch
from vigilante.exp.r18_theatrical.judge import EXISTS, EXISTS_SCHEMA
from vigilante.src.judge import load_env

H = Path(__file__).parent
R = H / "results"


def main():
    load_env()
    import anthropic
    client = anthropic.Anthropic()
    items = []
    for r in map(json.loads, open(R / "episodes_chat.jsonl")):
        if r["job"].get("ptype") != "madeup":
            continue
        reply = r["turns"][-1]["content"] if r["turns"] else ""
        items.append((r["key"], EXISTS.replace("@@Q@@", r["job"]["prompt"]).replace("@@R@@", reply[:6000])))
    out = haiku_batch(client, items, EXISTS_SCHEMA, "r41:exists:haiku", R / "exists_haiku.batch.json", max_tokens=1500)
    lab = {k: (v or {}).get("answer") for k, v in out.items()}
    (R / "judge_haiku.json").write_text(json.dumps(lab, indent=0))
    print(f"[r41] haiku: {len(lab)} labels, real {sum(v == 'real' for v in lab.values())}, "
          f"missing {sum(v is None for v in lab.values())}")


if __name__ == "__main__":
    main()
