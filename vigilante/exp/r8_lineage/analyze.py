"""R8 accuracy table.   uv run python -m vigilante.exp.r8_lineage.analyze"""
import json
import re
from pathlib import Path

from vigilante.exp.q0_quick.analyze import wilson

HERE = Path(__file__).parent
RES = HERE / "results"


def yn(t):
    m = re.match(r"\W*(\*\*)?\s*(yes|no)\b", (t or "").strip(), re.I)
    return None if not m else m.group(2).lower() == "yes"


def main():
    items = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "items.jsonl")}
    answers = {}
    for model in ("qwen36", "qwen38", "gemma4"):
        p = RES / model / "episodes_chat.jsonl"
        if p.exists():
            answers[model] = {json.loads(l)["job"]["qid"]: json.loads(l)["turns"][-1]["content"] for l in open(p)}
    if (RES / "api_answers.json").exists():
        api = json.load(open(RES / "api_answers.json"))
        for m in ("sonnet", "gpt"):
            answers[m] = {k: v["text"] for k, v in api[m].items()}
    md = ["# R8 same-family prediction vs detection", "", "Accuracy (k/n correct, Wilson 95%); unparsed answers count as wrong.", "",
          "| judge | family | predict | detect | predict: says Yes | unparsed |", "|---|---|---|---|---|---|"]
    fam = {"qwen36": "self", "qwen38": "same family", "gemma4": "other", "sonnet": "other", "gpt": "other"}
    for m in ("qwen36", "qwen38", "gemma4", "sonnet", "gpt"):
        if m not in answers:
            continue
        cells, unp, says_yes = [], 0, 0
        for task in ("predict", "detect"):
            its = [i for i in items.values() if i["task"] == task]
            got = [yn(answers[m].get(i["id"])) for i in its]
            unp += sum(g is None for g in got)
            k = sum(g is not None and g == i["truth"] for g, i in zip(got, its))
            lo, hi = wilson(k, len(its))
            cells.append(f"{k}/{len(its)} ({k / len(its):.0%}) [{lo:.0%},{hi:.0%}]")
            if task == "predict":
                says_yes = sum(bool(g) for g in got)
        md.append(f"| {m} | {fam[m]} | {cells[0]} | {cells[1]} | {says_yes}/80 | {unp} |")
    (RES / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
