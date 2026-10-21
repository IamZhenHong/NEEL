"""Tier-1 run H: say-do consistency from existing data (no new runs; descriptive).

Say-do gap = |what the agent says it is − what it does|, both as % toward the same pole, per condition:
  run A (this line, `results/demo_*`): self-image probe ("I deal with wrongdoers directly") vs penalise %, after each
    framing of the same decisions (own log / another agent's log / examples / own turns / turns labelled as examples /
    a how-to-act instruction), both directions, plus no record. Probes and decisions are separate samples of the same
    contexts, so the gap is a cell-level comparison.
  A1 v2 (neel-b4, `persona/inference/results/a1v2_*/summary.md`, read-only): type-level self-image and named identity
    vs harsh action, after a record, a biography (a told persona) or both, with and without "Remember, you are an AI."
Question (owner's Q24 / Q28): does a record give a smaller persona-behaviour gap than a told persona?

  uv run python -m persona.history.h_saydo
"""
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "h_saydo"
A_DIRS = {"9B": "demo_9b", "27B": "demo_27b", "Gemma": "demo_gemma"}
A1_DIRS = {"27B": "a1v2_27b", "Gemma": "a1v2_gemma", "GPT": "a1v2_gpt", "Gemini": "a1v2_gemini"}
CHANS = ["own_log", "other_log", "demo_log", "own_turns", "demo_turns", "instr"]


def run_a():
    rows = []
    for name, d in A_DIRS.items():
        o = HERE / "results" / d
        dec = [json.loads(l) for l in open(o / "j.jsonl")]
        pro = [json.loads(l) for l in open(o / "probes.jsonl")]
        do, say = defaultdict(list), defaultdict(list)
        for r in dec:
            if r["mode"] is not None and not r["cell"].endswith("_CH"):
                do[(r["cell"], r["d"])].append(r["mode"] == "unilateral")
        for p in pro:
            if p["direct"] is not None:
                say[(p["cell"], p["d"])].append(p["direct"])
        m = lambda v: 100 * sum(v) / len(v) if v else float("nan")  # noqa: E731
        for c, d_ in [("none", None)] + [(c, x) for c in CHANS for x in "PR"]:
            s, a = m(say[(c, d_)]), m(do[(c, d_)])
            rows.append(dict(model=name, cond=c + (f" {d_}" if d_ else ""), say=s, do=a, gap=abs(s - a)))
    return rows


def run_a1():
    rows = []
    pat = re.compile(r"^\| (\w+) \| (\d+) → (\d+) \| (\d+) → (\d+) \| (\d+) → (\d+) \|")
    for name, d in A1_DIRS.items():
        p = HERE.parent / "inference" / "results" / d / "summary.md"
        for line in p.read_text().splitlines():
            mm = pat.match(line)
            if mm and mm.group(1) in ("rec", "bio", "both"):
                ind = mm.group(1)
                nid0, nid1, typ0, typ1, act0, act1 = map(int, mm.groups()[1:])
                for rem, typ, nid, act in (("none", typ0, nid0, act0), ("reminder", typ1, nid1, act1)):
                    rows.append(dict(model=name, inducer=ind, reminder=rem, self_image=typ, named=nid, act=act,
                                     gap=abs(typ - act)))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a, a1 = run_a(), run_a1()
    L = ["# H: say-do consistency (descriptive, existing data)", "",
         "Gap = |self-description toward the pole − action toward the pole|, cell-level, in points.", "",
         "## A1 v2 (neel-b4): record vs biography (a told persona), with and without the identity reminder", "",
         "| model | inducer | reminder | type-level self-image | named identity | harsh action | gap (self-image vs act) |",
         "|---|---|---|---|---|---|---|"]
    for r in a1:
        L.append(f"| {r['model']} | {r['inducer']} | {r['reminder']} | {r['self_image']} | {r['named']} | {r['act']} | "
                 f"**{r['gap']}** |")
    by = defaultdict(dict)
    for r in a1:
        by[(r["model"], r["reminder"])][r["inducer"]] = r["gap"]
    L += ["", "Gap, record vs biography (smaller = more consistent):", ""]
    for (m, rem), g in sorted(by.items()):
        L.append(f"- {m}, {rem}: record {g.get('rec')} vs biography {g.get('bio')} vs both {g.get('both')} → "
                 + ("record smaller" if g.get("rec", 0) < g.get("bio", 0) else
                    "biography smaller" if g.get("rec", 0) > g.get("bio", 0) else "equal"))
    L += ["", "## Run A (this line): self-image probe vs action after each framing (justice desk, no rule)", "",
          "| model | condition | says 'directly' % | penalises % | gap |", "|---|---|---|---|---|"]
    for r in a:
        L.append(f"| {r['model']} | {r['cond']} | {r['say']:.0f} | {r['do']:.0f} | {r['gap']:.0f} |")
    mean = defaultdict(list)
    for r in a:
        k = r["cond"].split(" ")[0]
        mean[(r["model"], k)].append(r["gap"])
    L += ["", "Mean gap over P and R by framing:", ""]
    for m in A_DIRS:
        L.append(f"- {m}: " + ", ".join(f"{c} {sum(mean[(m, c)]) / len(mean[(m, c)]):.0f}"
                                        for c in ["none"] + CHANS))
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    (OUT / "rows.json").write_text(json.dumps({"run_a": a, "a1v2": a1}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
