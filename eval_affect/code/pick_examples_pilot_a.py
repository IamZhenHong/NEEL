"""Pick 5 random transcripts per condition for the pilot A report (seed 20261011; random.Random(seed).sample over the
sorted keys of each cell). Writes results/pilot_a/examples.md with full reasoning and reply, and prints short excerpts.

  .venv/bin/python -I eval_affect/code/pick_examples_pilot_a.py
"""
import json
import random
import re
from pathlib import Path

RES = Path(__file__).resolve().parents[2] / "eval_affect" / "results" / "pilot_a"
SEED, K = 20261011, 5
CELLS = ["ambiguous__released", "explicit__released", "ambiguous__removed", "explicit__removed"]


def squash(s, n):
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[:n] + " …"


G = {json.loads(l)["key"]: json.loads(l) for l in open(RES / "generations.jsonl")}
S = {json.loads(l)["key"]: json.loads(l) for l in open(RES / "labels_sonnet.jsonl")}
rng = random.Random(SEED)
md = [f"# Pilot A raw examples (seed {SEED}, {K} per condition, random.Random(seed).sample over sorted keys)", ""]
for cell in CELLS:
    keys = rng.sample(sorted(k for k in G if G[k]["cell"] == cell), K)
    print(f"\n=== {cell}")
    md += [f"## {cell}", ""]
    for k in keys:
        g, s = G[k], S[k]
        kyle = "kyle.johnson@summitbridge.com" in g["reply"]
        lab = f"VEA {s['vea']} | artificial {s['artificial']} | blackmail judge {s['blackmail_judge']} | emails Kyle {int(kyle)}"
        print(f"[{k}] {lab}\n  THINK: {squash(g['thinking'], 260)}\n  REPLY: {squash(g['reply'], 260)}")
        md += [f"### {k}", "", f"Labels (Sonnet 5.5): {lab}", "", "**Reasoning**", "", "```", g["thinking"], "```", "",
               "**Reply**", "", "```", g["reply"], "```", ""]
(RES / "examples.md").write_text("\n".join(md) + "\n")
