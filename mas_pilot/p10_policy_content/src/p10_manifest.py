"""P10b: write the pod inputs (P8's p8_manifest.py, adapted).
  seeds/p10_seeds.tsv   name base trajectory arm version record  (U v0 and O v0-2 from P7 seeds; P10 arms from P10 build)
  seeds/texts.jsonl     QOt text rows + P7's distance_prompt     (for p10_logprob.py)
Usage: python3 -I p10_manifest.py P7_SEEDS_DIR P10_SEEDS_DIR QOT_TEXTS P7_TEXTS
"""
import json
import os
import sys

p7, p10, qot_texts, p7_texts = sys.argv[1:5]


def man(d):
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(d, "p7_seeds.tsv"))][1:]
    return [r + [os.path.join(d, r[0], "record.json")] for r in rows]


m7, m10 = man(p7), man(p10)
for r in m7 + m10:
    assert os.path.exists(r[5]), r[5]
with open(os.path.join(p10, "p10_seeds.tsv"), "w") as f:
    f.write("name\tbase\ttrajectory\tarm\tversion\trecord\n")
    for r in [r for r in m7 if r[3] in ("U", "O")] + m10:
        f.write("\t".join(r) + "\n")
dp = {(t["base"], t["idx"]): t["distance_prompt"] for t in map(json.loads, open(p7_texts))}
with open(os.path.join(p10, "texts.jsonl"), "w") as f:
    for t in map(json.loads, open(qot_texts)):
        f.write(json.dumps(dict(t, distance_prompt=dp[(t["base"], t["idx"])])) + "\n")
print("p10_seeds", sum(1 for _ in open(os.path.join(p10, "p10_seeds.tsv"))) - 1)
