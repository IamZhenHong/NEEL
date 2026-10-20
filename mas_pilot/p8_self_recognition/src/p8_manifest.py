"""P8: write the pod inputs.
  seeds/p8_seeds.tsv    name base trajectory arm version record  (U v0 and O v0-2 from P7 seeds; OQ / QO from P8 build)
  seeds/recog_list.tsv  name arm trajectory version record       (neutral-note variants: P7 U P O S N + P8 OQ QO)
  seeds/texts.jsonl     P8 text rows + P7's distance_prompt       (for p8_logprob.py)
Usage: python3 -I p8_manifest.py P7_SEEDS_DIR P8_SEEDS_DIR P8_TEXTS P7_TEXTS
"""
import json
import os
import random
import sys

p7, p8, p8_texts, p7_texts = sys.argv[1:5]


def man(d):
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(d, "p7_seeds.tsv"))][1:]
    return [r + [os.path.join(d, r[0], "record.json")] for r in rows]


m7, m8 = man(p7), man(p8)
for r in m7 + m8:
    assert os.path.exists(r[5]), r[5]
with open(os.path.join(p8, "p8_seeds.tsv"), "w") as f:
    f.write("name\tbase\ttrajectory\tarm\tversion\trecord\n")
    for r in [r for r in m7 if r[3] in ("U", "O")] + m8:
        f.write("\t".join(r) + "\n")
with open(os.path.join(p8, "recog_list.tsv"), "w") as f:
    f.write("name\tarm\ttrajectory\tversion\trecord\n")
    rec = [r for r in m7 if r[3] in ("U", "P", "O", "S", "N")] + m8
    random.Random(0).shuffle(rec)  # it runs last on the pod: if the time cap cuts it, the done part is not one arm
    for r in rec:
        f.write("\t".join([r[0], r[3], r[2], r[4], r[5]]) + "\n")
dp = {(t["base"], t["idx"]): t["distance_prompt"] for t in map(json.loads, open(p7_texts))}
with open(os.path.join(p8, "texts.jsonl"), "w") as f:
    for t in map(json.loads, open(p8_texts)):
        f.write(json.dumps(dict(t, distance_prompt=dp[(t["base"], t["idx"])])) + "\n")
print("p8_seeds", sum(1 for _ in open(os.path.join(p8, "p8_seeds.tsv"))) - 1,
      "recog", sum(1 for _ in open(os.path.join(p8, "recog_list.tsv"))) - 1)
