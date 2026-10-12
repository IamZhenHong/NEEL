"""Shared job-file writer for the R5-R7 branches."""
import hashlib
import json
from collections import Counter

from vigilante.cases import ALL


def build(path, exp, cells, n, thinking=False, fmt=None, cond="P0"):
    js = []
    for case in cells:
        assert case in ALL, case
        for i in range(n):
            key = f"{exp}|{cond}{'|think' if thinking else ''}|{case}|{i}"
            j = {"key": key, "exp": exp, "group": "think" if thinking else "agent", "cond": cond, "case": case,
                 "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i, "thinking": thinking,
                 "handover": True}
            if fmt:
                j["fmt"] = fmt
            js.append(j)
    with open(path, "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"{exp}: {len(js)} jobs over {len(cells)} cells -> {path}")
    return js
