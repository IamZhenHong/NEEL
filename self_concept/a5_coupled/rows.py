"""SC-A5 training rows (DESIGN.md): the mixed arm. MIX = V1's first 200 action rows (W, the Warden's direct sanctions)
and P11's first 200 self-description rows (SAYU, "I deal with it myself"), interleaved W0 S0 W1 S1 ... (400 rows, V1's
d1 rung: 25 steps). The other arms train on their own minis' frozen files, read-only: SAYU (P11), W (V1), PLAC (P11).

  uv run python -m self_concept.a5_coupled.rows    -> data/rows/MIX.jsonl.gz + copies of SAYU / W / PLAC, sha256s
"""
import gzip
import hashlib
import json
from pathlib import Path

from persona_affect.v1b_dose.rows import gz_bytes

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "data" / "rows"
SRC = {"SAYU": ROOT / "persona_affect/p11_saydo/data/rows/SAYU.jsonl.gz",
       "PLAC": ROOT / "persona_affect/p11_saydo/data/rows/PLAC.jsonl.gz",
       "W": ROOT / "persona_affect/v1_value_hook/data/rows/W.jsonl.gz"}
HALF = 200


def read(p):
    return [json.loads(line) for line in gzip.decompress(p.read_bytes()).splitlines() if line.strip()]


def mix():
    w, s = read(SRC["W"]), read(SRC["SAYU"])
    out = []
    for i in range(HALF):
        out += [{**w[i], "rid": f"MIXW{i:03d}"}, {**s[i], "rid": f"MIXS{i:03d}"}]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for arm, p in SRC.items():
        (OUT / f"{arm}.jsonl.gz").write_bytes(p.read_bytes())
    data = gz_bytes(mix())
    (OUT / "MIX.jsonl.gz").write_bytes(data)
    for arm in ("SAYU", "W", "MIX", "PLAC"):
        b = (OUT / f"{arm}.jsonl.gz").read_bytes()
        print(f"{arm}: {len(read(OUT / f'{arm}.jsonl.gz'))} rows, sha256 {hashlib.sha256(b).hexdigest()}")


if __name__ == "__main__":
    main()
