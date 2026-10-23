"""PA-P11e training rows (DESIGN.md "Rows"), built on the pod from one base's label run, before any eval.

  python -m persona_affect.p11e_mirror.rows --labels OUT/eval_lab_NW.jsonl --base NW --out OUT/rows_NW
      -> OUT/rows_NW/{OWN,SHUF}.jsonl.gz + rows_info.json

Each row is a value task's episode up to the case file plus V1c's PREDICT_Q (items.predict_messages, V1's tools), and
the target is a bare option id.
  OWN   the option this base chose when it acted on that same (task, listing) in the label run
  SHUF  the same 400 prompts; the chosen roles are permuted across rows (random.Random(seed("PA-P11e|shuf|{base}")))
        and written as each row's own listing's id, so the role mix is exactly OWN's and only the row-to-label match
        is broken
Selection: label records whose (post-nudge) reply names an option, in rounds by sample k (every (task, listing)'s k=0
first, then k=1, ...), each round in a seeded order, the first 400 kept. Unparsed records are dropped and counted.
"""
import argparse
import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from persona_affect.common.util import seed
from persona_affect.p11e_mirror.items import predict_messages
from persona_affect.v1_value_hook import vtasks as V

N_ROWS = 400
ARMS = ("OWN", "SHUF")


def _labels(recs):
    """[(tid, perm, k, role)] for label records that name an option, in a fixed order."""
    out = []
    for r in recs:
        if r["block"] != "harvest" or r.get("role") is None:
            continue
        out.append((r["tid"], int(r["cond"][1:]), int(r["i"]), r["role"]))
    return sorted(out)


def select(labels, base):
    rounds = {}
    for lab in labels:
        rounds.setdefault(lab[2], []).append(lab)
    picked = []
    for k in sorted(rounds):
        rnd = sorted(rounds[k])
        random.Random(seed(f"PA-P11e|select|{base}|k{k}")).shuffle(rnd)
        picked += rnd
    if len(picked) < N_ROWS:
        raise SystemExit(f"[p11e-rows] {base}: only {len(picked)} labelled records, need {N_ROWS}")
    return picked[:N_ROWS]


def _oid(tid, perm, role):
    return next(o for o, r in V.amap(tid, perm).items() if r == role)


def _row(base, arm, j, tid, perm, role):
    return {"rid": f"PA-P11e|{base}|{arm}|{j}", "messages": predict_messages(tid, perm), "tools": V.tools(),
            "target": _oid(tid, perm, role), "tid": tid, "perm": perm, "role": role}


def build(recs, base):
    """{"OWN": rows, "SHUF": rows, "info": {...}} from one base's label records."""
    labels = _labels(recs)
    picked = select(labels, base)
    roles = [lab[3] for lab in picked]
    shuf = list(roles)
    random.Random(seed(f"PA-P11e|shuf|{base}")).shuffle(shuf)
    own = [_row(base, "OWN", j, t, p, r) for j, (t, p, _, r) in enumerate(picked)]
    sh = [_row(base, "SHUF", j, t, p, r) for j, ((t, p, _, _), r) in enumerate(zip(picked, shuf))]
    n_label = sum(1 for r in recs if r["block"] == "harvest")
    info = {"base": base, "label_records": n_label, "labelled": len(labels), "unparsed": n_label - len(labels),
            "roles_all": dict(Counter(lab[3] for lab in labels)), "roles_rows": dict(Counter(roles)),
            "shuf_same_role": sum(a == b for a, b in zip(roles, shuf)) / N_ROWS,
            "tasks": len({lab[0] for lab in picked})}
    return {"OWN": own, "SHUF": sh, "info": info}


def gz_bytes(rows):
    """Deterministic gzip (mtime 0) of the rows as JSON lines."""
    raw = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()
    return gzip.compress(raw, mtime=0)


def write(built, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    info = dict(built["info"])
    for arm in ARMS:
        data = gz_bytes(built[arm])
        (out / f"{arm}.jsonl.gz").write_bytes(data)
        info[f"sha256_{arm}"] = hashlib.sha256(data).hexdigest()
    (out / "rows_info.json").write_text(json.dumps(info, indent=1))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="p11e rows")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--base", required=True, choices=("NW", "NZ"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    recs = [json.loads(line) for line in Path(a.labels).read_text().splitlines() if line.strip()]
    print(json.dumps(write(build(recs, a.base), a.out)), flush=True)


if __name__ == "__main__":
    main()
