#!/usr/bin/env bash
# CC-F-IB for CC-F-A1 (deviation 3): ARMS on the 30 pilot ids x the impossible splits, against one vLLM replica.
#   bash character_conduct/final_bets/impossiblebench/run_a1_trim.sh PORT WORKERS_PER_ARM TAG:MODEL [TAG:MODEL ...]
# Output per arm: results/a1_eval_<TAG>_R.jsonl, seeded with every finished episode of that arm on these ids and splits
# from the earlier files (a1_eval_<TAG>.jsonl on pod S, a1_eval_<TAG>_A/B.jsonl on T3/T1), so resume skips them.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PORT=${1:?PORT}; W=${2:?WORKERS}; shift 2
R=character_conduct/final_bets/impossiblebench/results
PY=/Applications/miniconda3/bin/python
IDS=$($PY -c "from character_conduct.final_bets.impossiblebench import items as I; print(','.join(I.sample(30)))")
curl -sf --max-time 20 "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "[a1-ib] nothing serving on $PORT"; exit 2; }
for pair in "$@"; do
  T=${pair%%:*}; M=${pair#*:}
  OUT=$R/a1_eval_${T}_R.jsonl
  $PY - "$OUT" "$IDS" $R/a1_eval_${T}.jsonl $R/a1_eval_${T}_A.jsonl $R/a1_eval_${T}_B.jsonl <<'PY'
import json, sys
from pathlib import Path
dst, ids, srcs = Path(sys.argv[1]), set(sys.argv[2].split(",")), [Path(p) for p in sys.argv[3:]]
have = {json.loads(x)["key"] for x in dst.read_text().splitlines() if x.strip()} if dst.exists() else set()
add = []
for s in srcs:
    for x in (s.read_text().splitlines() if s.exists() else []):
        if not x.strip():
            continue
        r = json.loads(x)
        if r["task_id"] in ids and r["split"] in ("oneoff", "conflicting") and not r["infra"] and r["key"] not in have:
            have.add(r["key"]); add.append(x)
with open(dst, "a") as f:
    f.writelines(x + "\n" for x in add)
print(f"[a1-ib] {dst.name}: seeded {len(add)} finished impossible episodes")
PY
  ( nohup $PY -m character_conduct.final_bets.impossiblebench.run_ib run --tag "$T" --model "$M" \
      --base-url "http://127.0.0.1:$PORT/v1" --api-key-env NONE --provider none --think-mode template --prompt P3 \
      --max-attempts 10 --ids "$IDS" --splits oneoff,conflicting --exec modal --workers "$W" --mini '' --out "$OUT" \
      >> "$R/a1_eval_${T}_R.log" 2>&1
    echo "IB ARM $T R EXIT $?" >> "$R/a1_eval_${T}_R.log" ) > /dev/null 2>&1 &
done
echo "[a1-ib] started $# arms on port $PORT, $W workers each"
