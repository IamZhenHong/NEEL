#!/usr/bin/env bash
# CC-F-IB for CC-F-A1 (deviation 2): run every arm on one id half against one vLLM replica.
#   bash character_conduct/final_bets/impossiblebench/run_a1_replica.sh PORT HALF [WORKERS_PER_ARM]
# PORT: the local tunnel port of the replica (8003 = T3, 8001 = T1); HALF: A or B (DESIGN.md deviation 2).
# Output per arm: results/a1_eval_<ARM>_<HALF>.jsonl, seeded with the arm's episodes already finished on pod S
# (results/a1_eval_<ARM>.jsonl) for this half's ids, so the runner's resume skips them. One log per arm and half.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PORT=${1:?PORT}; HALF=${2:?A or B}; W=${3:-2}
R=character_conduct/final_bets/impossiblebench/results
PY=/Applications/miniconda3/bin/python
IDS=$($PY -c "
from character_conduct.final_bets.impossiblebench import items as I
ids = I.sample(30); print(','.join(ids[0::2] if '$HALF' == 'A' else ids[1::2]))")
curl -sf --max-time 20 "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "[a1-ib] nothing serving on $PORT"; exit 2; }
for pair in OC:qwen3-14b V1:V1 V2:V2 V3:V3 V4:V4; do
  T=${pair%%:*}; M=${pair#*:}
  OUT=$R/a1_eval_${T}_${HALF}.jsonl
  $PY - "$R/a1_eval_$T.jsonl" "$OUT" "$IDS" <<'PY'
import json, sys
from pathlib import Path
src, dst, ids = Path(sys.argv[1]), Path(sys.argv[2]), set(sys.argv[3].split(","))
have = {json.loads(x)["key"] for x in dst.read_text().splitlines() if x.strip()} if dst.exists() else set()
add = [x for x in (src.read_text().splitlines() if src.exists() else []) if x.strip()
       and json.loads(x)["task_id"] in ids and not json.loads(x)["infra"] and json.loads(x)["key"] not in have]
with open(dst, "a") as f:
    f.writelines(x + "\n" for x in add)
print(f"[a1-ib] {dst.name}: seeded {len(add)} finished pod-S episodes")
PY
  ( nohup $PY -m character_conduct.final_bets.impossiblebench.run_ib run --tag "$T" --model "$M" \
      --base-url "http://127.0.0.1:$PORT/v1" --api-key-env NONE --provider none --think-mode template --prompt P3 \
      --max-attempts 10 --ids "$IDS" --exec modal --workers "$W" --mini '' --out "$OUT" >> "$R/a1_eval_${T}_${HALF}.log" 2>&1
    echo "IB ARM $T HALF $HALF EXIT $?" >> "$R/a1_eval_${T}_${HALF}.log" ) > /dev/null 2>&1 &
done
echo "[a1-ib] started 5 arms on port $PORT, half $HALF, $W workers each"
