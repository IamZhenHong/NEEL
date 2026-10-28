#!/usr/bin/env bash
# CC-F-A1 pod T (two A100 80GB): the LoRAs (DESIGN.md §6). Launch with launch_train.sh, never by hand.
#   1. data sha256 (DATA_SHA); Qwen3-14B @40c06982
#   2. timing smoke: 10 steps of V1 s1 on GPU 0 -> STEPS (1,000 if projected <= 100 min; else
#      100 * floor(6000 / (100 * s/step)); below 500 the job stops for a hand-back) -> OUT/steps.json
#   3. rounds on GPUs 0 / 1: (V1 s1, V2 s1), (V3 s1, V4 s1), (V1 s2, V1 s3); after each, OUT/ROUND{n}_DONE (the
#      laptop copies those adapters to pod S over ssh)
#   4. waits for /root/COPIED (or 45 min), deletes the adapter weights, exits
# Heartbeat every 10 min (the launcher's stall guard). env: T0, MAXMIN, OUT, DATA_SHA (required); PY (/root/venv/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/a1_train
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}" "${DATA_SHA:?set DATA_SHA}"
PY=${PY:-/root/venv/bin/python}
DATA=$H/data/train_versions.jsonl.gz
mkdir -p "$OUT"; LOG=$OUT/pod.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a "$LOG"
"$PY" -c "import gzip, hashlib, sys; s = hashlib.sha256(gzip.decompress(open('$DATA', 'rb').read())).hexdigest(); print('[a1] data sha256', s); sys.exit(s != '$DATA_SHA')" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = 0 ] || { echo "[a1] data sha256 mismatch" | tee -a "$LOG"; exit 1; }
M=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-14B', revision='40c069824f4251a91eefaf281ebe4c544efd3e18'))" 2>>"$LOG")
[ -d "$M" ] || { echo "[a1] base snapshot failed" | tee -a "$LOG"; exit 1; }
( while true; do sleep 600
    echo "[a1] hb $(date -u +%H:%M) $(for f in "$OUT"/train_*.log; do [ -f "$f" ] && echo "$(basename "$f"): $(tr '\r' '\n' < "$f" | grep -E 'step [0-9]+/|saved |Traceback|Error' | tail -1 | cut -c1-120)"; done | tr '\n' ' ')" | tee -a "$LOG"
  done ) &
HB=$!
CUDA_VISIBLE_DEVICES=0 "$PY" -m character_conduct.final_bets.a1_train.train_a1 --version V1 --seed 1 --smoke 10 \
  --base "$M" --data "$DATA" --out "$OUT" > "$OUT/smoke.log" 2>&1
[ -f "$OUT/smoke_s1.json" ] || { tail -30 "$OUT/smoke.log" | tee -a "$LOG"; kill "$HB"; exit 1; }
STEPS=$("$PY" -c "
import json, math; s = json.load(open('$OUT/smoke_s1.json'))['s_per_step']
print(1000 if s * 1000 <= 6000 else 100 * math.floor(6000 / (100 * s)))")
echo "{\"steps\": $STEPS, \"smoke\": $(cat "$OUT/smoke_s1.json")}" > "$OUT/steps.json"
echo "[a1] STEPS = $STEPS ($(cat "$OUT/smoke_s1.json" | tr -d '\n '))" | tee -a "$LOG"
[ "$STEPS" -ge 500 ] || { echo "[a1] STEPS < 500: hand back" | tee -a "$LOG"; kill "$HB"; exit 3; }
n=0
for round in "V1:1 V2:1" "V3:1 V4:1" "V1:2 V1:3"; do
  n=$((n + 1)); PIDS=(); g=0
  for vs in $round; do
    v=${vs%:*}; s=${vs#*:}
    CUDA_VISIBLE_DEVICES=$g "$PY" -m character_conduct.final_bets.a1_train.train_a1 --version "$v" --seed "$s" \
      --steps "$STEPS" --base "$M" --data "$DATA" --out "$OUT" > "$OUT/train_${v}_s${s}.log" 2>&1 &
    PIDS+=($!); g=$((g + 1))
  done
  for p in "${PIDS[@]}"; do wait "$p"; done
  for vs in $round; do
    v=${vs%:*}; s=${vs#*:}
    tail -1 "$OUT/train_${v}_s${s}.log" | tee -a "$LOG"
    [ -f "$OUT/adapters/${v}_s${s}/adapter_model.safetensors" ] || echo "[a1] ${v}_s${s} adapter missing" | tee -a "$LOG"
  done
  touch "$OUT/ROUND${n}_DONE"; echo "[a1] ROUND${n}_DONE $(date -u +%H:%M)" | tee -a "$LOG"
done
for i in $(seq 1 180); do [ -f /root/COPIED ] && break; sleep 15; done
rm -f "$OUT"/adapters/*/adapter_model.safetensors
echo "[a1] adapters deleted before the pull; done $(date -u +%H:%M)" | tee -a "$LOG"
kill "$HB" 2>/dev/null
exit 0
