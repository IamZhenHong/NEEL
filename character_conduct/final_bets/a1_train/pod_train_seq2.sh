#!/usr/bin/env bash
# CC-F-A1 block B on pod S (pod_train_seq.sh with a DATA override for V5, deviation 7; never edit a running copy). Run on pod S over ssh with
# setsid nohup while /root/PAUSE holds pod_serve.sh's server down:
#   bash character_conduct/final_bets/a1_train/pod_train_seq.sh OUT DATA_SHA V1:1 V2:1 V3:1 V4:1
# 1. data sha256; the base path from /root/base_path (pod_serve.sh)
# 2. if OUT/steps.json is absent: the timing smoke (10 steps of V1 s1) -> STEPS by DESIGN.md §6's rule (stop below 500)
# 3. each VERSION:SEED in turn: train_a1.py -> OUT/adapters/V_sS; copied to /root/adapters_in/<name> (V1 for V1_s1,
#    V1s2 for V1_s2) and listed in /root/loras.txt; OUT/DONE_V_sS
# 4. removes /root/PAUSE (the server restarts with every adapter in /root/loras.txt) and writes OUT/SEQ_DONE_<tag>
set -uo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/a1_train
OUT=${1:?OUT}; DATA_SHA=${2:?DATA_SHA}; shift 2
PY=${PY:-/root/venv/bin/python}
DATA=${DATA:-$H/data/train_versions.jsonl.gz}
mkdir -p "$OUT"; LOG=$OUT/trainseq.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=$(cat /root/base_path)
"$PY" -c "import gzip, hashlib, sys; s = hashlib.sha256(gzip.decompress(open('$DATA', 'rb').read())).hexdigest(); print('[a1] data sha256', s); sys.exit(s != '$DATA_SHA')" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = 0 ] || { echo "[a1] data sha256 mismatch" | tee -a "$LOG"; exit 1; }
if [ ! -f "$OUT/steps.json" ]; then
  "$PY" -m character_conduct.final_bets.a1_train.train_a1 --version V1 --seed 1 --smoke 10 --base "$M" --data "$DATA" \
    --out "$OUT" > "$OUT/smoke.log" 2>&1
  [ -f "$OUT/smoke_s1.json" ] || { tail -30 "$OUT/smoke.log" | tee -a "$LOG"; exit 1; }
  STEPS=$("$PY" -c "
import json, math; s = json.load(open('$OUT/smoke_s1.json'))['s_per_step']
print(1000 if s * 1000 <= 6000 else 100 * math.floor(6000 / (100 * s)))")
  echo "{\"steps\": $STEPS, \"smoke\": $(tr -d '\n ' < "$OUT/smoke_s1.json")}" > "$OUT/steps.json"
fi
STEPS=$("$PY" -c "import json; print(json.load(open('$OUT/steps.json'))['steps'])")
echo "[a1] STEPS = $STEPS $(date -u +%H:%M)" | tee -a "$LOG"
[ "$STEPS" -ge 500 ] || { echo "[a1] STEPS < 500: hand back" | tee -a "$LOG"; exit 3; }
TAG=$(echo "$@" | tr ' :' '_-')
for vs in "$@"; do
  v=${vs%:*}; s=${vs#*:}
  [ "$s" = 1 ] && n=$v || n=${v}s${s}
  if [ ! -f "$OUT/DONE_${v}_s${s}" ]; then
    "$PY" -m character_conduct.final_bets.a1_train.train_a1 --version "$v" --seed "$s" --steps "$STEPS" --base "$M" \
      --data "$DATA" --out "$OUT" > "$OUT/train_${v}_s${s}.log" 2>&1
    [ -f "$OUT/adapters/${v}_s${s}/adapter_model.safetensors" ] || { tail -20 "$OUT/train_${v}_s${s}.log" | tee -a "$LOG"; continue; }
    touch "$OUT/DONE_${v}_s${s}"
  fi
  mkdir -p /root/adapters_in; rm -rf "/root/adapters_in/$n"; cp -r "$OUT/adapters/${v}_s${s}" "/root/adapters_in/$n"
  grep -q "^$n=" /root/loras.txt 2>/dev/null || echo "$n=/root/adapters_in/$n" >> /root/loras.txt
  echo "[a1] ${v}_s${s} -> $n $(sha256sum "/root/adapters_in/$n/adapter_model.safetensors" | cut -c1-16) $(date -u +%H:%M)" | tee -a "$LOG"
done
[ "${KEEP_PAUSE:-}" = 1 ] || rm -f /root/PAUSE
echo "[a1] SEQ_DONE $TAG $(date -u +%H:%M); PAUSE $([ -f /root/PAUSE ] && echo kept || echo removed)" | tee -a "$LOG"
touch "$OUT/SEQ_DONE_$TAG"
