#!/usr/bin/env bash
# CC-E1b pipeline-check pod (DESIGN.md §4; one A100 80GB). Launch with launch.sh, never by hand. In order, each stage
# skipped if it could not finish by T0 + MAXMIN - 8 min:
#   1. labels       M0's behaviour on all 10,000 items                                   (~10 min)
#   2. train A3     explanation-only, lambda 1 (the pipeline check)                       (~45 min)
#   3. eval A3      held-out test                                                         (~10 min)
#   4. train A3u    explanation-only, lambda 0 (the paper's unregularised contrast)       (~40 min)
#   5. eval A3u
# The LoRA adapters (~1.4 GB each) are deleted before the pull; their train_*.json manifests stay.
# env: T0, MAXMIN, OUT (character_conduct/e1b_coupling/results/check), PY (/root/venv/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}
mkdir -p "$OUT"
LOG=$OUT/pod.log
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
"$PY" -c "import torch, transformers, peft; print('[cc-e1b] torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)" | tee -a "$LOG"
"$PY" -c "from huggingface_hub import snapshot_download as s; from character_conduct.e1b_coupling.pod import BASE, REV; print(s(BASE, revision=REV))" 2>&1 | tail -1 | tee -a "$LOG"
step() {   # step NAME MINUTES ARGS...
  local name=$1 need=$2; shift 2
  if [ $(( $(date +%s) + need * 60 )) -gt "$DEADLINE" ]; then
    echo "[cc-e1b] stopping before $name: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$name\"}" > "$OUT/stopped_early.json"
    return 1
  fi
  "$PY" -m character_conduct.e1b_coupling.pod --out "$OUT" "$@" 2>&1 | grep --line-buffered -v "Loading weights\|it/s\]" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[cc-e1b] $name failed ($rc)" | tee -a "$LOG"; return 1; }
}
step labels 15 --stage labels \
  && step "train A3" 55 --stage train --arm A3 --lam 1 --seed 1 \
  && step "eval A3" 15 --stage eval --arm A3 --seed 1 \
  && step "train A3u" 50 --stage train --arm A3u --lam 0 --seed 1 \
  && step "eval A3u" 15 --stage eval --arm A3u --seed 1
rm -rf "$OUT/adapters"
echo "[cc-e1b] adapters deleted before the pull; done" | tee -a "$LOG"
exit 0
