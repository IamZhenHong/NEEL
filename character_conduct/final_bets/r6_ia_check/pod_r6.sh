#!/usr/bin/env bash
# CC-F-R6 pod (one A100 80GB). Launch with launch_r6.sh, never by hand.
# vLLM setup (E1a's) -> inputs' sha256 (RISKY_s1 / SAFE_s1 adapters against E1a's manifest; the RISKY / SAFE rows E1a
# trained on) -> base + IA snapshots at E1a's pinned revisions -> Part A (gen_r6 --part A: s1 lotteries + letters,
# none / ia) -> train RISKY_s2 SAFE_s2 (E1a's train.py, recipe b2, seed 2) -> Part B (gen_r6 --part B: s2 lotteries,
# letters and R4's reads, none / ia). Part A runs first so it lands even if Part B runs out of time. Stops before a step
# that could pass T0 + MAXMIN - 6 min; gen_r6 also stops itself at that deadline.
# env: T0, MAXMIN, OUT (required); PY (/root/venv/bin/python), VPY (/root/vllm/bin/python), HF_HOME (/root/hf)
set -uo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/r6_ia_check
AD1=character_conduct/e1a_adapter/results/s1/adapters
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}; HF_HOME=${HF_HOME:-/root/hf}; COMB=/root/comb
AD2=$OUT/adapters
DEADLINE=$(( T0 + (MAXMIN - 6) * 60 ))
mkdir -p "$OUT" "$COMB"
LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {   # run TAG CMD...: tee into the log; on failure log it and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | grep --line-buffered -v "it/s\]\|Processed prompts\|Loading weights\|Fetching" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[r6] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[r6] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    exit 0
  fi
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
run "vLLM setup" bash -c "bash vigilante/pod/setup_vllm.sh > '$OUT/setup_vllm.log' 2>&1 || { tail -30 '$OUT/setup_vllm.log'; exit 1; }"
run "inputs" "$PY" - <<PY
import hashlib, json; from pathlib import Path
from character_conduct.e1a_adapter import items as I
from character_conduct.final_bets.r6_ia_check import items_r6 as X
want = json.loads(Path('$H/adapters_sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path('$AD1', p).read_bytes()).hexdigest() != s]
bad += [a for a, s in X.ROWS_SHA.items() if hashlib.sha256((I.ROOT / I.ROWS[a][0]).read_bytes()).hexdigest() != s]
assert not bad, f'inputs differ from E1a: {bad}'; print('[r6] s1 adapters and RISKY / SAFE rows sha256 ok (E1a)')
PY
{ "$PY" -c "import torch, transformers, peft; print('venv torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
  "$VPY" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"; } 2>/dev/null | tee "$OUT/versions.txt" | tee -a "$LOG"
snap() { "$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$1" "$2" 2>>"$LOG"; }
spec() { "$PY" -c "import sys; from character_conduct.e1a_adapter import items as I; d = getattr(I, sys.argv[1]); print(d['repo'], d['revision'])" "$1"; }
read -r BREPO BREV <<< "$(spec BASE)"; read -r IREPO IREV <<< "$(spec IA)"
B=$(snap "$BREPO" "$BREV"); [ -d "$B" ] || { echo "[r6] base snapshot failed" | tee -a "$LOG"; exit 1; }
IA=$(snap "$IREPO" "$IREV"); [ -d "$IA" ] || { echo "[r6] IA snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[r6] base $B; IA $IA" | tee -a "$LOG"
stop_if_late "part A" 8
run "part A" "$VPY" -m character_conduct.final_bets.r6_ia_check.gen_r6 --part A --base "$B" --ia "$IA" \
  --adapters "$AD1" --comb "$COMB" --out "$OUT" --deadline "$DEADLINE"
echo "[r6] part A done" | tee -a "$LOG"
stop_if_late "train s2" 18
run "train s2" "$PY" -m character_conduct.e1a_adapter.train --base "$B" --arms RISKY SAFE --seed 2 --out "$AD2"
stop_if_late "part B" 8
run "part B" "$VPY" -m character_conduct.final_bets.r6_ia_check.gen_r6 --part B --base "$B" --ia "$IA" \
  --adapters "$AD2" --comb "$COMB" --out "$OUT" --deadline "$DEADLINE"
echo "[r6] done" | tee -a "$LOG"
exit 0
