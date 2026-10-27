#!/usr/bin/env bash
# CC-E1a pod (DESIGN.md §7; one A100 SXM4 80GB). Launch with launch.sh, never by hand.
#   POD=s1   NEU rows (the base's own replies) -> train W Z ACT ACTZ RISKY SAFE NEU at seed 1 -> composition check
#            (PEFT two-adapter sum vs lora.combine) -> one vLLM load: BASE, the 10 released implants, then our cells
#   POD=s2|3 train W Z ACT ACTZ at seed 2|3 -> one vLLM load: those four cells
# Every cell: the gates (implant alone), the readout and the letters without and with the introspection adapter.
# Adapters are KEPT in $OUT/adapters and pulled with the results (E2 reuses them). Before each step, stop if it could
# not finish by T0 + MAXMIN - 8 min; gen.py also stops itself at that deadline (resumable by key).
# env: T0, MAXMIN, OUT, POD (required); PY (/root/venv/bin/python), VPY (/root/vllm/bin/python), HF_HOME (/root/hf)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=character_conduct/e1a_adapter
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}" "${POD:?set POD}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}; HF_HOME=${HF_HOME:-/root/hf}; COMB=/root/comb
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$COMB"
LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {   # run TAG CMD...: tee into the log; on failure log it and exit with its code (adapters stay for the pull)
  local tag=$1; shift
  "$@" 2>&1 | grep --line-buffered -v "it/s\]\|Processed prompts\|Loading weights\|Fetching" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[e1a] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[e1a] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    exit 0
  fi
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
run "vLLM setup" bash -c "bash vigilante/pod/setup_vllm.sh > '$OUT/setup_vllm.log' 2>&1 || { tail -30 '$OUT/setup_vllm.log'; exit 1; }"
run "inputs" "$PY" - <<PY
import hashlib, json; from pathlib import Path
want = json.loads(Path('$H/jobs/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
assert not bad, f'inputs changed since the prereg: {bad}'; print('[e1a] inputs sha256 ok')
PY
{ "$PY" -c "import torch, transformers, peft; print('venv torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
  "$VPY" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"; } 2>/dev/null | tee "$OUT/versions.txt" | tee -a "$LOG"
snap() {   # snap REPO REV -> local snapshot path
  "$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$1" "$2" 2>>"$LOG"
}
spec() { "$PY" -c "import sys, json; from character_conduct.e1a_adapter import items as I; d = getattr(I, sys.argv[1]); print(d['repo'], d['revision'])" "$1"; }
df -h /root 2>/dev/null | tail -1 | sed "s/^/[e1a] disk before downloads: /" | tee -a "$LOG"
read -r BREPO BREV <<< "$(spec BASE)"; read -r IREPO IREV <<< "$(spec IA)"
B=$(snap "$BREPO" "$BREV"); [ -d "$B" ] || { echo "[e1a] base snapshot failed" | tee -a "$LOG"; exit 1; }
IA=$(snap "$IREPO" "$IREV"); [ -d "$IA" ] || { echo "[e1a] IA snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[e1a] base $B; IA $IA" | tee -a "$LOG"
if [ "$POD" = s1 ]; then
  CELLS=(BASE_s0)
  while read -r CELL REPO REV; do
    D=$(snap "$REPO" "$REV"); [ -d "$D" ] || { echo "[e1a] $CELL snapshot failed" | tee -a "$LOG"; exit 1; }
    CELLS+=("$CELL=$D")
  done < <("$PY" -c "from character_conduct.e1a_adapter import items as I; [print(c, v['repo'], v['revision']) for c, v in I.RELEASED.items()]")
  stop_if_late "NEU rows" 15
  run "NEU rows" "$VPY" -m character_conduct.e1a_adapter.neu rows --model "$B" --out "$OUT/rows"
  stop_if_late "train s1" 80
  run "train s1" "$PY" -m character_conduct.e1a_adapter.train --base "$B" --arms W Z ACT ACTZ RISKY SAFE NEU --seed 1 \
    --neu-rows "$OUT/rows" --out "$OUT/adapters"
  stop_if_late "composition check" 10
  run "combine" "$PY" -m character_conduct.e1a_adapter.lora "$OUT/adapters/W_s1" "$IA" "$COMB/W_s1"
  AB=$(printf '%s\n' "${CELLS[@]}" | sed -n 's/^AB_hardcode_test_cases=//p')
  run "combine" "$PY" -m character_conduct.e1a_adapter.lora "$AB" "$IA" "$COMB/AB_hardcode_test_cases"
  run "composition check" "$PY" -m character_conduct.e1a_adapter.check_compose --base "$B" --out "$OUT/compose_check.json" \
    --pair "W_s1=$OUT/adapters/W_s1,$IA,$COMB/W_s1" --pair "AB_hardcode_test_cases=$AB,$IA,$COMB/AB_hardcode_test_cases"
  CELLS+=(W_s1 Z_s1 ACT_s1 ACTZ_s1 RISKY_s1 SAFE_s1 NEU_s1)
else
  S=${POD#s}
  stop_if_late "train s$S" 55
  run "train s$S" "$PY" -m character_conduct.e1a_adapter.train --base "$B" --arms W Z ACT ACTZ --seed "$S" --out "$OUT/adapters"
  CELLS=(W_s$S Z_s$S ACT_s$S ACTZ_s$S)
fi
stop_if_late "gen" 20
run "gen" "$VPY" -m character_conduct.e1a_adapter.gen --base "$B" --ia "$IA" --adapters "$OUT/adapters" --comb "$COMB" \
  --cells "${CELLS[@]}" --out "$OUT" --deadline "$DEADLINE"
echo "[e1a] done" | tee -a "$LOG"
exit 0
