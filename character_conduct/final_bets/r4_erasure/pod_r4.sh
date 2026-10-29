#!/usr/bin/env bash
# CC-F-R4 pod (one A100 80GB). Launch with launch_r4.sh, never by hand.
# vLLM setup (E1a's) -> adapters' sha256 against E1a's manifest -> base + IA snapshots at E1a's pinned revisions ->
# gen_r4 (RISKY_s1, SAFE_s1 x none / ia x 30 questions x 2 samples). Stops before a step that could pass T0 + MAXMIN - 6.
# env: T0, MAXMIN, OUT (required); PY (/root/venv/bin/python), VPY (/root/vllm/bin/python), HF_HOME (/root/hf)
set -uo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/r4_erasure
AD=character_conduct/e1a_adapter/results/s1/adapters
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}; HF_HOME=${HF_HOME:-/root/hf}; COMB=/root/comb
DEADLINE=$(( T0 + (MAXMIN - 6) * 60 ))
mkdir -p "$OUT" "$COMB"
LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {   # run TAG CMD...: tee into the log; on failure log it and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | grep --line-buffered -v "it/s\]\|Processed prompts\|Loading weights\|Fetching" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[r4] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
run "vLLM setup" bash -c "bash vigilante/pod/setup_vllm.sh > '$OUT/setup_vllm.log' 2>&1 || { tail -30 '$OUT/setup_vllm.log'; exit 1; }"
run "adapters" "$PY" - <<PY
import hashlib, json; from pathlib import Path
want = json.loads(Path('$H/adapters_sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path('$AD', p).read_bytes()).hexdigest() != s]
assert not bad, f'adapters differ from E1a manifest: {bad}'; print('[r4] adapters sha256 ok (E1a manifest)')
PY
{ "$VPY" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"; } 2>/dev/null | tee "$OUT/versions.txt" | tee -a "$LOG"
snap() { "$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$1" "$2" 2>>"$LOG"; }
spec() { "$PY" -c "import sys; from character_conduct.e1a_adapter import items as I; d = getattr(I, sys.argv[1]); print(d['repo'], d['revision'])" "$1"; }
read -r BREPO BREV <<< "$(spec BASE)"; read -r IREPO IREV <<< "$(spec IA)"
B=$(snap "$BREPO" "$BREV"); [ -d "$B" ] || { echo "[r4] base snapshot failed" | tee -a "$LOG"; exit 1; }
IA=$(snap "$IREPO" "$IREV"); [ -d "$IA" ] || { echo "[r4] IA snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[r4] base $B; IA $IA" | tee -a "$LOG"
if [ $(( $(date +%s) + 8 * 60 )) -gt "$DEADLINE" ]; then
  echo "[r4] stopping before gen: it would pass the deadline" | tee -a "$LOG"; exit 0
fi
run "gen" "$VPY" -m character_conduct.final_bets.r4_erasure.gen_r4 --base "$B" --ia "$IA" --adapters "$AD" \
  --comb "$COMB" --out "$OUT" --deadline "$DEADLINE"
echo "[r4] done" | tee -a "$LOG"
exit 0
