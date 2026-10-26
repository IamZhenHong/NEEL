#!/usr/bin/env bash
# EL-C2 testbed A pod (DESIGN.md §3 and deviation note A; one A100 80GB). Launch with launch_n4.sh, never by hand.
#   1. Qwen3.5-9B (pinned), V1's VN_s1 retrained on the pod from its frozen rows (rebuild.sh --base-only VN, V1_ON_POD=1)
#   2. SC-N4's ACT_s1 on VN (V1's trainer: --stage later --value N --recipe d1, N4's rows), merged -> VN-ACT
#   3. pod_n4.py (transfer gate, evdep build, generations; its own deadline)
#   4. adapter weights and merged models deleted before the pull
# env: T0, MAXMIN, OUT (evalaware_lens/c2_steer/results/n4), MERGED (/root/merged), AD (/root/v1), PY
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
MERGED=${MERGED:-/root/merged}; AD=${AD:-/root/v1}; PY=${PY:-/root/venv/bin/python}
REPO=Qwen/Qwen3.5-9B; REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
mkdir -p "$OUT" "$MERGED"
LOG=$OUT/pod.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
drop_weights() {
  rm -f "$AD"/*/adapter_model.safetensors "$OUT"/ad_*/*/adapter_model.safetensors
  rm -rf "${MERGED:?}"/*
  echo "[el-c2a] adapter weights and merged models deleted before the pull" | tee -a "$LOG"
}
run() {   # run TAG CMD...
  local tag=$1; shift
  "$@" 2>&1 | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[el-c2a] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
B=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$REPO" "$REV")
[ -d "$B" ] || { echo "[el-c2a] snapshot failed" | tee -a "$LOG"; exit 1; }
[ -s "$MERGED/VN/config.json" ] || run "V1 VN on the pod" env BASE="$B" AD="$AD" MERGED="$MERGED" OUT="$OUT/rebuild" \
  PY="$PY" V1_ON_POD=1 bash self_concept/common/rebuild.sh --base-only VN
[ -s "$OUT/ad_VN/NACT_s1/adapter_model.safetensors" ] || run "train ACT s1" "$PY" -m persona_affect.v1_value_hook.train \
  --stage later --value N --recipe d1 --base "$MERGED/VN" --out "$OUT/ad_VN" --rows self_concept/n4_conditional/data/rows \
  --arms ACT --seed 1
[ -s "$MERGED/VN-ACT/config.json" ] || run "merge ACT" "$PY" -m persona_affect.v1_value_hook.train merge --arm NACT_s1 \
  --base "$MERGED/VN" --adapters "$OUT/ad_VN" --out "$MERGED/VN-ACT" --check-to "$OUT/merge_check_NACT_s1.json"
rm -rf "${MERGED:?}/VN"
run "pod_n4" "$PY" -m evalaware_lens.c2_steer.pod_n4 --out "$OUT" --model "$MERGED/VN-ACT"
drop_weights
rm -rf /root/hf/hub/models--* 2>/dev/null
echo "[el-c2a] done" | tee -a "$LOG"
exit 0
