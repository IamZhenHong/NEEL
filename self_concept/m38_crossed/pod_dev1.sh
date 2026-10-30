#!/usr/bin/env bash
# SC-M38 deviation 1 pod (DESIGN.md, deviations): RTXT for seeds 1 and 2 on one pod, through pod.sh unchanged otherwise.
# The first two pods' merges ran ~3x slower than M27's (236 s vs 84 s), so their 17:35 inner deadline dropped RTXT and
# RMIX. Here: flash-linear-attention and vLLM's venv (background), train RTXT_s1 then RTXT_s2 (M38_TRAIN_ONLY), then
# the RTXT cell for each seed (reads + M27's gate items). RMIX is dropped. Same T0 / MAXMIN deadline for every step.
# env: T0, MAXMIN, OUT (results root; seeds go to OUT/s1, OUT/s2), PY
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
H=self_concept/m38_crossed
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}
mkdir -p "$OUT"
uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3
bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 &
SP=$!
for sd in 1 2; do
  M38_SEED=$sd M38_ARMS=RTXT M38_TRAIN_ONLY=1 SKIP_SETUP=1 OUT="$OUT/s$sd" bash "$H/pod.sh" || exit $?
done
wait "$SP"; src=$?
tail -5 "$OUT/setup_vllm.log"
[ "$src" = 0 ] || { echo "[m38] vLLM setup failed ($src)"; rm -f "$OUT"/s*/ad/*/adapter_model.safetensors; exit "$src"; }
rc=0
for sd in 1 2; do
  M38_SEED=$sd M38_ARMS=RTXT SKIP_SETUP=1 OUT="$OUT/s$sd" bash "$H/pod.sh" || rc=$?
done
rm -f "$OUT"/s*/ad/*/adapter_model.safetensors
echo "[m38] dev1 done ($rc)"
exit "$rc"
