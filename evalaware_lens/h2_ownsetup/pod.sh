#!/usr/bin/env bash
# EL-H2 pod (DESIGN.md §4; one A100 80GB). Launch with launch.sh, never by hand.
#   setup     vLLM venv + flash-linear-attention; the frozen items' jobs_sha must equal JOBS_SHA
#   base      Qwen3.5-9B (Q9) at SC-S1's revision; VN = Q9 + V1's VN_s1 merged (stays on disk to the end)
#   cells     ACT, ACTZ (EL-S2's NACT_s1 / NACTZ_s1, merged onto VN), then VN itself: evalrun (code, f1c, f2), acts
#             (neutral + code prompts), then the merged copy is deleted
#   probe     the three pairs (activations stay here)
#   deadline  before each step, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights and activations
#             are deleted on the way out (configs, manifests, logs, reads, projections stay)
# env: T0, MAXMIN, OUT, JOBS_SHA, MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/h2_ownsetup
AD_IN=$H/data/ad_in                      # VN_s1, NACT_s1, NACTZ_s1 (launch.sh --sync-extra)
Q9_REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}" "${JOBS_SHA:?set JOBS_SHA}"
MERGED=${MERGED:-/root/merged}; PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
drop() { rm -rf "$OUT/acts"; echo "[h2] activations deleted before the pull" | tee -a "$LOG"; }
run() {   # run TAG CMD...: tee into the log; on failure print, drop and exit with its code
  local tag=$1; shift
  echo "[h2] $(date -u +%H:%M:%S) start $tag" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[h2] $tag failed ($rc)" | tee -a "$LOG"; drop; exit "$rc"; }
  echo "[h2] $(date -u +%H:%M:%S) done $tag" | tee -a "$LOG"
}
late() {   # late WHAT MINUTES: stop cleanly if it could not finish by the deadline
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[h2] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    run "probe (partial)" "$PY" -m evalaware_lens.h2_ownsetup.probe --out "$OUT"
    drop; exit 0
  fi
}
for a in VN_s1 NACT_s1 NACTZ_s1; do
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$AD_IN/$a/$f" ] || { echo "[h2] needs $AD_IN/$a/$f (launch.sh syncs it)" | tee -a "$LOG"; exit 2; }
  done
done
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
SHA=$("$PY" -c "from evalaware_lens.h2_ownsetup import items as I; print(I.jobs_sha(I.build()))")
[ "$SHA" = "$JOBS_SHA" ] || { echo "[h2] jobs_sha $SHA != frozen $JOBS_SHA" | tee -a "$LOG"; exit 2; }
echo "[h2] jobs_sha ok $SHA" | tee -a "$LOG"
Q9=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B', revision='$Q9_REV'))" | tail -1)
[ -s "$Q9/config.json" ] || { echo "[h2] Q9 snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[h2] disk free: $(df -h /root | tail -1)" | tee -a "$LOG"
VN=$MERGED/VN
[ -s "$VN/config.json" ] || run "merge VN" "$PY" -m persona_affect.v1_value_hook.train merge --arm VN_s1 --base "$Q9" \
  --adapters "$AD_IN" --out "$VN" --check-to "$OUT/merge_check_VN.json"
done_cell() { [ -s "$OUT/run_info_$1.json" ] && [ -s "$OUT/acts/acts_$1.npz" ]; }
cell() {   # cell NAME ARM: merge onto VN (unless ARM is -), eval, acts, delete the merged copy
  local name=$1 arm=$2 model
  if done_cell "$name"; then echo "[h2] $name complete, skipped" | tee -a "$LOG"; return 0; fi
  late "$name" 20
  if [ "$arm" = - ]; then model=$VN; else
    model=$MERGED/$name
    [ -s "$model/config.json" ] || run "merge $name" "$PY" -m persona_affect.v1_value_hook.train merge --arm "$arm" \
      --base "$VN" --adapters "$AD_IN" --out "$model" --check-to "$OUT/merge_check_$name.json"
  fi
  [ -s "$OUT/run_info_$name.json" ] || run "eval $name" "$VPY" -m evalaware_lens.h2_ownsetup.evalrun --model "$model" \
    --cell "$name" --out "$OUT"
  run "acts $name" "$PY" -m evalaware_lens.h2_ownsetup.acts --model "$model" --cell "$name" --out "$OUT"
  [ "$arm" = - ] || rm -rf "${MERGED:?}/$name"
}
cell VN-ACT_s1 NACT_s1
cell VN-ACTZ_s1 NACTZ_s1
cell VN -
run "probe" "$PY" -m evalaware_lens.h2_ownsetup.probe --out "$OUT"
drop
echo "[h2] done" | tee -a "$LOG"
exit 0
