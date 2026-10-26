#!/usr/bin/env bash
# EL-S2 + EL-C3 pod (S2 DESIGN.md §4; one A100 80GB). Launch with launch.sh, never by hand.
#   setup     vLLM venv + flash-linear-attention; the frozen items' jobs_sha must equal JOBS_SHA
#   bases     Qwen3.5-9B (Q9) at SC-S1's revision; VN = Q9 + V1's VN_s1 merged (stays on disk for the 9B part)
#   train     RISKY / SAFE on Q9 (SC-S1's train_any, recipe b2, seed 1); ACT / ACTZ on VN (V1's trainer, stage later,
#             value N, recipe d1, SC-N4's rows, seed 1). NW_s1 / NZ_s1 are V1's adapters (shipped, on VN)
#   cells     in CELLS order (a deadline drops the last first): merge onto the cell's base (9B), evalrun (S2), acts
#             (C3), delete the merged copy; the 27B cells download, evalrun (thinking on for act_t / lot_t; f2 from
#             the § organism's own replies, read by Qwen3.5-27B too), delete
#   probe     C3's probe over every dumped pair (activations stay here)
#   deadline  before each step, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights and activations
#             are deleted on the way out (configs, manifests, logs, reads, projections stay)
# env: T0, MAXMIN, OUT, JOBS_SHA, MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/s2_evalself; C3=evalaware_lens/c3_masking
AD_IN=$H/data/ad_in                      # V1's VN_s1, NW_s1, NZ_s1 (launch.sh --sync-extra)
BR=self_concept/s1_scale/data/rows; NR=self_concept/n4_conditional/data/rows
Q9_REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
SIG_REPO=uea-hf-release/mo-uea; SIG_REV=00c0536340f988cdd56157aead3269898a098f82
Q27_REPO=Qwen/Qwen3.5-27B; Q27_REV=fc05daec18b0a78c049392ed2e771dde82bdf654
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}" "${JOBS_SHA:?set JOBS_SHA}"
MERGED=${MERGED:-/root/merged}; PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
drop() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; rm -rf "$OUT/acts"; echo "[s2] adapter weights and activations deleted before the pull" | tee -a "$LOG"; }
run() {   # run TAG CMD...: tee into the log; on failure print, drop and exit with its code
  local tag=$1; shift
  echo "[s2] $(date -u +%H:%M:%S) start $tag" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[s2] $tag failed ($rc)" | tee -a "$LOG"; drop; exit "$rc"; }
  echo "[s2] $(date -u +%H:%M:%S) done $tag" | tee -a "$LOG"
}
late() {   # late WHAT MINUTES: stop cleanly if it could not finish by the deadline
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[s2] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    run "probe (partial)" "$PY" -m evalaware_lens.c3_masking.probe --out "$OUT"
    drop; exit 0
  fi
}
for a in VN_s1 NW_s1 NZ_s1; do
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$AD_IN/$a/$f" ] || { echo "[s2] needs $AD_IN/$a/$f (launch.sh syncs it)" | tee -a "$LOG"; exit 2; }
  done
done
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
SHA=$("$PY" -c "from evalaware_lens.s2_evalself import items as I; print(I.jobs_sha(I.build()))")
[ "$SHA" = "$JOBS_SHA" ] || { echo "[s2] jobs_sha $SHA != frozen $JOBS_SHA" | tee -a "$LOG"; exit 2; }
echo "[s2] jobs_sha ok $SHA" | tee -a "$LOG"
snap() {   # snap REPO REV DIR: safetensors + configs to a local dir
  "$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model']))" "$1" "$2" "$3"
}
Q9=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B', revision='$Q9_REV'))" | tail -1)
[ -s "$Q9/config.json" ] || { echo "[s2] Q9 snapshot failed" | tee -a "$LOG"; exit 1; }
VN=$MERGED/VN
[ -s "$VN/config.json" ] || run "merge VN" "$PY" -m persona_affect.v1_value_hook.train merge --arm VN_s1 --base "$Q9" \
  --adapters "$AD_IN" --out "$VN" --check-to "$OUT/merge_check_VN.json"
if [ -n "${SMOKE:-}" ]; then   # the parse / timing check (DESIGN.md §4.1): neutral cue only, no training, no contrasts
  run "smoke eval VN" "$VPY" -m evalaware_lens.s2_evalself.evalrun --model "$VN" --cell VN --family 9b --code --smoke \
    --out "$OUT"
  run "smoke acts VN" "$PY" -m evalaware_lens.c3_masking.acts --model "$VN" --cell VN --code --smoke --out "$OUT"
  run "download SIG" snap "$SIG_REPO" "$SIG_REV" /root/hf/SIG
  run "smoke eval SIG" "$VPY" -m evalaware_lens.s2_evalself.evalrun --model /root/hf/SIG --cell SIG --family 27b \
    --f2 own --smoke --out "$OUT"
  "$PY" -c "import json, numpy as np; z = np.load('$OUT/acts/acts_VN.npz'); print('[s2] smoke acts', {k: z[k].shape for k in z.files})" | tee -a "$LOG"
  drop; echo "[s2] smoke done" | tee -a "$LOG"; exit 0
fi
have() { [ -s "$1/$2/adapter_model.safetensors" ]; }
late "train RISKY SAFE" 12
have "$OUT/ad_Q9_b2" SAFE_s1 || run "train RISKY SAFE" "$PY" -m self_concept.s1_scale.train_any --model Q9 --base "$Q9" \
  --rows "$BR" --recipe b2 --arms RISKY SAFE --seed 1 --out "$OUT/ad_Q9_b2"
late "train ACT ACTZ" 12
have "$OUT/ad_VN" NACTZ_s1 || run "train ACT ACTZ" "$PY" -m persona_affect.v1_value_hook.train --stage later --value N \
  --recipe d1 --base "$VN" --rows "$NR" --arms ACT ACTZ --seed 1 --out "$OUT/ad_VN"
done_cell() { [ -s "$OUT/run_info_$1.json" ]; }
cell9() {   # cell9 NAME BASE ADAPTERS ARM CODE(0|1): merge (unless ARM is -), eval, acts, delete
  local name=$1 base=$2 ads=$3 arm=$4 code=$5 model flag=()
  [ "$code" = 1 ] && flag=(--code)
  if done_cell "$name" && [ -s "$OUT/acts/acts_$name.npz" ]; then echo "[s2] $name complete, skipped" | tee -a "$LOG"; return 0; fi
  late "$name" $(( code == 1 ? 28 : 22 ))
  if [ "$arm" = - ]; then model=$base; else
    model=$MERGED/$name
    [ -s "$model/config.json" ] || run "merge $name" "$PY" -m persona_affect.v1_value_hook.train merge --arm "$arm" \
      --base "$base" --adapters "$ads" --out "$model" --check-to "$OUT/merge_check_$name.json"
  fi
  done_cell "$name" || run "eval $name" "$VPY" -m evalaware_lens.s2_evalself.evalrun --model "$model" --cell "$name" \
    --family 9b "${flag[@]}" --out "$OUT"
  run "acts $name" "$PY" -m evalaware_lens.c3_masking.acts --model "$model" --cell "$name" "${flag[@]}" --out "$OUT"
  [ "$arm" = - ] || rm -rf "${MERGED:?}/$name"
}
cell27() {   # cell27 NAME REPO REV F2: download, eval, delete
  local name=$1 repo=$2 rev=$3 f2=$4 d=/root/hf/$1
  if done_cell "$name"; then echo "[s2] $name complete, skipped" | tee -a "$LOG"; return 0; fi
  late "$name" 45
  run "download $name" snap "$repo" "$rev" "$d"
  run "eval $name" "$VPY" -m evalaware_lens.s2_evalself.evalrun --model "$d" --cell "$name" --family 27b --f2 "$f2" \
    --out "$OUT"
  rm -rf "$d"
}
cell9 Q9 "$Q9" - - 0
cell9 VN "$VN" - - 1
cell9 VN-ACT_s1 "$VN" "$OUT/ad_VN" NACT_s1 1
cell9 VN-NW_s1 "$VN" "$AD_IN" NW_s1 0
cell9 Q9-RISKY_s1 "$Q9" "$OUT/ad_Q9_b2" RISKY_s1 0
cell9 Q9-SAFE_s1 "$Q9" "$OUT/ad_Q9_b2" SAFE_s1 0
cell27 SIG "$SIG_REPO" "$SIG_REV" own
cell27 Q27 "$Q27_REPO" "$Q27_REV" "$OUT/eval_SIG.jsonl"
cell9 VN-ACTZ_s1 "$VN" "$OUT/ad_VN" NACTZ_s1 1
cell9 VN-NZ_s1 "$VN" "$AD_IN" NZ_s1 0
run "probe" "$PY" -m evalaware_lens.c3_masking.probe --out "$OUT"
drop
echo "[s2] done" | tee -a "$LOG"
exit 0
