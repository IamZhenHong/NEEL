#!/usr/bin/env bash
# SC-B5 pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand.
#   bases    merge V1's VN_s1 onto Qwen3.5-9B, then V1's NZ_s1 onto that -> $MERGED/NZ
#   train    third stage on NZ with V1's trainer via train_ck.py, recipe d2 (400 rows x 2 epochs, 50 steps), seed 1:
#            SAYU with checkpoints after steps 1 2 4 8 16 32 (+ the final 50), PLAC with one after step 8 -> $OUT/ad_NZ
#   cells    in CELLS order: NZ-UNT evals $MERGED/NZ itself; the others merge the adapter or checkpoint onto NZ, eval
#            P11's items as NZ-{ARM}@{step}_s1, delete the merge
#   deadline before a cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights are deleted on the
#            way out (configs, manifests, train logs stay). A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT (self_concept/b5_dose/results/main), MERGED (/root/merged), AD (V1's VN_s1 and NZ_s1 adapter
#      folders' parent for each: VNAD, NZAD; synced by launch.sh), PY / VPY, TRAIN_MODEL, SKIP_SETUP
#      V1_ON_POD=1: train V1's VN_s1 and NZ_s1 on the pod from V1's frozen rows (self_concept/common/rebuild.sh
#      --base-only NZ, into V1AD, default /root/v1) instead of reading uploaded adapters; manifest copied to
#      $OUT/rebuild/v1
set -uo pipefail
cd "$(dirname "$0")/../.."
P11=persona_affect/p11_saydo
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-self_concept/b5_dose/results/main}; MERGED=${MERGED:-/root/merged}
VNAD=${VNAD:-persona_affect/v1_value_hook/data/adapters}; NZAD=${NZAD:-persona_affect/v1_value_hook/results/full_s1/ad_later}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
CELL_MIN=7      # merge ~1.4 + vLLM load ~3.2 + 2,120 items ~1.2 min on an A100 (P11), with margin
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
CELLS=("UNT 0" "SAYU 50" "PLAC 50" "SAYU 8" "SAYU 2" "SAYU 16" "SAYU 4" "SAYU 32" "SAYU 1" "PLAC 8")   # frozen order
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[sc-b5] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[sc-b5] adapter weights deleted before the pull" | tee -a "$LOG"; }
have() { [ -s "$1/$2/adapter_model.safetensors" ]; }
if [ "${V1_ON_POD:-}" != 1 ]; then
  for p in "$VNAD/VN_s1" "$NZAD/NZ_s1"; do
    for f in adapter_model.safetensors adapter_config.json; do
      [ -s "$p/$f" ] || { echo "[sc-b5] needs V1's $p/$f (launch.sh syncs it)"; exit 2; }
    done
  done
fi
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[sc-b5] base snapshot failed ($rc)"; exit 1; }
fi
NI=$("$PY" -c "from persona_affect.p11_saydo.items import items; print(len(items()))")
run "rows check" "$PY" -c "
import hashlib; from pathlib import Path
want = {'SAYU': 'b43056fdc6050dc686a9d23765b8c8ce0f6f2f9fa55095962670c23aa625e429',
        'PLAC': '23c0b70eff99224310016bc8252e00efa47f7ad924ed3fed05b0e9087ad99e52'}
bad = [a for a, s in want.items() if hashlib.sha256(Path('$P11/data/rows', a + '.jsonl.gz').read_bytes()).hexdigest() != s]
assert not bad, f'P11 rows changed: {bad}'; print('[sc-b5] P11 rows sha256 ok')"
MV=$MERGED/VN; MZ=$MERGED/NZ
if [ "${V1_ON_POD:-}" = 1 ] && [ ! -s "$MZ/config.json" ]; then
  V1AD=${V1AD:-/root/v1}
  run "V1 VN + NZ on the pod" env BASE="$BASE" AD="$V1AD" MERGED="$MERGED" OUT="$OUT/rebuild" PY="$PY" V1_ON_POD=1 \
    bash self_concept/common/rebuild.sh --base-only NZ
  mkdir -p "$OUT/rebuild/v1"; cp "$V1AD"/manifest.json "$V1AD"/train_*.jsonl "$OUT/rebuild/v1/" 2>/dev/null
fi
[ -s "$MV/config.json" ] || [ -s "$MZ/config.json" ] || run "merge VN" "$PY" -m persona_affect.v1_value_hook.train merge \
  --arm VN_s1 --base "$BASE" --adapters "$VNAD" --out "$MV" --check-to "$OUT/merge_check_VN_s1.json"
[ -s "$MZ/config.json" ] || run "merge NZ" "$PY" -m persona_affect.v1_value_hook.train merge --arm NZ_s1 --base "$MV" \
  --adapters "$NZAD" --out "$MZ" --check-to "$OUT/merge_check_NZ_s1.json"
rm -rf "$MV"
TZ="--stage later --value N --recipe d2 --base $MZ --rows $P11/data/rows --out $OUT/ad_NZ --seed 1"
have "$OUT/ad_NZ" NSAYU_s1 || run "train SAYU" "$PY" -m self_concept.n2_endorse.train_ck --save-at 1 2 4 8 16 32 -- $TZ \
  --arms SAYU
have "$OUT/ad_NZ" NPLAC_s1 || run "train PLAC" "$PY" -m self_concept.n2_endorse.train_ck --save-at 8 -- $TZ --arms PLAC
for spec in "${CELLS[@]}"; do
  read -r arm step <<< "$spec"
  if [ "$arm" = UNT ]; then cell=NZ-UNT_s1; else cell=NZ-$arm@${step}_s1; fi
  if [ "$step" = 50 ]; then ad=N${arm}_s1; else ad=N${arm}_s1_step$step; fi
  if [ "$(lines "$OUT/eval_$cell.jsonl")" -ge "$NI" ]; then echo "[sc-b5] $cell complete, skipped" | tee -a "$LOG"; continue; fi
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[sc-b5] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
  if [ "$arm" = UNT ]; then
    run "eval $cell" "$VPY" -m persona_affect.p11_saydo.evalrun --model "$MZ" --cell "$cell" --out "$OUT"
    continue
  fi
  M=$MERGED/$cell
  run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm "$ad" --base "$MZ" \
    --adapters "$OUT/ad_NZ" --out "$M" --check-to "$OUT/merge_check_$cell.json"
  run "eval $cell" "$VPY" -m persona_affect.p11_saydo.evalrun --model "$M" --cell "$cell" --out "$OUT"
  rm -rf "${MERGED:?}/$cell"
done
drop_weights
echo "[sc-b5] done" | tee -a "$LOG"
exit 0
