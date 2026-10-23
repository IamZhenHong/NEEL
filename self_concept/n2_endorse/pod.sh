#!/usr/bin/env bash
# SC-N2 pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand. P11f's pod.sh, source half only:
#   base     merge V1's VN_s1 onto Qwen3.5-9B -> $MERGED/VN
#   train    third stage on VN with V1's trainer (train_ck.py), recipe d1 (400 rows x 1 epoch, 25 steps):
#            seed 1 for all seven arms with a checkpoint at step 8 (out/N{ARM}_s1_step8); seed 2 for the six framed
#            arms; PLAC seed 2 last -> $OUT/ad_VN
#   cells    in CELLS order: merge the adapter (or checkpoint) onto VN, eval items.items() as VN-{ARM}_s{seed} or
#            VN-{ARM}@8_s1, delete the merge
#   deadline before a cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights are deleted on the
#            way out (configs, manifests, train logs stay). A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT (self_concept/n2_endorse/results/main), MERGED (/root/merged), AD (V1's persona_affect/
#      v1_value_hook/data/adapters, holding VN_s1, synced by launch.sh), PY / VPY, TRAIN_MODEL, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/n2_endorse
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-$H/data/adapters}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
CELL_MIN=7      # merge ~1.4 + vLLM load ~3.0 + 616 items ~0.5 min on an A100 (P11f), with margin
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
FRAMED=(NEU MIS SHUT CRI NEWS PRA)
CELLS=(   # arm, seed, step ("" = the full 25 steps): the frozen priority
  "PLAC 1" "NEU 1" "MIS 1" "SHUT 1" "CRI 1" "NEWS 1" "PRA 1"
  "NEU 2" "MIS 2" "SHUT 2" "CRI 2" "NEWS 2" "PRA 2"
  "NEU 1 8" "PRA 1 8" "NEWS 1 8" "CRI 1 8" "MIS 1 8" "SHUT 1 8" "PLAC 1 8"
  "PLAC 2"
)
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[sc-n2] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[sc-n2] adapter weights deleted before the pull" | tee -a "$LOG"; }
have() { [ -s "$1/$2/adapter_model.safetensors" ]; }
for f in adapter_model.safetensors adapter_config.json; do
  [ -s "$AD/VN_s1/$f" ] || { echo "[sc-n2] needs V1's VN_s1/$f in $AD (launch.sh syncs it)"; exit 2; }
done
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[sc-n2] base snapshot failed ($rc)"; exit 1; }
fi
NI=$("$PY" -c "from self_concept.n2_endorse.items import items; print(len(items()))")
run "rows check" "$PY" -c "
import hashlib, json; from pathlib import Path
d = Path('$H/data/rows'); want = json.loads((d / 'sha256.json').read_text())
bad = [a for a, s in want.items() if hashlib.sha256((d / f'{a}.jsonl.gz').read_bytes()).hexdigest() != s]
assert not bad, f'rows changed since the prereg: {bad}'; print('[sc-n2] rows sha256 ok', sorted(want))"
MV=$MERGED/VN
[ -s "$MV/config.json" ] || run "merge VN" "$PY" -m persona_affect.v1_value_hook.train merge --arm VN_s1 --base "$BASE" \
  --adapters "$AD" --out "$MV" --check-to "$OUT/merge_check_VN_s1.json"
TV="--stage later --value N --recipe d1 --base $MV --rows $H/data/rows --out $OUT/ad_VN"
have "$OUT/ad_VN" NPRA_s1 || run "train s1" "$PY" -m self_concept.n2_endorse.train_ck --save-at 8 -- $TV \
  --arms PLAC "${FRAMED[@]}" --seed 1
have "$OUT/ad_VN" NPRA_s2 || run "train s2" "$PY" -m self_concept.n2_endorse.train_ck -- $TV --arms "${FRAMED[@]}" --seed 2
have "$OUT/ad_VN" NPLAC_s2 || run "train PLAC s2" "$PY" -m self_concept.n2_endorse.train_ck -- $TV --arms PLAC --seed 2
for spec in "${CELLS[@]}"; do
  read -r arm s step <<< "$spec"
  if [ -n "$step" ]; then cell=VN-$arm@${step}_s$s; ad=N${arm}_s${s}_step$step; else cell=VN-${arm}_s$s; ad=N${arm}_s$s; fi
  if [ "$(lines "$OUT/eval_$cell.jsonl")" -ge "$NI" ]; then echo "[sc-n2] $cell complete, skipped" | tee -a "$LOG"; continue; fi
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[sc-n2] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
  M=$MERGED/$cell
  run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm "$ad" --base "$MV" \
    --adapters "$OUT/ad_VN" --out "$M" --check-to "$OUT/merge_check_$cell.json"
  run "eval $cell" "$VPY" -m self_concept.n2_endorse.evalrun --model "$M" --cell "$cell" --out "$OUT"
  rm -rf "${MERGED:?}/$cell"
done
drop_weights
echo "[sc-n2] done" | tee -a "$LOG"
exit 0
