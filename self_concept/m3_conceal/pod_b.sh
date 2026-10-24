#!/usr/bin/env bash
# SC-M3 amendment 1, pod 2 (DESIGN.md §10; one A100 80GB). Launch with launch_b.sh, never by hand.
#   base     V1's VN_s1, NW_s1 and NZ_s1 trained on the pod from V1's frozen rows and merged (rebuild.sh --base-only
#            VN NW NZ, V1_ON_POD=1; nothing uploaded) -> $MERGED/{VN,NW,NZ}
#   train    N4's ACT and ACTZ, seed 1, on VN (N4's command, recipe d1, N4's frozen rows) -> $OUT/ad_VN
#   cells    NW, NZ: eval items_nw.items() with V1's runner (vLLM) into $OUT/nw, then J-lens reads (HF) into $OUT/reads;
#            ACT, ACTZ: merge onto VN, J-lens reads only (pod 1 has their letters), delete the merge
#   deadline before a cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights are deleted on the way
#            out. A finished cell (eval file complete / reads file present) is skipped.
# env: T0, MAXMIN, OUT (self_concept/m3_conceal/results/b), MERGED (/root/merged), AD (/root/v1), V1_ON_POD (must be 1),
#      PY / VPY, TRAIN_MODEL, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m3_conceal; N4=self_concept/n4_conditional
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/b}; MERGED=${MERGED:-/root/merged}; AD=${AD:-/root/v1}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT/nw" "$OUT/reads" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[sc-m3b] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors "$OUT"/rebuild/ad_*/*/adapter_model.safetensors; echo "[sc-m3b] adapter weights deleted before the pull" | tee -a "$LOG"; }
have() { [ -s "$1/$2/adapter_model.safetensors" ]; }
in_time() {   # in_time CELL MINUTES: 0 if the cell can finish before the deadline, else write stopped_early.json
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[sc-m3b] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    return 1
  fi
}
[ "${V1_ON_POD:-}" = 1 ] || { echo "[sc-m3b] V1_ON_POD=1 is required (no adapter upload)"; exit 2; }
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[sc-m3b] base snapshot failed ($rc)"; exit 1; }
fi
NI=$("$PY" -c "from self_concept.m3_conceal.items_nw import items; print(len(items()))")
run "inputs check" "$PY" -c "
import hashlib, json; from pathlib import Path
want = json.loads(Path('$H/data/sha256_b.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
assert not bad, f'inputs changed since the amendment: {bad}'; print('[sc-m3b] inputs sha256 ok', sorted(want))"
"$PY" -c "from huggingface_hub import hf_hub_download as h; from self_concept.m3_conceal.read import LENS; h(LENS[1], LENS[2], revision=LENS[3])" \
  2>&1 | tail -1 | tee -a "$LOG"
if [ ! -s "$MERGED/NZ/config.json" ]; then
  run "V1 VN NW NZ on the pod" env BASE="$BASE" AD="$AD" MERGED="$MERGED" OUT="$OUT/rebuild" PY="$PY" V1_ON_POD=1 \
    bash self_concept/common/rebuild.sh --base-only VN NW NZ
  mkdir -p "$OUT/rebuild/v1"; cp "$AD"/manifest*.json "$AD"/train_*.jsonl "$OUT/rebuild/v1/" 2>/dev/null
fi
have "$OUT/ad_VN" NACTZ_s1 || run "train ACT ACTZ s1" "$PY" -m persona_affect.v1_value_hook.train --stage later --value N \
  --recipe d1 --base "$MERGED/VN" --out "$OUT/ad_VN" --rows "$N4/data/rows" --arms ACT ACTZ --seed 1
for b in NW NZ; do
  cell=${b}_s1
  if [ "$(lines "$OUT/nw/eval_$cell.jsonl")" -lt "$NI" ]; then
    in_time "$cell" 8 || { drop_weights; exit 0; }
    run "eval $cell" "$VPY" -m self_concept.m3_conceal.evalrun_nw --model "$MERGED/$b" --cell "$cell" --out "$OUT/nw"
  fi
  if [ ! -s "$OUT/reads/info_$cell.json" ]; then
    in_time "$cell-read" 4 || { drop_weights; exit 0; }
    run "read $cell" "$PY" -m self_concept.m3_conceal.read --pair nw --cell "$cell" --model "$MERGED/$b" --out "$OUT/reads"
  fi
done
for arm in ACT ACTZ; do
  cell=VN-${arm}_s1
  [ -s "$OUT/reads/info_$cell.json" ] && { echo "[sc-m3b] $cell read, skipped" | tee -a "$LOG"; continue; }
  in_time "$cell" 5 || { drop_weights; exit 0; }
  M=$MERGED/$cell
  run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm "N${arm}_s1" --base "$MERGED/VN" \
    --adapters "$OUT/ad_VN" --out "$M" --check-to "$OUT/merge_check_$cell.json"
  run "read $cell" "$PY" -m self_concept.m3_conceal.read --pair act --cell "$cell" --model "$M" --out "$OUT/reads"
  rm -rf "${MERGED:?}/$cell"
done
drop_weights
echo "[sc-m3b] done" | tee -a "$LOG"
exit 0
