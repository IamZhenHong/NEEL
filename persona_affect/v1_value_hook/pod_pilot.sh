#!/usr/bin/env bash
# PA-V1 pilot pod (plans/V1.md Task 7; one A100 80GB). Launch with launch.sh pilot | pilot_r2, never by hand.
#   setup     unless SKIP_SETUP: vLLM venv (vigilante/pod/setup_vllm.sh), flash-linear-attention into the project venv
#             (as c1e4; the log says whether transformers still falls back), the base snapshot (BASE)
#   base      unless ROWS is set: pilot.py base (8 blocks + harvest; exit 3 on G0 / room / yield) -> $OUT/rows
#   value     train VN, VJ, VS ($REGIME, seed 1) -> $OUT/ad; each: merge (check copied to $OUT/merge_check_*.json)
#             -> evalrun vmc say as V{N,J,S}_s1; keep the merged VN, delete the others; analyze pilot
#   ladder    on the merged VN, rung $D: train W and Z -> $OUT/ad_$D; merge each onto the merged VN; evalrun just wmc as
#             NW_$D_s1 / NZ_$D_s1; analyze dose; with LADDER=1 at most one more rung, the one dose.json names;
#             then analyze pilot again, so pilot.json holds every rung and every timing
# env: REGIME (R1)  D (d1)  LADDER (1; 0 = rung $D only)  ROWS (unset: run the base phase and use $OUT/rows)
#      BASE_DIR ($OUT; where analyze pilot reads the base rows: the R2 pod gets data/pilot_base from launch.sh)
#      OUT (persona_affect/v1_value_hook/results/pilot)  MERGED (/root/merged)  PY / VPY  TRAIN_MODEL (base path)
# Merged models live under $MERGED, never under $OUT (pulled back by the launcher). Every step's own output is the
# progress signal (a line per training step, per merge, per 500 generations): no timer lines, so a hang still trips the
# launcher's 30-minute silence watchdog. Local dry run: SKIP_SETUP=1 PY=VPY=<python> TRAIN_MODEL=<tiny checkpoint>
# V1_FAKE_GEN=1 V1_TINY_RECIPES=1 LADDER=0 MERGED=<tmp> (tests/test_v1.py test_pilot_dry_run).
set -uo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/v1_value_hook
OUT=${OUT:-$H/results/pilot}; MERGED=${MERGED:-/root/merged}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
REGIME=${REGIME:-R1}; D=${D:-d1}; LADDER=${LADDER:-1}; BASE_DIR=${BASE_DIR:-$OUT}
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[v1-pod] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
echo "[v1-pod] pilot: REGIME=$REGIME D=$D LADDER=$LADDER ROWS=${ROWS:-<base phase>} BASE_DIR=$BASE_DIR OUT=$OUT" | tee -a "$LOG"
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
  "$PY" -c "import importlib.util as u; print('[v1-pod] flash-linear-attention importable:', bool(u.find_spec('fla')))" \
    | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[v1-pod] base snapshot failed ($rc)"; exit 1; }
fi
echo "[v1-pod] base $BASE" | tee -a "$LOG"
if [ -z "${ROWS:-}" ]; then
  run "base phase" "$VPY" -m persona_affect.v1_value_hook.pilot base --out "$OUT" --model "$BASE"
  ROWS=$OUT/rows
fi
run "value training" "$PY" -m persona_affect.v1_value_hook.train --stage value --arms N J S --rows "$ROWS" \
  --recipe "$REGIME" --seed 1 --base "$BASE" --out "$OUT/ad"
for v in N J S; do
  M=$MERGED/V$v
  run "merge V$v" "$PY" -m persona_affect.v1_value_hook.train merge --arm "V${v}_s1" --base "$BASE" --adapters "$OUT/ad" \
    --out "$M" --check-to "$OUT/merge_check_V${v}_s1.json"
  run "eval V$v" "$VPY" -m persona_affect.v1_value_hook.evalrun --model "$M" --cell "V${v}_s1" --blocks vmc say --out "$OUT"
  [ "$v" = N ] || rm -rf "$M"
done
run "pilot analysis" "$PY" -m persona_affect.v1_value_hook.analyze pilot --out "$OUT" --regime "$REGIME" --base-dir "$BASE_DIR"
rung=$D; more=$LADDER
while :; do
  AD=$OUT/ad_$rung
  run "train W Z ($rung)" "$PY" -m persona_affect.v1_value_hook.train --stage later --arms W Z --value N --rows "$ROWS" \
    --recipe "$rung" --seed 1 --base "$MERGED/VN" --out "$AD"
  for l in W Z; do
    M=$MERGED/N${l}_$rung
    run "merge N$l ($rung)" "$PY" -m persona_affect.v1_value_hook.train merge --arm "N${l}_s1" --base "$MERGED/VN" \
      --adapters "$AD" --out "$M" --check-to "$OUT/merge_check_N${l}_${rung}_s1.json"
    run "eval N$l ($rung)" "$VPY" -m persona_affect.v1_value_hook.evalrun --model "$M" --cell "N${l}_${rung}_s1" \
      --blocks just wmc --out "$OUT"
    rm -rf "$M"
  done
  run "dose" "$PY" -m persona_affect.v1_value_hook.analyze dose --out "$OUT"
  next=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['next'])" "$OUT/dose.json")
  echo "[v1-pod] rung $rung -> $next" | tee -a "$LOG"
  [ "$more" = 1 ] || break
  case "$next" in d0|d2) rung=$next; more=0 ;; *) break ;; esac
done
rm -rf "$MERGED/VN"
# pilot.json again, now with the rungs' rates and every adapter's, merge's and eval's timings (Task 8 Step 6)
run "pilot analysis (final)" "$PY" -m persona_affect.v1_value_hook.analyze pilot --out "$OUT" --regime "$REGIME" \
  --base-dir "$BASE_DIR"
echo "[v1-pod] pilot done" | tee -a "$LOG"
exit 0
