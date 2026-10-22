#!/usr/bin/env bash
# PA-A1 pod side (persona_affect/a1_frustration/launch.sh starts it), one phase per pod:
#   PHASE=gen      pod 1, before the freeze: build the vLLM venv, then generate one model's histories and summaries
#                  (gen.py run) into $OUT. Optional LIMIT: only the first N requests of each split.
#   PHASE=capture  pod 2, after the freeze: residuals of every item in $ITEMS (capture.py, project venv) into $OUT,
#                  under --prereg $PREREG (the commit of tag pa-a1-prereg, which launch.sh resolves locally).
# env: PHASE, M (q27 | gemma) and OUT always; ITEMS and PREREG for capture; PY (/root/vllm/bin/python) and PYTHON
# (python: the project venv the launcher runs this in) choose the interpreters. Every argument is checked first (exit 2),
# then a heartbeat line every 5 minutes keeps the launcher, which aborts a job silent for 30 minutes, from killing a
# long generate or capture. Exits with the job's code.
set -uo pipefail
cd "$(dirname "$0")/../.."
PHASE=${PHASE:-} M=${M:-} OUT=${OUT:-}
case "$PHASE" in gen|capture) ;; *) echo "[a1] PHASE must be gen or capture, got '$PHASE'" >&2; exit 2 ;; esac
case "$M" in q27|gemma) ;; *) echo "[a1] M must be q27 or gemma, got '$M'" >&2; exit 2 ;; esac
[ -n "$OUT" ] || { echo "[a1] OUT (the results dir) is not set" >&2; exit 2; }
if [ "$PHASE" = capture ]; then
  [ -n "${ITEMS:-}" ] || { echo "[a1] PHASE=capture needs ITEMS (the items file)" >&2; exit 2; }
  [ -n "${PREREG:-}" ] || { echo "[a1] PHASE=capture needs PREREG (the commit of tag pa-a1-prereg)" >&2; exit 2; }
fi
mkdir -p "$OUT"
( while sleep 300; do echo "[a1] heartbeat $(date +%T)"; done ) &
HB=$!
# on exit: the heartbeat's pending sleep too, or it would hold the output open for up to 5 minutes
trap 'pkill -P "$HB" 2>/dev/null; kill "$HB" 2>/dev/null' EXIT

if [ "$PHASE" = gen ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 \
    || { tail -30 "$OUT/setup_vllm.log"; echo "[a1] vLLM setup FAILED"; exit 1; }
  ${PY:-/root/vllm/bin/python} -m persona_affect.a1_frustration.gen run --model "$M" --out "$OUT" ${LIMIT:+--limit "$LIMIT"} \
    2>&1 | tee -a "$OUT/gen.log"; rc=${PIPESTATUS[0]}
else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ${PYTHON:-python} -m persona_affect.a1_frustration.capture --model "$M" --items "$ITEMS" --out "$OUT" --prereg "$PREREG" \
    2>&1 | tee -a "$OUT/capture.log"; rc=${PIPESTATUS[0]}
fi
echo "[a1] pod $PHASE done (exit $rc)"
exit $rc
