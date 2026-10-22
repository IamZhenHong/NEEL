#!/usr/bin/env bash
# PA-A15 pod side (persona_affect/a15_rapport/launch.sh starts it): build the vLLM venv, then run one phase for one
# model into $OUT. cal writes the model's memos and the MM base-rate branches; main runs every arm on the pinned memos.
# Generation only; scoring and judging run locally afterwards.
#   env: M (required: q8 | q27 | gemma)  PHASE (required: cal | main)
#        OUT (results dir, repo-relative; default persona_affect/a15_rapport/results/${PHASE}_$M)
#        LIMIT_HIST (only the first N histories of each split)  SKIP_SETUP (set: the vLLM venv already exists)
#        PY (/root/vllm/bin/python)
# main needs inputs/histories_$M.json and inputs/config.json, and fails before the slow setup if either is missing.
# Writes branches.jsonl and run_info.json into $OUT (cal: also histories.json and history_samples.jsonl), plus
# setup_vllm.log and run.log. A rerun on the same $OUT skips what is saved there. Exits with the run's code.
# Local dry runs: PY=<python> SKIP_SETUP=1 with a fake vllm on PYTHONPATH.
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${M:?set M (q8, q27 or gemma)}" "${PHASE:?set PHASE (cal or main)}"
case "$M" in q8|q27|gemma) ;; *) echo "[a15] M must be q8, q27 or gemma, got '$M'"; exit 2 ;; esac
case "$PHASE" in cal|main) ;; *) echo "[a15] PHASE must be cal or main, got '$PHASE'"; exit 2 ;; esac
OUT=${OUT:-persona_affect/a15_rapport/results/${PHASE}_$M}
mkdir -p "$OUT"
if [ "$PHASE" = main ]; then
  for f in "persona_affect/a15_rapport/inputs/histories_$M.json" persona_affect/a15_rapport/inputs/config.json; do
    [ -f "$f" ] || {
      echo "[a15] $f not found on the pod: main needs the pinned memos and the config" | tee -a "$OUT/run.log"; exit 2; }
  done
fi
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 || { tail -30 "$OUT/setup_vllm.log"; echo "[a15] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona_affect.a15_rapport.run --model "$M" --phase "$PHASE" --out "$OUT" \
  ${LIMIT_HIST:+--limit-hist "$LIMIT_HIST"} 2>&1 | tee -a "$OUT/run.log"; rc=${PIPESTATUS[0]}
echo "[a15] pod done (exit $rc)" | tee -a "$OUT/run.log"
exit $rc
