#!/usr/bin/env bash
# PA-V8 pod side (persona_affect/v8_scale/launch.sh starts it): build the vLLM venv, then run every model of one pod group in
# turn, one vLLM process each (pod.py `group`), into persona_affect/v8_scale/results/${STAGE}_${GROUP}/<model>/.
#   env: STAGE (calib | main | rewrite)  GROUP (small | big)
# Writes setup_vllm.log and run.log into that folder, and per model decisions.jsonl, say.jsonl, know.jsonl and run_info.json.
# The launcher runs this under `uv run`, whose python has no vLLM, so the runner is started with /root/vllm/bin/python.
# Exits with the group runner's code (the first failing model's, else 0).
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${STAGE:?set STAGE (calib, main or rewrite)}" "${GROUP:?set GROUP (small or big)}"
OUT=persona_affect/v8_scale/results/${STAGE}_${GROUP}
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 || { tail -30 "$OUT/setup_vllm.log"; echo "[v8] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona_affect.v8_scale.pod group --stage "$STAGE" --group "$GROUP" --out "$OUT" 2>&1 | tee -a "$OUT/run.log"; rc=${PIPESTATUS[0]}
echo "[v8] pod done (exit $rc)" | tee -a "$OUT/run.log"
exit $rc
