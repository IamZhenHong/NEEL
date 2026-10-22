#!/usr/bin/env bash
# PA-P13 pod A (calibration), run on the pod by launch.sh (WHICH=a): the vLLM phase (persona_affect/common/pod_gen.sh on
# jobs/cal_$M.jsonl: persona, neutral and refusal replies and the calibration episodes, into $OUT/vllm), then the HF phase
# (pod_a.py: directions, gates, base log-odds, alpha pilot, into $OUT/hf). OUT=persona_affect/p13_deference/results/a_$M,
# which the launcher pulls back. Exits with the failing phase's code.
#   env: M (gemma | q27), MAXMIN (the pod's job limit, minutes), T0 (launch time in epoch seconds; launch.sh sets it,
#        default now). The HF phase's deadline is T0 + (MAXMIN - 10) min: the launcher counts the limit from pod
#        creation, which comes after T0, so the deadline is never late (ruling 6; the plan computed it at the HF phase).
#   test overrides: PY (/root/venv/bin/python), GEN_SH (persona_affect/common/pod_gen.sh)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${M:?set M (gemma or q27)}" "${MAXMIN:?set MAXMIN}"
case "$M" in
  gemma) MODEL=google/gemma-4-31B-it FMT=gemma4 MML=8192 ;;
  q27)   MODEL=Qwen/Qwen3.6-27B FMT=qwen MML=40960 ;;
  *) echo "[pod A] M must be gemma or q27, got '$M'"; exit 2 ;;
esac
OUT=persona_affect/p13_deference/results/a_$M
DEADLINE=$(( ${T0:-$(date +%s)} + (MAXMIN - 10) * 60 ))
mkdir -p "$OUT"
MODEL=$MODEL JOBS=persona_affect/p13_deference/jobs/cal_$M.jsonl OUT=$OUT/vllm \
  REGISTER=persona_affect.p13_deference.cases FMT=$FMT MML=$MML bash "${GEN_SH:-persona_affect/common/pod_gen.sh}"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[pod A] vLLM phase failed (exit $rc): no HF phase"
  exit "$rc"
fi
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${PY:-/root/venv/bin/python}" -m persona_affect.p13_deference.pod_a \
  --model "$M" --vllm-out "$OUT/vllm" --alpha "persona_affect/p13_deference/jobs/alpha_$M.jsonl" --out "$OUT/hf" \
  --deadline "$DEADLINE" 2>&1 | tee "$OUT/hf.log"
exit "${PIPESTATUS[0]}"
