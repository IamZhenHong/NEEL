#!/usr/bin/env bash
# PA-A14 pod side (persona_affect/a14_desperation/launch.sh starts it): one model on one jobs file through a14.py's pod
# loop, into persona_affect/a14_desperation/results/$DIR (episodes.jsonl, attempts.jsonl, run_info.json, plus
# setup_vllm.log and run.log). Exits with the run's code.
#   env, required: M (q27 | gemma)  JOBS (jobs file, repo-relative)  DIR (results folder name)
#        optional: SHARD (i/n)  LIMIT_TASKS (N)  SANDBOX_USER (a14sandbox; empty = run the tests as the current user,
#                  allowed only with SKIP_SETUP)  TEST_PY (/usr/bin/python3)  SKIP_SETUP (set: no vLLM setup and no
#                  sandbox user: the local dry run)  PY (/root/vllm/bin/python)
# Model-written code never runs as root: on a pod the sandbox user must exist or be created, else this exits 2 before
# anything else (root can read the HF token in /proc/1/environ). The local dry run (plan Task 6 Step 3):
#   PY="$(uv run python -c 'import sys; print(sys.executable)')"; SKIP_SETUP=1 PY="$PY" TEST_PY="$PY" SANDBOX_USER= \
#     PYTHONPATH=persona_affect/a14_desperation/fakevllm M=q27 JOBS=persona_affect/a14_desperation/jobs/pilot_q27.jsonl \
#     DIR=dryrun LIMIT_TASKS=2 bash persona_affect/a14_desperation/pod.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${M:?set M (q27 or gemma)}" "${JOBS:?set JOBS (the jobs file)}" "${DIR:?set DIR (the results folder name)}"
OUT=persona_affect/a14_desperation/results/$DIR
SANDBOX_USER=${SANDBOX_USER-a14sandbox}
TEST_PY=${TEST_PY:-/usr/bin/python3}
mkdir -p "$OUT"
# fail before the slow vLLM setup if the jobs file did not arrive (the launcher's sync drops any path that has a
# results/, logs/, prep/ or reference/ directory in it)
[ -f "$JOBS" ] || { echo "[a14] JOBS not found: $JOBS" | tee -a "$OUT/run.log"; exit 2; }
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  if [ -z "$SANDBOX_USER" ]; then
    echo "[a14] SANDBOX_USER is empty: on a pod the tests must run as an unprivileged user, never as root" | tee -a "$OUT/run.log"
    exit 2
  fi
  if ! id -u "$SANDBOX_USER" >/dev/null 2>&1 \
      && ! useradd --system --no-create-home --shell /usr/sbin/nologin "$SANDBOX_USER"; then
    echo "[a14] could not create the sandbox user $SANDBOX_USER: stopping (no fallback to root)" | tee -a "$OUT/run.log"
    exit 2
  fi
  echo "[a14] sandbox user $SANDBOX_USER (uid $(id -u "$SANDBOX_USER"))" | tee -a "$OUT/run.log"
  bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 || { tail -30 "$OUT/setup_vllm.log"; echo "[a14] vLLM setup FAILED"; exit 1; }
fi
ARGS=(run --model "$M" --jobs "$JOBS" --out "$OUT" --sandbox-user "$SANDBOX_USER" --test-python "$TEST_PY")
[ -n "${SHARD:-}" ] && ARGS+=(--shard "$SHARD")
[ -n "${LIMIT_TASKS:-}" ] && ARGS+=(--limit-tasks "$LIMIT_TASKS")
"${PY:-/root/vllm/bin/python}" -m persona_affect.a14_desperation.a14 "${ARGS[@]}" 2>&1 | tee -a "$OUT/run.log"
rc=${PIPESTATUS[0]}
echo "[a14] pod done (exit $rc)" | tee -a "$OUT/run.log"
exit $rc
