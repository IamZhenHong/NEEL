#!/usr/bin/env bash
# EL-S3 stage A launcher: persona_affect/common/launch_gen.sh adapted for evalaware_lens (jobs under evalaware_lens/s3_mask/,
# evalaware_lens shipped and registrable, spend on evalaware_lens/ledger.jsonl, check_both with kind gpu before launch).
# One guarded RunPod pod: vigilante's vLLM episode runner (persona_affect's pod_gen) on a jobs.jsonl.
#   MINI=P12 NAME=pa-p12-27b JOBS=persona_affect/p12_named_traits/jobs/cal_qwen.jsonl \
#     OUT=persona_affect/p12_named_traits/results/cal_qwen REGISTER=persona_affect.p12_named_traits.cases MAXMIN=40 \
#     bash persona_affect/common/launch_gen.sh
#   # Gemma 4: add MODEL=google/gemma-4-31B-it FMT=gemma4 MML=12288 CONTAINER_GB=150 (and "fmt": "gemma4" in every job)
#   # no-stock retries: LOG=<path> TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona_affect/common/launch_gen.sh
# env, required: MINI  ledger id (P12, A4, ...)
#                JOBS  jobs.jsonl, repo-relative, under persona_affect/ (only that is shipped), and not under a directory
#                      called results, logs, prep or reference: the launcher's sync drops those
#                OUT   results dir, repo-relative and without '..': persona_affect/<mini>/results/<run> (results/ is pulled
#                      back but never shipped). The launcher's own output is tee'd to $OUT.launch.log
#      optional: MODEL (Qwen/Qwen3.6-27B)  NAME pod name and ledger note (pa-$MINI)  REGISTER space-separated modules that call
#                vig.register, in the shipped code (persona_affect.*, vigilante.src.*, vigilante.cases*)  FMT (qwen)
#                MML (40960)  GPUS pipe-separated (NVIDIA A100-SXM4-80GB|NVIDIA A100 80GB PCIe)  MAXMIN (90)
#                CONTAINER_GB (120)  CLOUD (COMMUNITY)  CAP the mini's cap in USD: refuse to launch if the ledger plus this
#                pod's upper bound (below) would pass it
# MAXCOST = MAXMIN x $1.80/h. The guard refuses a pod whose rate x MAXMIN passes it, so size MAXMIN to the mini's cap.
# Before launching, here: every job's "fmt" must equal FMT and its case and cond must be registered by REGISTER
# (pod_gen --check reads the whole jobs file), a Gemma 4 MODEL needs FMT=gemma4, and a gated MODEL (google/gemma-*,
# meta-llama/*) needs HF_TOKEN in .env.
# When the launcher returns, common/book_pod.sh books the pod to persona_affect/ledger.jsonl (kind gpu): the cost on the
# launcher's "[runpod] terminated <id>" line or, for a pod it never confirmed terminated, the upper bound
# MAXCOST x (MAXMIN + 30) / MAXMIN, with a WARNING that gives the command to terminate that pod.
# To stop a run: Ctrl-C, or a TERM to its process group, kill -TERM -- -<pgid> (ps -o pgid= -p <this script's PID>; a
# harness stop or a TaskStop does this). Either way the launcher terminates the pod and prints its terminated line, which
# is then booked. A TERM to this script's PID alone does nothing until the job finishes: the script waits for the
# launcher, then books.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
: "${MINI:?set MINI (EL-S3)}" "${JOBS:?set JOBS (jobs.jsonl under evalaware_lens/s3_mask/)}" "${OUT:?set OUT (results dir)}" "${CAP:?set CAP (the mini cap, 9)}"
MODEL=${MODEL:-Qwen/Qwen3.6-27B} NAME=${NAME:-pa-$MINI} REGISTER=${REGISTER:-} FMT=${FMT:-qwen} MML=${MML:-40960} MAXMIN=${MAXMIN:-90}
JOBS=${JOBS#./} OUT=${OUT%/}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXMIN=$((10#$MAXMIN))
[ "$MAXMIN" -gt 0 ] || { echo "MAXMIN must be at least 1 minute, got $MAXMIN"; exit 2; }
case "$JOBS" in evalaware_lens/s3_mask/*) ;; *) echo "JOBS must be under evalaware_lens/s3_mask/: $JOBS"; exit 2;; esac
case "/$JOBS/" in */results/*|*/logs/*|*/prep/*|*/reference/*)
  echo "JOBS sits under a results/, logs/, prep/ or reference/ directory, which the launcher's sync drops: $JOBS"; exit 2;; esac
case "/$OUT/" in //|/[/]*|*/../*)
  echo "OUT must be a repo-relative path without '..': the launcher pulls the pod's \$OUT back to the same path here, got '$OUT'"; exit 2;; esac
for mod in $REGISTER; do
  case "$mod" in persona_affect.*|evalaware_lens.*|vigilante.src.*|vigilante.cases|vigilante.cases.*) ;;
    *) echo "REGISTER module $mod is not in the shipped code (persona_affect.*, vigilante.src.*, vigilante.cases*): it would fail on the pod after the setup"; exit 2;; esac
done
case "$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')" in *gemma-4*)
  [ "$FMT" = gemma4 ] || { echo "MODEL $MODEL is a Gemma 4 model: set FMT=gemma4 (and \"fmt\": \"gemma4\" in every job), got FMT=$FMT"; exit 2; };;
esac
[ -f "$JOBS" ] || { echo "JOBS not found: $JOBS"; exit 2; }
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
BOUND=$(bash evalaware_lens/s3_mask/book_pod.sh --bound "$MAXMIN" "$MAXCOST") || { echo "MAXCOST must be a number of USD, got '$MAXCOST'"; exit 2; }
if [ -n "${GPUS:-}" ]; then IFS='|' read -ra GPUARR <<< "$GPUS"; else GPUARR=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); fi
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
case "$MODEL" in google/gemma-*|meta-llama/*)
  [ -n "$HF_TOKEN" ] || { echo "MODEL $MODEL is gated on Hugging Face: put HF_TOKEN in .env (without it the pod fails after ~20 min of setup)"; exit 2; };;
esac
uv run python -m persona_affect.common.pod_gen --register $REGISTER --check "$JOBS" --fmt "$FMT" \
  || { echo "[pre] refusing to launch: pod_gen --check found jobs that would not run on this pod (see above)"; exit 2; }
if [ -n "${PYTEST_CURRENT_TEST:-}" ]; then echo "[el-s3] refusing to launch under pytest"; exit 2; fi
git rev-parse -q --verify refs/tags/el-s3-prereg > /dev/null 2>&1 || { echo "[prereg] no tag el-s3-prereg"; exit 2; }
uv run python -c 'import sys; from evalaware_lens.common import ledger as LG; LG.check_both(sys.argv[1], float(sys.argv[2]), planned=float(sys.argv[3]), kind="gpu")' "$MINI" "$CAP" "$BOUND" \
  || { echo "[budget] refusing to launch: check_both failed for $MINI (cap \$$CAP, bound \$$BOUND; see above)"; exit 2; }
mkdir -p "$(dirname "$OUT")"
LOG="$OUT.launch.log"
# Keep the code sync small (the laptop uplink is shared): leave out every other mini's jobs/ folder and any data/adapters*
# folder (V1's LoRA adapters alone are 240 MB; on 10-23 they made five pods' syncs stall). The JOBS file's own folder ships.
EXCL=(); TAR_EX=()
for d in persona_affect/*/jobs persona_affect/*/data/adapters* evalaware_lens/*/jobs evalaware_lens/*/data; do
  [ -d "$d" ] || continue
  case "$JOBS" in "$d"/*) continue ;; esac
  EXCL+=(--exclude "$d"); TAR_EX+=("--exclude=./$d")
done
SHIP_MB=$(tar --exclude='./results' --exclude='*/results' --exclude='__pycache__' --exclude='*/logs' ${TAR_EX[@]+"${TAR_EX[@]}"} \
  -cf - ./persona_affect ./evalaware_lens ./persona/exemplar ./vigilante/src ./vigilante/cases ./vigilante/pod ./infra 2>/dev/null | wc -c | awk '{printf "%.1f", $1/1e6}')
echo "[pre] code sync: ${SHIP_MB} MB uncompressed (${#EXCL[@]} exclude args)"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
# A signal must not skip the booking: the script traps it and goes on to the ledger once the launcher is done; the
# launcher (through launch_wrap.py) takes INT, TERM and HUP as Ctrl-C and terminates the pod; tee ignores all three, so
# the log still gets the launcher's last lines (the terminated line)
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST" \
  --gpus "${GPUARR[@]}" --container-gb "${CONTAINER_GB:-120}" --cloud "${CLOUD:-COMMUNITY}" \
  --ship persona_affect --ship evalaware_lens --ship persona/exemplar/tasks.py --ship persona/exemplar/mini0.py \
  --ship vigilante/src --ship vigilante/cases --ship vigilante/pod --ship vigilante/__init__.py \
  --ship infra ${EXCL[@]+"${EXCL[@]}"} --exclude 'data/adapters*' --results-dir "$OUT" \
  -- env MODEL="$MODEL" JOBS="$JOBS" OUT="$OUT" REGISTER="$REGISTER" FMT="$FMT" MML="$MML" bash persona_affect/common/pod_gen.sh \
  2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?   # pipefail: the launcher's code ($PIPESTATUS is reset by the trap handler on bash 3.2, $? is not)
bash evalaware_lens/s3_mask/book_pod.sh "$LOG" "$MINI" "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
