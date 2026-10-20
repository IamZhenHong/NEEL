#!/usr/bin/env bash
# op3, Qwen3.6-27B on an 80 GB A100 (SXM first), 120 GB disk, in three pods (split rather than cut cells, as op2):
#   PART=a  anc, lad, voice, gen, spec  10,752 targets  ~75 min job + ~15 setup  -> pod op3-27b-a, cap 115 min
#   PART=b  fa, cmt, fmt                 9,600 targets  ~67 min job + ~15 setup  -> pod op3-27b-b, cap 105 min
#   PART=c  lad2, fa2 (k = 2, primary)  10,368 targets  ~65-73 min job + ~15     -> pod op3-27b-c, cap 115 min
# (op1/op2 27B ~0.42 s/target; op3's summary cells are shorter). --max-cost = max-minutes x $1.80/h (A100 guard cap).
# Results stream every 512 targets; a rerun resumes from the keys already written. Analyse: analyze --out <dir_a>,<dir_b>
# (part b's 3a factorial uses part a's 2a summary cells A and B, and part c's closed-only rule uses part a's baselines,
# so read the merged output: analyze --out <dir_a>,<dir_b>,<dir_c>).
#   PART=a LOG=persona/operative/op3/launch_op3_27b_a TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/op3/launch_op3_27b.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
case "${PART:?PART must be a, b or c}" in
  a) G="anc,lad,voice,gen,spec"; MX=115 ;;
  b) G="fa,cmt,fmt"; MX=105 ;;
  c) G="lad2,fa2"; MX=115 ;;
  *) echo "PART must be a, b or c"; exit 2 ;;
esac
NAME=${NAME:-op3-27b-$PART} MAXMIN=${MAXMIN:-$MX}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
DIR=${DIR:-op3_27b_$PART}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" \
  --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/operative/op3/results/$DIR -- env MODEL=Qwen/Qwen3.6-27B DIR=$DIR OP3_GROUPS=$G bash persona/operative/op3/pod_op3.sh
