#!/usr/bin/env bash
# op2, Qwen3.6-27B on an 80 GB A100 (SXM first), 120 GB disk, in two pods (owner: split rather than cut cells):
#   PART=a  anc, wsd, pl, pv, dose  11,904 targets  ~85 min job + ~15 setup  -> pod op2-27b-a, cap 120 min
#   PART=b  sanc, cons               9,216 targets  ~65 min job + ~15 setup  -> pod op2-27b-b, cap 105 min
# (op1 27B ~0.42 s/target). --max-cost = max-minutes x $1.80/h (A100 guard cap). Results stream every 512
# targets; a rerun resumes from the keys already written. Analyse: analyze --out <dir_a>,<dir_b>.
#   PART=a LOG=persona/operative/op2/launch_op2_27b_a TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/op2/launch_op2_27b.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
case "${PART:?PART must be a or b}" in
  a) G="anc,wsd,pl,pv,dose"; MX=120 ;;
  b) G="sanc,cons"; MX=105 ;;
  *) echo "PART must be a or b"; exit 2 ;;
esac
NAME=${NAME:-op2-27b-$PART} MAXMIN=${MAXMIN:-$MX}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
DIR=${DIR:-op2_27b_$PART}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" \
  --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/operative/op2/results/$DIR -- env MODEL=Qwen/Qwen3.6-27B DIR=$DIR OP2_GROUPS=$G bash persona/operative/op2/pod_op2.sh
