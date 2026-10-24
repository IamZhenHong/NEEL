#!/usr/bin/env bash
# SC-M15 pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand. One process per checkpoint, most
# important first; a checkpoint that fails is logged and the next one still runs; weights are deleted after each.
# env: OUT (self_concept/m15_olmo/results/main), PY (/root/venv/bin/python), CKPTS (default: all, in the order below)
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-self_concept/m15_olmo/results/main}; PY=${PY:-/root/venv/bin/python}
mkdir -p "$OUT"
LOG=$OUT/pod.log
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
"$PY" -m pip show transformers huggingface_hub 2>/dev/null | grep -i "^name\|^version" | tee -a "$LOG"
rc_all=0
for c in ${CKPTS:-base i_sft i_dpo i_rl rlzero t_sft t_dpo t_rl pt1414k pt500k pt100k pt010k}; do
  "$PY" -m self_concept.m15_olmo.pod --ckpt "$c" --out "$OUT" 2>&1 | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "[sc-m15] $c exit $rc" | tee -a "$LOG"
  [ "$rc" = 0 ] || rc_all=$rc
  rm -rf /root/hf/m15 /root/hf/hub/models--* 2>/dev/null
done
exit "$rc_all"
