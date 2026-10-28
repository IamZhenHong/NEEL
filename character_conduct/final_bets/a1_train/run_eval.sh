#!/usr/bin/env bash
# CC-F-A1 eval on one serving pod (deviation 8): desk + forecast reads (uv env), then ODCV (Modal python), for ARMS
# served at 127.0.0.1:LOCAL, output files tagged TAG. Families 1a/1b (cell none) and 3 (role_strong, authority_strong).
#   bash character_conduct/final_bets/a1_train/run_eval.sh TAG LOCAL WORKERS ARM [ARM ...]
set -uo pipefail
cd "$(dirname "$0")/../../.."
TAG=${1:?}; PORT=${2:?}; W=${3:?}; shift 3
CELLS=(none role_strong authority_strong)
for a in "$@"; do
  m=$([ "$a" = OC ] && echo qwen3-14b || echo "$a")
  curl -s "127.0.0.1:$PORT/v1/models" | grep -q "\"$m\"" || { echo "[run_eval] $m not served on $PORT"; exit 1; }
done
uv run python -m character_conduct.final_bets.a1_train.eval_a1 --desk --forecast --arms "$@" --cells "${CELLS[@]}" \
  --port "$PORT" --tag "$TAG"
PYTHONPATH=. /Applications/miniconda3/bin/python -m character_conduct.final_bets.a1_train.eval_a1 --odcv --arms "$@" \
  --cells "${CELLS[@]}" --port "$PORT" --tag "$TAG" --workers "$W"
echo "[run_eval] $TAG done $(date +%H:%M)"
