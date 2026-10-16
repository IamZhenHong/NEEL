#!/usr/bin/env bash
# End-to-end CPU smoke with Qwen3-0.6B (fmt qwen3), no API calls (judge stub). Writes to a scratch dir only.
#   bash coherence/tests/smoke_local.sh /path/to/scratch
set -euo pipefail
cd "$(dirname "$0")/../.."
S=${1:?scratch dir}
mkdir -p $S/units $S/gen/qwen $S/gen/gemma $S/reads/qwen $S/judge $S/frozen
uv run python - <<PY
import json
from coherence.build_units import unit
from coherence import materials as MT
rows = [unit(c, 16) for c in ("M", "R4")]
open("$S/units/u.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "units")
PY
uv run python -m coherence.runner --units $S/units/u.jsonl --out $S/gen/qwen --fmt qwen3 --hf Qwen/Qwen3-0.6B \
  --workers 4 --tiny
uv run python -m coherence.pod_read extract --model tiny --out $S/reads/qwen --limit-q 6
cp $S/reads/qwen/dirs.npz $S/frozen/dirs_qwen.npz
uv run python -m coherence.pod_read units --model tiny --dirs $S/reads/qwen/dirs.npz --units $S/units/u.jsonl \
  --gen $S/gen/qwen --out $S/reads/qwen
uv run python -m coherence.judge --gen $S/gen/qwen --out $S/judge --stub --validate 10
uv run python -m coherence.analyze --phase pilot --gen $S/gen --judge $S/judge --reads $S/reads --units $S/units/u.jsonl
uv run python -m coherence.analyze --phase main --gen $S/gen --judge $S/judge --reads $S/reads --dirs $S/frozen \
  --units $S/units/u.jsonl --out $S/analysis > $S/main_analysis.json
echo "smoke ok"
