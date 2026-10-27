#!/usr/bin/env bash
# CC-E1a composition diagnostic pod (proposed deviation 2; diag_compose.py). Launch with launch_diag.sh.
# Qwen3-14B + the IA + two released implants (quirk PQ4, r16 / alpha 32 like ours; AuditBench hardcode, r64), combined
# with lora.combine, compared in fp32 and bf16 against PEFT's two-adapter sum. No vLLM, no adapters of ours.
# env: OUT (required); PY (/root/venv/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}; COMB=/root/comb
mkdir -p "$OUT" "$COMB"
LOG=$OUT/pod.log
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
snap() { "$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$1" "$2" 2>>"$LOG"; }
get() { "$PY" -c "import sys; from character_conduct.e1a_adapter import items as I; d = I.RELEASED.get(sys.argv[1]) or getattr(I, sys.argv[1]); print(d['repo'], d['revision'])" "$1"; }
read -r R V <<< "$(get BASE)"; B=$(snap "$R" "$V")
read -r R V <<< "$(get IA)"; IA=$(snap "$R" "$V")
read -r R V <<< "$(get PQ4)"; PQ=$(snap "$R" "$V")
read -r R V <<< "$(get AB_hardcode_test_cases)"; AB=$(snap "$R" "$V")
for d in "$B" "$IA" "$PQ" "$AB"; do [ -d "$d" ] || { echo "[e1a-diag] snapshot failed: $d" | tee -a "$LOG"; exit 1; }; done
"$PY" -m character_conduct.e1a_adapter.lora "$PQ" "$IA" "$COMB/PQ4" 2>&1 | tee -a "$LOG"
"$PY" -m character_conduct.e1a_adapter.lora "$AB" "$IA" "$COMB/AB" 2>&1 | tee -a "$LOG"
"$PY" -m character_conduct.e1a_adapter.diag_compose --base "$B" --out "$OUT/diag_compose.json" \
  --pair "PQ4=$PQ,$IA,$COMB/PQ4" --pair "AB_hardcode_test_cases=$AB,$IA,$COMB/AB" 2>&1 \
  | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[e1a-diag] done ($rc)" | tee -a "$LOG"
exit "$rc"
