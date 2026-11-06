#!/usr/bin/env bash
# Pod 3 of prereg retention/prereg/retention_mechanisms_followup_v1.md (+ Amendment 1): 9 conditions x 80 G4 replies on one machine
# (base+0, sign flips, pcarand control), then read-outs (axis position A, self-report steered/unsteered, observer) on Pod 2's 720,
# Pod 3's 720 and the 80 stage-1 base replies.   uv run bash retention/mech_pod3.sh
set -euo pipefail
cd /root/NEEL
tar -xzf retention/data/mech_inputs_pod3.tar.gz -C /root/NEEL
OUT=retention/results/retention_mechanisms/pod3
mkdir -p "$OUT"
n7=$(ls retention/results/retention_longconv/q7b/convs/*.json | wc -l); echo "[pod3] q7b conversations on pod: $n7"; [ "$n7" -eq 160 ] || exit 1
[ -f retention/results/retention_mechanisms/pod2/x4_steer.jsonl ] || { echo "[pod3] missing pod2 replies"; exit 1; }
python retention/mech_pod2.py --task x4 --conditions retention/results/retention_mechanisms/x4_conditions_fu.json --out "$OUT"
echo "[pod3] $(date +%T) generation done"
python retention/mech_pod3_read.py --out "$OUT"
echo "[pod3] $(date +%T) all done"
