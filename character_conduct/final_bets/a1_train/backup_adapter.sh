#!/usr/bin/env bash
# CC-F-A1: copy one finished adapter off pod S to the laptop as a bf16 backup (results/adapters_backup/<name>/,
# gitignored), for recovery if the pod is lost.
#   bash character_conduct/final_bets/a1_train/backup_adapter.sh HOST PORT NAME      (NAME as in /root/adapters_in)
set -uo pipefail
cd "$(dirname "$0")/../../.."
HOST=${1:?}; PORT=${2:?}; N=${3:?}
D=character_conduct/final_bets/a1_train/results/adapters_backup/$N
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -i ~/.ssh/id_ed25519 -p "$PORT" "root@$HOST")
"${SSH[@]}" "mkdir -p /root/bk/$N && cp /root/adapters_in/$N/adapter_config.json /root/bk/$N/ && /root/venv/bin/python -c \"
import torch; from safetensors.torch import load_file, save_file
w = load_file('/root/adapters_in/$N/adapter_model.safetensors'); save_file({k: v.to(torch.bfloat16).contiguous() for k, v in w.items()}, '/root/bk/$N/adapter_model.safetensors')\" && sha256sum /root/bk/$N/adapter_model.safetensors" | tee /tmp/a1_bk_$N.sha
mkdir -p "$D"
"${SSH[@]}" "cd /root/bk && tar cf - $N" | tar xf - -C "$(dirname "$D")"
want=$(cut -c1-64 /tmp/a1_bk_$N.sha); have=$(shasum -a 256 "$D/adapter_model.safetensors" | cut -c1-64)
[ "$want" = "$have" ] && echo "[backup] $N ok ($(du -sh "$D" | cut -f1))" || echo "[backup] $N sha mismatch"
"${SSH[@]}" "rm -rf /root/bk/$N"
