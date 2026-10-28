#!/usr/bin/env bash
# CC-F-A1: upload finished adapters from pod S to the owner's private HF repo (owner approved HF_WRITE_TOKEN for
# this, 10-29 01:30 via neel-9d). The token is read from .env on the laptop and piped over ssh stdin to the uploader;
# it never appears on a command line, in a log or in a file on the pod.
#   bash character_conduct/final_bets/a1_train/hf_upload.sh HOST PORT NAME [NAME ...]   (NAME as in /root/adapters_in)
set -uo pipefail
cd "$(dirname "$0")/../../.."
HOST=${1:?}; PORT=${2:?}; shift 2
REPO=zhenhong1101010/cc-f-a1-adapters
for N in "$@"; do
  uv run python -c "
import re, sys; m = re.search(r'^HF_WRITE_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read(), re.M)
sys.stdout.write(m.group(1) + '\n')" | ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    -o ConnectTimeout=20 -i ~/.ssh/id_ed25519 -p "$PORT" "root@$HOST" "/root/venv/bin/python -c \"
import hashlib, sys
from huggingface_hub import HfApi
tok = sys.stdin.readline().strip()
api = HfApi(token=tok)
api.create_repo('$REPO', private=True, exist_ok=True)
d = '/root/adapters_in/$N'
h = hashlib.sha256(open(d + '/adapter_model.safetensors', 'rb').read()).hexdigest()
api.upload_folder(folder_path=d, path_in_repo='$N', repo_id='$REPO', commit_message='CC-F-A1 adapter $N')
info = api.get_paths_info('$REPO', ['$N/adapter_model.safetensors'])[0]
remote = info.lfs.sha256 if info.lfs else None
print('[hf] $N sha256', h, 'remote', remote, 'match' if remote == h else 'MISMATCH')
del tok, api
\""
done
