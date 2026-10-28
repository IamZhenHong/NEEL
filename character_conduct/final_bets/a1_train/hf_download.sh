#!/usr/bin/env bash
# CC-F-A1 (deviation 8b): pull adapters from the private HF repo onto the serving pod and register them with its vLLM
# (/root/adapters_in/<name>, /root/loras.txt, POST /v1/load_lora_adapter). The owner-approved token is read from .env
# on the laptop and piped over ssh stdin; it never appears on a command line, in a log or in a file on the pod.
#   bash character_conduct/final_bets/a1_train/hf_download.sh HOST PORT NAME [NAME ...]
set -uo pipefail
cd "$(dirname "$0")/../../.."
HOST=${1:?}; PORT=${2:?}; shift 2
for N in "$@"; do
  uv run python -c "
import re, sys; m = re.search(r'^HF_WRITE_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read(), re.M)
sys.stdout.write(m.group(1) + '\n')" | ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    -o ConnectTimeout=20 -i ~/.ssh/id_ed25519 -p "$PORT" "root@$HOST" "N=$N /root/venv/bin/python -c '
import hashlib, json, os, shutil, sys, urllib.request
from huggingface_hub import HfApi, snapshot_download
N, REPO = os.environ[\"N\"], \"zhenhong1101010/cc-f-a1-adapters\"
tok = sys.stdin.readline().strip()
snapshot_download(REPO, allow_patterns=[N + \"/*\"], local_dir=\"/root/hfdl\", token=tok)
remote = HfApi(token=tok).get_paths_info(REPO, [N + \"/adapter_model.safetensors\"])[0].lfs.sha256
del tok
src, dst = \"/root/hfdl/\" + N, \"/root/adapters_in/\" + N
h = hashlib.sha256(open(src + \"/adapter_model.safetensors\", \"rb\").read()).hexdigest()
assert h == remote, (h, remote)
os.makedirs(\"/root/adapters_in\", exist_ok=True); shutil.rmtree(dst, ignore_errors=True); shutil.move(src, dst)
lines = open(\"/root/loras.txt\").read().split() if os.path.exists(\"/root/loras.txt\") else []
if not any(l.startswith(N + \"=\") for l in lines):
    open(\"/root/loras.txt\", \"a\").write(N + \"=\" + dst + \"\n\")
try:
    req = urllib.request.Request(\"http://127.0.0.1:8000/v1/load_lora_adapter\", method=\"POST\",
        data=json.dumps({\"lora_name\": N, \"lora_path\": dst}).encode(), headers={\"content-type\": \"application/json\"})
    r = urllib.request.urlopen(req, timeout=120).read().decode()[:200]
except Exception as e:
    r = \"load failed: \" + str(e)[:200]
print(\"[hf-dl]\", N, \"sha256\", h, \"match; load:\", r)
'"
done
