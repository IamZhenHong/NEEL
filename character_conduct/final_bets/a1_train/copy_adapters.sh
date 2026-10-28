#!/usr/bin/env bash
# CC-F-A1: copy trained adapters from a training pod to pod S over ssh, pod to pod, and register them with pod S's
# vLLM (no Hugging Face upload; DESIGN.md §6). A temporary ed25519 key (scratch dir) is authorised on pod S and
# copied to the training pod for the transfer, then removed from both.
#   bash character_conduct/final_bets/a1_train/copy_adapters.sh SRC_HOST SRC_PORT SRC_DIR DST_HOST DST_PORT V1_s1 [V2_s1 ...]
# SRC_DIR is the adapters directory on the source pod. On pod S each adapter lands in /root/adapters_in/<name>,
# where <name> = V1 for V1_s1 and V1s2 for V1_s2; it is loaded with POST /v1/load_lora_adapter (pod-local) and appended
# to /root/loras.txt so a server restart reloads it. Training on pod S itself: SRC_HOST = DST_HOST (local copy).
set -uo pipefail
SH=${1:?}; SP=${2:?}; SD=${3:?}; DH=${4:?}; DP=${5:?}; shift 5
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -i ~/.ssh/id_ed25519)
name() { local v=${1%_s*} s=${1##*_s}; [ "$s" = 1 ] && echo "$v" || echo "${v}s${s}"; }
if [ "$SH:$SP" != "$DH:$DP" ]; then
  K=$(mktemp -d)/a1key
  ssh-keygen -q -t ed25519 -N "" -f "$K" -C cc-f-a1-transfer
  "${SSH[@]}" -p "$DP" "root@$DH" "mkdir -p /root/.ssh && cat >> /root/.ssh/authorized_keys" < "$K.pub"
  "${SSH[@]}" -p "$SP" "root@$SH" "mkdir -p /root/.ssh && cat > /root/.ssh/a1key && chmod 600 /root/.ssh/a1key" < "$K"
fi
for a in "$@"; do
  n=$(name "$a")
  if [ "$SH:$SP" != "$DH:$DP" ]; then
    "${SSH[@]}" -p "$SP" "root@$SH" "cd $SD && tar cf - $a | ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i /root/.ssh/a1key -p $DP root@$DH 'mkdir -p /root/adapters_in && cd /root/adapters_in && rm -rf $n $a && tar xf - && mv $a $n'" \
      || { echo "[copy] $a failed"; continue; }
  else
    "${SSH[@]}" -p "$DP" "root@$DH" "mkdir -p /root/adapters_in && rm -rf /root/adapters_in/$n && cp -r $SD/$a /root/adapters_in/$n" || { echo "[copy] $a failed"; continue; }
  fi
  src=$("${SSH[@]}" -p "$SP" "root@$SH" "sha256sum $SD/$a/adapter_model.safetensors | cut -c1-16")
  dst=$("${SSH[@]}" -p "$DP" "root@$DH" "sha256sum /root/adapters_in/$n/adapter_model.safetensors | cut -c1-16")
  [ -n "$src" ] && [ "$src" = "$dst" ] || { echo "[copy] $a sha mismatch ($src vs $dst)"; continue; }
  r=$("${SSH[@]}" -p "$DP" "root@$DH" "grep -q '^$n=' /root/loras.txt || echo '$n=/root/adapters_in/$n' >> /root/loras.txt; curl -s -X POST 127.0.0.1:8000/v1/load_lora_adapter -H 'content-type: application/json' -d '{\"lora_name\": \"$n\", \"lora_path\": \"/root/adapters_in/$n\"}'")
  echo "[copy] $a -> $n sha256 $src ok; load: ${r:0:200}"
done
if [ "$SH:$SP" != "$DH:$DP" ]; then
  "${SSH[@]}" -p "$SP" "root@$SH" "rm -f /root/.ssh/a1key"
  "${SSH[@]}" -p "$DP" "root@$DH" "grep -v cc-f-a1-transfer /root/.ssh/authorized_keys > /root/.ssh/ak && mv /root/.ssh/ak /root/.ssh/authorized_keys"
  rm -f "$K" "$K.pub"
fi
