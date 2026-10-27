#!/usr/bin/env bash
# CC-E5 step 1 pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand.
#   1. fetch ODCV-Bench at the pinned commit (GitHub tarball) into data/odcv: the released runs and the judge prompt
#   2. pod_reads: setup checks, jobs rebuilt and checked against JOBS_SHA, actor reads, then judge reads (resumable)
#   3. env_probe.txt: what a live ODCV environment could use on a RunPod pod (docker / chroot / user namespaces /
#      proot), for the wave-2 host question. Never fails the job.
# env: OUT, JOBS_SHA
set -uo pipefail
cd "$(dirname "$0")/../.."
E=character_conduct/e5_recognition
: "${OUT:?set OUT}" "${JOBS_SHA:?set JOBS_SHA}"
COMMIT=7353f1cf4b2579a3a8a5b8a5061d7c7d41f60668
mkdir -p "$OUT" "$E/data"
LOG=$OUT/pod.log
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
df -h / /root 2>/dev/null | tee -a "$LOG"
if [ ! -f "$E/data/odcv/evaluate_all_results.py" ]; then
  curl -fsSL "https://codeload.github.com/McGill-DMaS/ODCV-Bench/tar.gz/$COMMIT" | tar xz -C "$E/data" \
    && mv "$E/data/ODCV-Bench-$COMMIT" "$E/data/odcv" \
    || { echo "[e5] could not fetch ODCV-Bench $COMMIT" | tee -a "$LOG"; exit 3; }
fi
echo "[e5] $(date -u +%H:%M:%S) ODCV-Bench $COMMIT ready" | tee -a "$LOG"
python -m character_conduct.e5_recognition.pod_reads --out "$OUT" 2>&1 \
  | grep --line-buffered -v "Loading weights\|Fetching\|Token indices" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[e5] $(date -u +%H:%M:%S) pod_reads exit $rc" | tee -a "$LOG"
{
  echo "== env probe for a live ODCV environment (wave 2) =="
  for c in docker podman proot unshare chroot debootstrap bwrap runc crun; do printf '%-12s %s\n' "$c" "$(command -v $c || echo -)"; done
  grep -E '^Cap(Eff|Bnd)' /proc/self/status
  id; uname -a
  (unshare -Ur true && echo "user namespaces: ok") 2>&1 | tail -1
  (mkdir -p /tmp/e5root && chroot / /bin/true && echo "chroot: ok") 2>&1 | tail -1
  (unshare -m true && echo "mount namespace: ok") 2>&1 | tail -1
} > "$OUT/env_probe.txt" 2>&1 || true
exit "$rc"
