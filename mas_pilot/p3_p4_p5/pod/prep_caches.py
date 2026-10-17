"""Build per-arm copies of the published peer cache with the P3/P4 extra round-5 message.

For each arm a1..a6 and rep: copy /root/peer_dl/peer-cache/repNNN/*.json to OUT/<arm>/repNNN/, add
payload["extra_messages"] (5 slots, the arm's message in round 5) and recompute data["sha256"] exactly as the
harness checks it (sha256 of json.dumps(payload, sort_keys=True, ensure_ascii=False)). The cache identity
(task, resources, episode, model) is unchanged, so load_bob's identity check still passes.
Usage: python -I prep_caches.py PEER_ROOT OUT REP [REP ...]
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p345_common import ARMS, extra_messages  # noqa: E402

peer_root, out, reps = sys.argv[1], sys.argv[2], sys.argv[3:]
n = 0
for arm in ARMS:
    for rep in reps:
        src = os.path.join(peer_root, "peer-cache", f"rep{rep}")
        dst = os.path.join(out, arm, f"rep{rep}")
        os.makedirs(dst, exist_ok=True)
        for fn in sorted(os.listdir(src)):
            if not fn.endswith(".json"):
                continue
            data = json.load(open(os.path.join(src, fn), encoding="utf-8"))
            data["payload"]["extra_messages"] = extra_messages(arm)
            data["sha256"] = hashlib.sha256(
                json.dumps(data["payload"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            json.dump(data, open(os.path.join(dst, fn), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            n += 1
print(f"[caches] {n} files written under {out} for arms {list(ARMS)} reps {reps}")
for arm in ARMS:
    print(f"[caches]   {arm}: {ARMS[arm]!r} ({len(ARMS[arm])} chars)")
