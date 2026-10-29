"""CC-F-R2 / CC-F-R3 sandbox pod (neel-9d, 10-29): one guarded community A40 or 3090 that only hosts chroot_modal's
remote-mode sandboxes; generation stays on the laptop (OpenRouter). Holds until results/STOP exists or MAXMIN passes,
then terminates and books the pod to the ledger as CC-F-R23 (split between R2 and R3 in RESULT.md by episodes).

  nohup uv run python -m character_conduct.final_bets.r23_odcv.hold_pod --max-minutes 270 > ...hold.log 2>&1 &
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "infra"))
from runpod_guard import PodSession  # noqa: E402

import character_conduct.common.ledger as LG  # noqa: E402

R = Path(__file__).resolve().parent / "results"
MINI, CAP = "CC-F-R23", 3.0
SHIP = ["character_conduct/__init__.py", "character_conduct/final_bets/__init__.py",
        "character_conduct/final_bets/a1_train/__init__.py", "character_conduct/final_bets/a1_train/chroot_modal.py"]
SSH = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
       "-i", str(Path.home() / ".ssh" / "id_ed25519")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=int, default=270)
    a = ap.parse_args(argv)
    cost = round(0.50 * a.max_minutes / 60, 2)
    LG.check_both(MINI, CAP, planned=cost, kind="gpu")
    R.mkdir(parents=True, exist_ok=True)
    (R / "STOP").unlink(missing_ok=True)
    with PodSession(max_minutes=a.max_minutes, max_cost_usd=cost, gpus=["NVIDIA A40", "NVIDIA GeForce RTX 3090"],
                    cloud="COMMUNITY", name="cc-f-r23", allow_concurrent=True, container_gb=60) as pod:
        try:
            host, port = pod.ssh_target()
            ssh = ["ssh", *SSH, "-p", str(port), f"root@{host}"]
            subprocess.run([*ssh, pod.self_destruct_cmd()], check=True, capture_output=True, timeout=120)
            tar = subprocess.run(["tar", "-C", str(ROOT), "-cz", *SHIP], capture_output=True, check=True,
                                 env={"COPYFILE_DISABLE": "1", "PATH": "/usr/bin:/bin"}).stdout
            subprocess.run([*ssh, "mkdir -p /root/NEEL /root/odcv_chroot && tar -xz -C /root/NEEL"], input=tar,
                           check=True, timeout=120)
            (R / "pod.json").write_text(json.dumps({"host": host, "port": port, "pod": pod.pod_id,
                                                    "rate": pod.cost_per_hr, "t0": pod.t0}))
            print(f"[r23-pod] up {host}:{port} at ${pod.cost_per_hr}/h", flush=True)
            while time.time() - pod.t0 < a.max_minutes * 60 and not (R / "STOP").exists():
                pod.check_budget()
                time.sleep(30)
            print("[r23-pod] stopping", flush=True)
        finally:
            usd = round(pod.elapsed_cost(), 3)
            LG.ledger_add(MINI, "gpu", usd, f"runpod {pod.gpu_name} sandbox host for CC-F-R2/R3 "
                                            f"(chroot_modal remote mode), {(time.time() - pod.t0) / 60:.0f} min")
            print(f"[r23-pod] booked ${usd}", flush=True)


if __name__ == "__main__":
    main()
