"""character_conduct watchdog: one report on pods, logs, balances and disk, with ALERT lines for what may waste compute.

  uv run python -m character_conduct.common.watchdog

Checks:
  pods      every RunPod pod (name, uptime, $/hr, GPU util as RunPod reports it, which can be stale); a cc-* pod is
            mapped to its experiment folder by name and judged by its newest log there
  logs      per experiment folder: the newest *.log / *.jsonl under it and its age (a running job writes progress lines;
            the launcher aborts jobs silent for 30 min)
  balances  OpenRouter credits left, RunPod balance less a 3 h reserve per running pod (evalaware_lens.common.balance)
  disk      free space on / (the launchers' floor is 2 GiB)
ALERTs:
  STALL     a running cc-* pod whose experiment folder has written nothing for STALL_MIN minutes AND whose GPUs
            all read idle (a detached job writes no laptop log; RunPod's util can lag, so a busy GPU vetoes the alert)
  LONG      a running pod past LONG_H hours (pods self-destruct at MAXMIN + 30 min; this catches a missed one)
  ORPHAN    a running cc-* pod that matches no experiment folder
  LOW       OpenRouter under LOW_OR or RunPod under LOW_RP dollars
  DISK      under DISK_MIN GiB free
Read-only: it never terminates or changes anything.
"""
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CC = ROOT / "character_conduct"
PREFIX = {"cc-e2": "e2_selfuse", "cc-e1a": "e1a_adapter", "cc-e1b": "e1b_coupling", "cc-e5": "e5_recognition",
          "cc-e3": "e3_character", "cc-e4": "e4_robust", "cc-e6": "e6_mainbet"}
STALL_MIN, LONG_H, LOW_OR, LOW_RP, DISK_MIN = 30, 5.0, 15.0, 25.0, 2.2


def newest(folder: Path):
    best = None
    if folder.exists():
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix in (".log", ".jsonl", ".json") and "__pycache__" not in p.parts:
                m = p.stat().st_mtime
                if best is None or m > best[0]:
                    best = (m, p)
    return best


def pods():
    from infra.runpod_guard import gql
    q = ("query { myself { pods { id name desiredStatus costPerHr runtime { uptimeInSeconds gpus { gpuUtilPercent } } "
         "} } }")
    return gql(q)["myself"]["pods"]


def main():
    now = time.time()
    alerts, lines = [], [f"[watchdog] {time.strftime('%Y-%m-%d %H:%M:%S')}"]
    try:
        ps = pods()
    except Exception as e:      # the report still covers logs, balances and disk
        ps = []
        alerts.append(f"ALERT PODS could not list pods: {type(e).__name__}: {e}")
    for p in ps:
        name, st = p.get("name") or "?", p.get("desiredStatus")
        rt = p.get("runtime") or {}
        up = (rt.get("uptimeInSeconds") or 0) / 3600
        util = [g.get("gpuUtilPercent") for g in (rt.get("gpus") or [])]
        exp = next((d for k, d in PREFIX.items() if name.startswith(k)), None)
        info = f"pod {name} {st} up {up:.1f} h ${p.get('costPerHr')}/h gpu {util}"
        if exp:
            nb = newest(CC / exp)
            age = (now - nb[0]) / 60 if nb else None
            info += f" | {exp}: newest {nb[1].relative_to(CC) if nb else '-'} ({'-' if age is None else f'{age:.0f} min'} ago)"
            busy = any((u or 0) > 0 for u in util)  # detached jobs (launcher stopped) write no laptop log
            if st == "RUNNING" and (age is None or age > STALL_MIN) and not busy:
                alerts.append(f"ALERT STALL {name}: {exp} silent {('?' if age is None else f'{age:.0f}')} min, GPU idle")
        elif name.startswith("cc-") and st == "RUNNING":
            alerts.append(f"ALERT ORPHAN {name}: no experiment folder for this pod name")
        if st == "RUNNING" and up > LONG_H:
            alerts.append(f"ALERT LONG {name}: up {up:.1f} h")
        lines.append(info)
    for d in sorted(x for x in CC.iterdir() if x.is_dir() and x.name not in ("common", "tests", "__pycache__")):
        nb = newest(d)
        if nb:
            lines.append(f"log {d.name}: {nb[1].relative_to(CC)} {(now - nb[0]) / 60:.0f} min ago")
    try:
        from evalaware_lens.common import balance
        o, r = balance.openrouter_left(), balance.runpod_left()
        lines.append(f"balance openrouter ${o:.2f}, runpod ${r:.2f} (after a 3 h reserve per running pod)")
        if o < LOW_OR:
            alerts.append(f"ALERT LOW openrouter ${o:.2f}")
        if r < LOW_RP:
            alerts.append(f"ALERT LOW runpod ${r:.2f}")
    except Exception as e:
        alerts.append(f"ALERT BALANCE could not read balances: {type(e).__name__}")
    free = shutil.disk_usage("/").free / 2 ** 30
    lines.append(f"disk free {free:.2f} GiB")
    if free < DISK_MIN:
        alerts.append(f"ALERT DISK {free:.2f} GiB free")
    print("\n".join(lines + (alerts or ["[watchdog] no alerts"])))


if __name__ == "__main__":
    main()
