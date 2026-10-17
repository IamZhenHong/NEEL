"""Watch a P1 main pod and exit (one notification) at a decision point.

Reads the launcher log (heartbeats stream from the pod) and the pod's uptime from the RunPod API.
Exit conditions, checked every POLL seconds:
  FIRST   saved episodes first reach MIN_FRAC of planned (report the first projection)   [--once-first]
  TRIM    saved >= MIN_FRAC of planned AND projected cost > cap  (A3 rev uniform trim)
  FAIL    tracebacks > 0, a trajectory exited non-zero, FATAL, lost contact, or the job ended badly
  DONE    the job exited
  MARK    the elapsed wall time reached --until-min (periodic check-in)
Projection = spend so far / (saved / planned)  (A3; checks start at >= MIN_FRAC saved).

Usage: python3 -I watch_pod.py LAUNCH_LOG POD_ID CAP_USD PLANNED [--once-first] [--until-min M]
"""
import re
import sys
import time

sys.path.insert(0, __import__("os").path.join(__import__("os").getcwd(), "infra"))
from runpod_guard import account  # noqa: E402

POLL, RATE, MIN_FRAC = 120, 1.79, 0.20


def last_heartbeat(text):
    hb = [l for l in text.splitlines() if re.match(r"\[arms\] \d\d:\d\d \+\d+m running=", l)]
    return hb[-1] if hb else ""


def saved(hb):
    return sum(int(x) for x in re.findall(r":(\d+)ep", hb))


def main():
    log, pod, cap, planned = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    once_first = "--once-first" in sys.argv
    until = None
    if "--until-min" in sys.argv:
        until = time.time() + 60 * float(sys.argv[sys.argv.index("--until-min") + 1])
    while True:
        text = open(log, errors="replace").read()
        hb = last_heartbeat(text)
        n = saved(hb)
        up = None
        try:
            for p in account().pods:
                if p.get("id") == pod:
                    up = (p.get("runtime") or {}).get("uptimeInSeconds")
        except Exception as e:  # noqa: BLE001
            print(f"(account() failed: {e})", flush=True)
        spend = (up or 0) / 3600 * RATE
        proj = spend / (n / planned) if n else float("inf")
        tb = re.findall(r"tracebacks=(\d+)", hb)
        bad_done = re.findall(r"\[arms\] DONE (\S+) rc=([1-9]\d*)", text)
        status = (f"saved {n}/{planned} | pod uptime {(up or 0)/60:.0f} min, spend ${spend:.2f} | "
                  f"projection ${proj:.2f} vs cap ${cap:.2f} | {hb[:60]}")
        if "job exit code" in text or "terminated " + pod in text:
            print("DONE " + status, flush=True); return
        if (tb and int(tb[0]) > 0) or bad_done or "FATAL" in text or "lost contact" in text:
            print(f"FAIL tracebacks={tb} bad_done={bad_done[:5]} | " + status, flush=True); return
        if n >= MIN_FRAC * planned:
            if proj > cap:
                print("TRIM " + status, flush=True); return
            if once_first:
                print("FIRST " + status, flush=True); return
        if until and time.time() > until:
            print("MARK " + status, flush=True); return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
