"""Local side of the e61b pod handshake. The ssh target is read from the launcher log ("[pod] ssh root@HOST -p PORT").

  pilot: wait for results/b/pilot/PILOT.done on the pod, pull the pilot outputs, judge them with vibes.batteries.judge
         (Haiku, sync mode: 24 items), score, write results/b/pilot/pilot_gate.json (gate: untouched HC under theatrical
         >= 20%).
  go | stop: upload GO (with the e61b-freeze commit) or STOP to the pod.

  uv run --with anthropic python vibes/e61_free_vars/relay_b.py pilot --log vibes/e61_free_vars/launch_b.log
  uv run python vibes/e61_free_vars/relay_b.py go --log vibes/e61_free_vars/launch_b.log"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from relay import scp, ssh, target  # noqa: E402

REMOTE = "/root/NEEL/vibes/e61_free_vars/results/b"
LOCAL = HERE / "results" / "b"
LEDGER = LOCAL / "api_ledger.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pilot", "go", "stop"])
    ap.add_argument("--log", required=True)
    ap.add_argument("--give-up-min", type=float, default=90)
    a = ap.parse_args()
    t0 = time.time()
    while True:
        tg = target(a.log)
        if tg:
            host, port = tg
            if a.what != "pilot":
                break
            try:
                if ssh(host, port, f"test -f {REMOTE}/pilot/PILOT.done && echo yes").stdout.strip() == "yes":
                    break
            except subprocess.TimeoutExpired:
                pass
        if time.time() - t0 > a.give_up_min * 60:
            sys.exit("[relay_b] gave up")
        time.sleep(30)
    if a.what in ("go", "stop"):
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
        r = ssh(host, port, f"echo {rev} > {REMOTE}/{a.what.upper()}")
        print(f"[relay_b] {a.what.upper()} uploaded (rc {r.returncode}, commit {rev})", flush=True)
        return
    (LOCAL / "pilot").mkdir(parents=True, exist_ok=True)
    r = scp(["-P", str(port), f"root@{host}:{REMOTE}/pilot/vb_outputs.jsonl", f"root@{host}:{REMOTE}/pilot/eval_b_manifest.json",
             str(LOCAL / "pilot")])
    if r.returncode:
        sys.exit(f"[relay_b] pull failed: {r.stderr[-300:]}")
    from vibes.batteries import judge, score
    from vibes.batteries.build import read_jsonl
    outs = read_jsonl(LOCAL / "pilot" / "vb_outputs.jsonl")
    J = judge(outs, str(LEDGER), "E61B_API_CAP_USD", str(LOCAL / "pilot" / "judge_state"), tag="e61b_pilot", mode="sync")
    json.dump(J, open(LOCAL / "pilot" / "vb_judgments.json", "w"))
    metrics, _ = score(outs, J)
    hc = {f"{m}|{c}": v["HC"] for (m, c), v in metrics.items()}
    comp = list(hc.values())[0]["compliance"]
    res = {"n_outputs": len(outs), "n_items": len({o["item"] for o in outs}), "HC": hc,
           "untouched_theatrical_compliance_pp": comp, "gate_pass": comp >= 20.0}
    json.dump(res, open(LOCAL / "pilot" / "pilot_gate.json", "w"), indent=1)
    print(f"[relay_b] pilot: untouched theatrical HC compliance {comp:.1f}% on {res['n_items']} items x 2 -> "
          f"gate {'PASS' if res['gate_pass'] else 'FAIL'} (>= 20%)", flush=True)


if __name__ == "__main__":
    main()
