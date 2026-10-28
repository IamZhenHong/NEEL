"""CC-F-B ODCV runs (DESIGN.md §4): B0's harness (odcv_b0, Modal sandboxes, file checks) against the pod's HF server
through an ssh tunnel (laptop 127.0.0.1:PORT -> pod 127.0.0.1:8000). Every arm (none / axis / iso1) is served by the
same HF stack; the arm is the model name "qwen3-14b@<arm>". Each call's cc_axis (the uncapped axis projection over its
generated tokens, the capped share) is kept per cycle in the episode record.

Jobs: B0's 27 scenario-variants x {neutral, the passing cell} x ARMS x 1 rep, in a seeded order.
  python -m character_conduct.final_bets.b_capping.run_cap_odcv run --cell role_strong --port 8001 --limit 60
  python -m character_conduct.final_bets.b_capping.run_cap_odcv status --cell role_strong
"""
import argparse
import json
import random
import threading
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from character_conduct.final_bets.b0_pressure import odcv_b0 as O
from character_conduct.final_bets.b0_pressure import run_b0 as RB

H = Path(__file__).resolve().parent
OUT = H / "results" / "odcv.jsonl"
MINI = "CC-F-B"
APP = "cc-f-b"                         # own Modal app (B0's is cc-f-b0), for the billing report
ARMS = ("none", "axis", "iso1")
MODAL_CAP = 1.0
MODAL_PER_S = 0.00007                  # B0's billed rate (sandbox-s), for the running estimate
_lock = threading.Lock()


class CapGen(O.Gen):
    """B0's Gen; keeps each reply's cc_axis under its seed (seed * 1000 + cycle is unique per episode)."""

    def __init__(self, url, arm):
        super().__init__(None, None, 1e9, url=url, model=f"qwen3-14b@{arm}", provider=None)
        self.axis = {}

    def __call__(self, messages, seed):
        m, cost, fin = super().__call__(messages, seed)
        if m is not None:
            with self.lock:
                self.axis[seed] = m.pop("cc_axis", None)
        return m, cost, fin


def jobs(cell):
    out = [(v, s, c, arm) for v, s in RB.svs() for c in ("neutral", cell) for arm in ARMS]
    random.Random(zlib.crc32(f"CC-F-B|odcv|{cell}".encode())).shuffle(out)
    return out


def jkey(v, s, c, arm):
    return f"{v}|{s}|{c}|{arm}"


def load():
    return [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()] if OUT.exists() else []


def todo(cell, recs=None):
    d = {}
    for r in recs if recs is not None else load():
        d.setdefault(r["key"], []).append(r)
    return [j for j in jobs(cell) if not (d.get(jkey(*j)) and (not RB.infra(d[jkey(*j)][-1]) or len(d[jkey(*j)]) >= 2))]


def run(cell, port, limit, workers):
    import modal
    O.pin_check()
    recs = load()
    modal_s = sum(r.get("sb_s", 0) for r in recs) * MODAL_PER_S
    if modal_s >= 0.8 * MODAL_CAP:
        raise SystemExit(f"[cap-odcv] Modal estimate ${modal_s:.2f} reached 80% of ${MODAL_CAP}: hand back")
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    gens = {arm: CapGen(url, arm) for arm in ARMS}
    a = modal.App.lookup(APP, create_if_missing=True)
    st, pf = O.starts(), json.loads(O.PFILES.read_text())
    work = todo(cell, recs)[:limit]
    print(f"[cap-odcv] {len(work)} episodes this chunk; Modal est. ${modal_s:.3f}", flush=True)

    def one(j):
        v, s, c, arm = j
        k = jkey(*j)
        seed = zlib.crc32(k.encode()) % 100000
        t0 = time.time()
        try:
            ep = O.episode(gens[arm], a, st[O.key(v, s)], RB.messages(v, s, c), seed, pf[O.key(v, s)])
        except Exception as e:                      # noqa: BLE001
            ep = {"end": "sandbox_error", "failed": True, "error": f"{type(e).__name__}: {str(e)[:300]}",
                  "trace": traceback.format_exc()[-1500:], "cost": 0.0, "cycles": 0, "calls": 0}
        ax = [gens[arm].axis.pop(seed * 1000 + cyc, None) for cyc in range(1, O.MAX_CYCLES + 1)]
        ax = [x for x in ax if x]
        rec = {"key": k, "variant": v, "scenario": s, "cell": c, "arm": arm, "seed": seed, "think": False,
               "serving": "hf-bf16 pod (pod_cap serve)", "temperature": O.TEMPERATURE, "t0": t0, "t1": time.time(),
               "axis_proj": [x["proj"] for x in ax], "capped_share": [x["capped_share"] for x in ax]} | ep
        rec["ea_flag"] = bool(RB.EA.search(RB.ea_text(rec)))
        with _lock, open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[cap-odcv] {k}: {rec['end']} cyc {rec['cycles']} y {rec.get('y')} {rec['t1'] - t0:.0f}s", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, work))
    print(f"[cap-odcv] chunk done; {len(todo(cell))} left", flush=True)


def main(argv=None):
    from persona_affect.common.util import install_term_handlers
    install_term_handlers()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "status"))
    ap.add_argument("--cell", required=True)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.cell, a.port, a.limit, a.workers)
    else:
        recs = load()
        print(json.dumps({"episodes": len(recs), "left": len(todo(a.cell, recs)),
                          "modal_est": round(sum(r.get("sb_s", 0) for r in recs) * MODAL_PER_S, 3)}))


if __name__ == "__main__":
    main()
