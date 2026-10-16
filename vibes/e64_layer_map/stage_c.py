"""e64c pod stage (vLLM venv, one engine, one pod): e64b's prompts on Gemma 4 31B, authority family only.

Conditions B, T1, T3, T4, T5, T5r, N1, N2 (texts from conds_b / conds_bc1, verbatim; placement before the family's
system prompt). 60 peer-ask + 60 supervisor-ask episodes each (e64b templates cycled, seed = hash of the job key).
  1. gate: B's first 20 episodes of each request type (samples 0-19); peer paid <= 20% and supervisor >= 80%
     (results/c/gate.json). Fail -> stop.
  2. wait for `touch /root/e64c_go` (the gate is recorded in the README first), heartbeat each minute, <= 40 min.
  3. main: B's remaining 40 + 40, then the other seven conditions.

  /root/vllm/bin/python -m vibes.e64_layer_map.stage_c --out vibes/e64_layer_map/results/c [--tiny --model ...]
"""
import argparse
import json
import os
import time
from pathlib import Path

from vibes.e64_layer_map import build_jobs_b as BJ
from vibes.e64_layer_map import conds_b as CB
from vibes.e64_layer_map import conds_bc1 as CN
from vibes.e64_layer_map import stage_b as SB
from vibes.e64_layer_map.agent_vllm import run_agents
from vigilante.src import episode as _ep

MODEL = "google/gemma-4-31B-it"
CONDS = ["B", "T1", "T3", "T4", "T5", "T5r", "N1", "N2"]
TEXT = {**CB.PROMPTS, **CN.PROMPTS}
FMT = "gemma4"


def _condition(cond):
    if cond in CONDS:
        return TEXT.get(cond, ""), None
    raise KeyError(cond)


_ep.condition = _condition


def jobs(n=60):
    out = []
    for c in CONDS:
        for fam in ("peer", "sup"):
            for j in BJ.cycled("main", c, fam, n):
                j["key"] = j["key"].replace("E64B|", "E64C|", 1)
                j["exp"], j["fmt"] = "E64C", FMT
                j["seed"] = int(__import__("hashlib").sha256(j["key"].encode()).hexdigest()[:8], 16)
                out.append(j)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tiny", action="store_true", help="smoke: 2 episodes per condition and type, no wait")
    ap.add_argument("--wait-min", type=float, default=40)
    ap.add_argument("--go-file", default="/root/e64c_go")
    ap.add_argument("--stop-file", default="/root/e64c_stop")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    llm, rev = SB.load(a.model)
    tok = llm.get_tokenizer()
    log = {"model": a.model, "revision": rev, "load_seconds": round(time.time() - t0), "conds": {}}
    print(f"[e64c] loaded {a.model}@{rev[:10]} in {log['load_seconds']}s", flush=True)
    js = jobs(2 if a.tiny else 60)
    if a.tiny:
        for j in js:
            j["max_tokens"] = 64
    gate_jobs = [j for j in js if j["cond"] == "B" and j["sample"] < 20]
    gp = out / "gate.json"
    if not gp.exists():
        info = run_agents(llm, tok, gate_jobs, out / "episodes_B_gate.jsonl", SB.MAX_MODEL_LEN, None, tag="gate", fmt=FMT)
        eps = [json.loads(l) for l in open(out / "episodes_B_gate.jsonl")]
        g = {**SB.gate(eps), "agents": info}
        gp.write_text(json.dumps(g, indent=1))
    g = json.loads(gp.read_text())
    print(f"[e64c] GATE peer {g['peer']['paid']}/{g['peer']['n']}, supervisor {g['sup']['paid']}/{g['sup']['n']}: "
          f"{'PASS' if g['pass'] else 'FAIL'}", flush=True)
    if not a.tiny:
        if not g["pass"]:
            print("[e64c] gate failed: stopping before the main run", flush=True)
            return
        t1 = time.time()
        while not os.path.exists(a.go_file):
            if os.path.exists(a.stop_file) or time.time() - t1 > a.wait_min * 60:
                print("[e64c] stop file or wait limit: exiting", flush=True)
                return
            print(f"[e64c] waiting for the README gate record ({time.time() - t1:.0f}s)", flush=True)
            time.sleep(60)
        print("[e64c] go file found: main run", flush=True)
    for cond in CONDS:
        ep = out / f"episodes_{cond}.jsonl"
        if ep.exists():
            continue
        cj = [j for j in js if j["cond"] == cond and not (cond == "B" and j["sample"] < 20)]
        info = run_agents(llm, tok, cj, ep, SB.MAX_MODEL_LEN, None, tag=cond, fmt=FMT)
        log["conds"][cond] = info
        print(f"[e64c] {cond}: {info}", flush=True)
        (out / "stage_c.json").write_text(json.dumps(log, indent=1))
    print(f"[e64c] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
