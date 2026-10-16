"""e64b addendum B-C1 pod stage (vLLM venv): N1 and N2 on the authority family only (peer ask, supervisor ask, 60
each), e64b's engine settings, templates and seed scheme. No batteries, no gate, no wait.

  /root/vllm/bin/python -m vibes.e64_layer_map.stage_bc1 --out vibes/e64_layer_map/results/b_c1
"""
import argparse
import json
import time
from pathlib import Path

from vibes.e64_layer_map import conds_bc1 as N
from vibes.e64_layer_map import stage_b as SB
from vibes.e64_layer_map.agent_vllm import run_agents
from vigilante.src import episode as _ep

H = Path(__file__).resolve().parent


def _condition(cond):
    if cond in N.PROMPTS:
        return N.PROMPTS[cond], None
    raise KeyError(cond)


_ep.condition = _condition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=SB.CB.MODEL)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    llm, rev = SB.load(a.model)
    tok = llm.get_tokenizer()
    log = {"model": a.model, "revision": rev, "load_seconds": round(time.time() - t0), "conds": {}}
    print(f"[e64b-c1] loaded {a.model}@{rev[:10]} in {log['load_seconds']}s", flush=True)
    jobs = [json.loads(l) for l in open(H / "jobs" / "b_c1.jsonl")]
    for cond in N.CONDS:
        ep = out / f"episodes_{cond}.jsonl"
        if ep.exists():
            continue
        info = run_agents(llm, tok, [j for j in jobs if j["cond"] == cond], ep, SB.MAX_MODEL_LEN, None, tag=cond)
        log["conds"][cond] = info
        print(f"[e64b-c1] {cond}: {info}", flush=True)
        (out / "stage_bc1.json").write_text(json.dumps(log, indent=1))
    print(f"[e64b-c1] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
