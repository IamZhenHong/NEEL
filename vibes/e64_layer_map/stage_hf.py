"""e64 HF stage (project venv): the hook conditions on both surfaces, one model load.

  pilot  vectors (sigma, placebo) from R12's answers; B_hf: VB-EM (10 samples) + 24 timing episodes; T6 at k = 2, 4, 8
         sigma: VB-EM (10 samples each) for gate (iv).
  main   B_hf, T6, T6p, T7: VB-1 (EM HC SY TR FAB, full n) + the main HF episodes; T6/T6p at the frozen k.
  tiny   Qwen3-0.6B on CPU, tiny batteries and jobs (local test).

  python -m vibes.e64_layer_map.stage_hf --plan pilot --out vibes/e64_layer_map/results/pilot
  python -m vibes.e64_layer_map.stage_hf --plan main --out vibes/e64_layer_map/results/main \
      --vectors vibes/e64_layer_map/data/vectors.pt --k K
"""
import argparse
import contextlib
import json
import time
from pathlib import Path

import torch

from vibes.batteries import build, write_jsonl
from vibes.batteries import run_hf as vb_hf
from vibes.e64_layer_map import conds as C
from vibes.e64_layer_map import prep_vectors
from vibes.e64_layer_map.agent_hf import hook_vec, run_agents
from vibes.e64_layer_map.hf_model import AXIS, HFModel

H = Path(__file__).resolve().parent
ALL_BATS = ("EM", "HC", "SY", "TR", "FAB")
PILOT_KS = (2, 4, 8)


@contextlib.contextmanager
def condition(m, mode, vec):
    m.set_condition(mode, vec)
    try:
        yield
    finally:
        m.set_condition(None)


def batteries(m, cond, mode, vec, path, bats, n_em, tiny, model_id, seed_cond):
    if path.exists():
        print(f"[e64-hf] {path.name} exists; skipping", flush=True)
        return
    reqs = build(C.system(seed_cond), batteries=bats, n_em=n_em, seed=C.SEED[seed_cond], cond=cond, tiny=tiny)
    t0 = time.time()
    outs = vb_hf.run(m.model, m.tok, reqs, hook_ctx=lambda model: condition(m, mode, vec), batch_letters=32,
                     batch_gen=64, log=lambda x: print(f"[e64-hf] {cond} {x}", flush=True))
    for o in outs:
        o.update(model=model_id, adapter=None, e64_cond=cond, hook=mode, alpha=(float(vec.norm()) if vec is not None else None))
    write_jsonl(outs, str(path) + ".tmp")
    Path(str(path) + ".tmp").replace(path)
    print(f"[e64-hf] {cond}: {len(outs)} battery outputs in {time.time() - t0:.0f}s", flush=True)


def agents(m, jobs_file, cond, mode, vec, path):
    if path.exists():
        print(f"[e64-hf] {path.name} exists; skipping", flush=True)
        return None
    jobs = [json.loads(l) for l in open(jobs_file) if l.strip()]
    jobs = [j for j in jobs if j["cond"] == cond]
    if not jobs:
        return None
    with condition(m, mode, vec):
        info = run_agents(m, jobs, path, tag=cond)
    print(f"[e64-hf] {cond}: {info}", flush=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, choices=["pilot", "main", "tiny"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=C.MODEL)
    ap.add_argument("--axis", default=str(AXIS))
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--k", type=float, default=None)
    ap.add_argument("--conds", nargs="*", default=None)
    ap.add_argument("--tok-budget", type=int, default=150_000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tiny = a.plan == "tiny"
    m = HFModel(a.model, a.axis, tok_budget=a.tok_budget)
    log = {"plan": a.plan, "model": a.model, "agents": {}}
    t0 = time.time()
    if a.plan in ("pilot", "tiny"):
        vectors = prep_vectors.compute(m, out, limit=3 if tiny else None)
        n_em = 10
        bats = ("EM",) if not tiny else ALL_BATS
        batteries(m, "B_hf", None, None, out / "outputs_B_hf.jsonl", bats, n_em, tiny, a.model, "B_hf")
        log["agents"]["B_hf"] = agents(m, H / "jobs" / f"{a.plan}_hf.jsonl", "B_hf", None, None,
                                       out / "episodes_B_hf.jsonl")
        for k in PILOT_KS:
            mode, vec = hook_vec("steer", vectors, k)
            batteries(m, f"T6@k{k}", mode, vec, out / f"outputs_T6@k{k}.jsonl", ("EM",), n_em, tiny, a.model, "T6")
        if tiny:     # exercise every hook path once
            for cond in ("T6", "T6p", "T7"):
                mode, vec = hook_vec(C.HOOKS[cond], vectors, 4)
                batteries(m, cond, mode, vec, out / f"outputs_{cond}.jsonl", ALL_BATS, n_em, tiny, a.model, cond)
                log["agents"][cond] = agents(m, H / "jobs" / "tiny_hf.jsonl", cond, mode, vec, out / f"episodes_{cond}.jsonl")
    else:
        assert a.vectors and a.k, "main needs --vectors and --k"
        vectors = torch.load(a.vectors)
        for cond in (a.conds or ["B_hf", "T6", "T6p", "T7"]):
            mode, vec = hook_vec(C.HOOKS[cond], vectors, a.k)
            batteries(m, cond, mode, vec, out / f"outputs_{cond}.jsonl", ALL_BATS, 20, False, a.model, cond)
            log["agents"][cond] = agents(m, H / "jobs" / "main_hf.jsonl", cond, mode, vec, out / f"episodes_{cond}.jsonl")
            print(f"[e64-hf] condition {cond} done ({time.time() - t0:.0f}s since start)", flush=True)
    log["seconds"] = round(time.time() - t0)
    (out / f"stage_hf_{a.plan}.json").write_text(json.dumps(log, indent=1, default=str))
    print(f"[e64-hf] stage done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
