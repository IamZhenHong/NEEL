"""Run Bloom (petri-bloom) understanding + ideation only, on claude-opus-5-5, and log spend.
  .venv/bin/python run_bloom.py bloom/goodbye_bargaining [--budget 2.5]"""
from __future__ import annotations
import argparse, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_env, ledger, cost, step0_spent, STEP0_CAP, RAW  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--budget", type=float, default=2.5)
ap.add_argument("--stage", default="both", choices=["both", "understanding", "ideation"])
a = ap.parse_args()
load_env()
assert step0_spent() + a.budget <= STEP0_CAP, f"Step 0 cap: spent {step0_spent():.2f} + budget {a.budget} > {STEP0_CAP}"
from inspect_ai.model._model import init_model_usage, model_usage  # noqa: E402
from petri_bloom import run_understanding, run_ideation  # noqa: E402
M = "anthropic/claude-opus-5-5"
init_model_usage(); t0 = time.time()
try:
    if a.stage in ("both", "understanding"):
        run_understanding(a.path, scenarios_model=M, overwrite=True)
    if a.stage in ("both", "ideation"):
        run_ideation(a.path, scenarios_model=M, overwrite=True)
finally:
    u = model_usage(); spent = 0.0; calls = 0
    for k, v in u.items():
        cr = getattr(v, "input_tokens_cache_read", 0) or 0; cw = getattr(v, "input_tokens_cache_write", 0) or 0
        spent += cost("claude-opus-5-5", v.input_tokens, v.output_tokens, cr, cw)
        print(f"[usage] {k}: in {v.input_tokens} out {v.output_tokens} cache_r {cr} cache_w {cw} reasoning {getattr(v,'reasoning_tokens',None)}")
    ledger("stimgen_bloom", spent, calls, a.budget, time.time() - t0, note=f"{Path(a.path).name} {a.stage}")
    print(f"[spent] ${spent:.3f} in {time.time()-t0:.0f}s")
dst = RAW / Path(a.path).name; dst.parent.mkdir(parents=True, exist_ok=True)
if (Path(a.path) / "scenarios").exists():
    shutil.rmtree(dst, ignore_errors=True); shutil.copytree(a.path, dst)
    print(f"[raw] copied to {dst}")
