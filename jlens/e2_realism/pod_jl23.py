"""One pod, one model load, both cards (jl2 jlens/e2_realism, jl3 jlens/e3_hallucination), Qwen3.6-27B.
Order: lenses + self-check + spider check (jlens G1 repeated as a setup check), jl2 reads, jl3 greedy + reads,
WorkspaceBench hallucination capture. Each stage saves on its own; a failed stage is logged and the next one runs.
  python -m jlens.e2_realism.pod_jl23 --out2 jlens/e2_realism/results --out3 jlens/e2_realism/results/jl3
Local CPU test of every code path: uv run python -m jlens.e2_realism.pod_jl23 --tiny --limit 6 --out2 <d> --out3 <d>"""
import argparse
import json
import os
import traceback
from pathlib import Path

import torch

from jlens.e2_realism.lens_core import MODEL, Reader, load_lenses, load_model, log, selfcheck, spider_check
from jlens.e2_realism.reads_jl2 import run_jl2
from jlens.e3_hallucination.reads_jl3 import run_jl3, run_wsb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out2", required=True)
    ap.add_argument("--out3", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tiny", action="store_true", help="CPU test on a shrunk random model with random lenses")
    ap.add_argument("--stages", nargs="+", default=["jl2", "jl3", "wsb"])
    a = ap.parse_args()
    out2, out3 = Path(a.out2), Path(a.out3)
    out2.mkdir(parents=True, exist_ok=True)
    out3.mkdir(parents=True, exist_ok=True)
    import transformers
    tok, model, text = load_model(a.tiny)
    R = Reader(tok, model, text)
    info = {"model": MODEL, "model_revision": getattr(model.config, "_commit_hash", None), "torch": torch.__version__,
            "transformers": transformers.__version__, "git": os.environ.get("GIT_REV"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    load_lenses(R, a.tiny, info)
    info["selfcheck_maxdiff"] = selfcheck(R)
    info["spider_check"] = spider_check(R)
    for o in (out2, out3):
        (o / "run_info.json").write_text(json.dumps(info, indent=1))
    failed = []
    for name, fn, o in (("jl2", run_jl2, out2), ("jl3", run_jl3, out3), ("wsb", run_wsb, out3)):
        if name not in a.stages:
            continue
        try:
            fn(R, o, a.limit)
        except Exception:
            failed.append(name)
            log(f"STAGE {name} FAILED:\n{traceback.format_exc()}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log(f"all done; failed stages: {failed}")
    if failed:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
