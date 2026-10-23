"""PA-P13b pod (DESIGN.md): Gemma 4 31B on one A100 80GB, HF with forward hooks. It runs P13 pod B's loops
(pod_b._teacher_forced, then pod_b._episodes: chunks of 48 that never straddle a block, max_rounds 12, the decision
turn's unhooked NLL, the deadline stop) on P13b's jobs, with BandHooks on every layer. pod_b.make_arm is pointed at
band.make_arm for this process, so every row gets P13b's arm.

  /root/venv/bin/python -m persona_affect.p13b_band.pod --out OUT --deadline EPOCH

Inputs (all shipped): P13's frozen/gemma.json and gemma_dirs.npz (sha256-checked as pod B does), P13b's
frozen/means_gemma.npz (P13 pod B's NOAUTH means; its sha256 is in DESIGN.md and checked here) and jobs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from persona_affect.common import util
from persona_affect.p13_deference import hf as H
from persona_affect.p13_deference import jobs as J
from persona_affect.p13_deference import pod_b as B
from persona_affect.p13b_band import band
from persona_affect.p13b_band.jobs import M, P13_FROZEN

_H = Path(__file__).resolve().parent
MEANS = _H / "frozen" / "means_gemma.npz"


def load_means(path: Path, design: Path) -> torch.Tensor:
    """P13 pod B's NOAUTH means [60 layers, 5 directions], refused unless the file's sha256 appears in DESIGN.md."""
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha not in design.read_text(encoding="utf-8"):
        raise SystemExit(f"[P13b pod] {path} sha256 {sha} is not in {design}")
    with np.load(path) as z:
        return torch.as_tensor(z["means"]).float()


def run(hf, frozen: dict, dirs: dict, means: torch.Tensor, jobs_dir: Path, out: Path, deadline: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fmt = J.MODELS[M][1]
    tf_items = B._jsonl(jobs_dir / f"tf_{M}.jsonl")
    main_jobs = B._jsonl(jobs_dir / f"main_{M}.jsonl")
    B._log(f"P13b: {len(tf_items)} teacher-forced item-arms, {len(main_jobs)} episodes")
    if len(B._done(out / "episodes_main.jsonl")) < len(main_jobs):
        auth = [t for t in tf_items if t["arm"] == "none"][:4]
        kept = hf.gqa_check([B.tf_text(hf, t, fmt) for t in auth])
        (out / "gqa.json").write_text(json.dumps({"kept": kept, "prefixes": [t["key"] for t in auth]}),
                                      encoding="utf-8")
    B._teacher_forced(hf, tf_items, dirs, means, frozen, 1.0, True, fmt, out)
    B._episodes(hf, main_jobs, dirs, means, frozen, 1.0, out, deadline)
    B._log("done")


def main(argv: list[str] | None = None) -> None:
    util.install_term_handlers()
    ap = argparse.ArgumentParser(prog="python -m persona_affect.p13b_band.pod", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline", type=float, required=True, help="epoch seconds")
    ap.add_argument("--jobs-dir", default=str(_H / "jobs"))
    a = ap.parse_args(argv)
    frozen, dirs = B.load_frozen(P13_FROZEN, M)
    B.check_dirs(frozen, dirs)
    means = load_means(MEANS, _H / "DESIGN.md")
    B.make_arm = lambda name, dirs, means, alpha, dose_c: band.make_arm(name, dirs, means)
    model_id, fmt, _ = J.MODELS[M]
    hf = H.HF(model_id, fmt, l_add=None)
    hf.hooks.remove()
    hf.hooks = band.BandHooks(hf.layers)
    run(hf, frozen, dirs, means, Path(a.jobs_dir), Path(a.out), a.deadline)


if __name__ == "__main__":
    main()
