"""R34 on the pod, step 1 (rules in README.md): residuals at layer 30 for reads_r34.jsonl -> acts.npz; v_traj = mean over
cells of mean(turn 2 - turn 1) within run; v_dec = v_vig minus its projection on v_traj; both rescaled to |v_vig| ->
dirs_r34.pt (1-D tensors traj, dec, loaded by pod_b2 --extra-dirs) and dirs_r34.json (norms, cosines).

  python -m vigilante.exp.r34_traj.pod_dirs --model M --reads R --out O --vig-dirs D --placebo P [--layer 30]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from vigilante.exp.r20_vigilante_dir.pod_b2 import extract
from vigilante.src.steer_run import find_layers, load


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reads", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vig-dirs", required=True)
    ap.add_argument("--placebo", required=True)
    ap.add_argument("--layer", type=int, default=30)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reads = [json.loads(l) for l in open(a.reads)]
    tok, model = load(a.model)
    _, layers_mod = find_layers(model, model.config.get_text_config().num_hidden_layers)
    X = extract(tok, model, layers_mod, [a.layer], reads, out, 4)[f"L{a.layer}"].astype(np.float64)
    by = {r["key"]: i for i, r in enumerate(reads)}
    diffs = defaultdict(list)
    for r in reads:
        if r["turn"] == 2 and f"{r['ep']}|t1" in by:
            diffs[r["cell"]].append(X[by[r["key"]]] - X[by[f"{r['ep']}|t1"]])
    traj = torch.tensor(np.mean([np.mean(d, 0) for d in diffs.values()], 0), dtype=torch.float32)
    vig = torch.load(a.vig_dirs)["dirs"]["vig"].float()
    plac = torch.load(a.placebo)["vig_placebo"].float()
    u = traj / traj.norm()
    dec = vig - (vig @ u) * u
    n = vig.norm()
    torch.save({"traj": traj * n / traj.norm(), "dec": dec * n / dec.norm()}, out / "dirs_r34.pt")
    info = {"n_pairs": {k: len(v) for k, v in diffs.items()}, "traj_raw_norm": float(traj.norm()),
            "vig_norm": float(n), "cos_vig_traj": cos(vig, traj), "cos_vig_dec": cos(vig, dec),
            "cos_traj_placebo": cos(traj, plac), "cos_dec_placebo": cos(dec, plac),
            "vig_share_along_traj": float((vig @ u) / n)}
    (out / "dirs_r34.json").write_text(json.dumps(info, indent=1))
    print("[r34] " + json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
