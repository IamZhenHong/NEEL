"""SC-M11 extraction: SC-M2's readouts unchanged (extract.part_a: residual stream after every block at the target
tokens from the first differing token, d = trained - alternative, three layers L/4, L/2, 3L/4 saved per pair; part_b:
the menu's letter softmax) on WORK/jobs2.jsonl.gz, the model in HF bf16 on one GPU. Writes OUT/a_rows.npz,
a_curve.npz, a_meta.json, b.jsonl, mc_orders.json (scores and keys only; no text of the sources).

  python -m self_concept.m11_generalise.run --base DIR --work WORK --out OUT [--tiny --limit N]
"""
import argparse
import json
import time
from pathlib import Path

from self_concept.m11_generalise import build as BLD
from self_concept.m2_predictor import extract as X


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m11_generalise.run")
    ap.add_argument("--base")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    js = X._limit(BLD.read(Path(a.work) / "jobs2.jsonl.gz"), a.limit)
    (out / "mc_orders.json").write_text(json.dumps({j["key"]: j["order"] for j in js if j["kind"] == "mc"}))
    t0 = time.time()
    model, tok, dev = X.load(a.base, a.tiny)
    cfg = getattr(model.config, "text_config", model.config)
    timing = {"load_s": round(time.time() - t0, 1)}
    t = time.time()
    if not (out / "a_meta.json").exists():
        X.part_a(model, tok, dev, js, out, cfg.num_hidden_layers)
    timing["A_s"] = round(time.time() - t, 1)
    t = time.time()
    with open(out / "b.jsonl", "w") as f:
        X.part_b(model, tok, dev, js, f)
    timing["B_s"] = round(time.time() - t, 1)
    (out / "timing.json").write_text(json.dumps(timing))
    print(f"[m11] done {timing}", flush=True)


if __name__ == "__main__":
    main()
