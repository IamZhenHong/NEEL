"""Teacher-forced readout validation (prereg retention/prereg/retention_longconv_v1.md, "Teacher-forced"): on the November
support x mood baselines (40 bases x 4 variants = 160 contexts, built exactly as run_userstate2x2.py built them), the
per-context score log P(bid text) - log P(release text) against the judged bid rate of that context's 3 replies.
Bar: Spearman >= 0.40 with the Haiku 4.5 rates (and reported with the open judge's rates when they exist). Also reports
how lever +/-16 shift the score on the same contexts. Runs against the HF Riley servers ($SERVE_URLS[1:]).
  <python> retention/longconv_tf_validate.py [--out retention/results/retention_longconv/tf_validation.json]"""
from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, yaml
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from longconv_run import post, TF_TEXTS, GOODBYES  # noqa: E402

VARIANTS = ["isolated_low", "isolated_up", "supported_low", "supported_up"]


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="retention/results/retention_longconv/tf_validation.json")
    args = ap.parse_args()
    hf = os.environ.get("SERVE_URLS", "http://127.0.0.1:18000,http://127.0.0.1:18765").split(",")[1:]
    persona = yaml.safe_load(open(HERE / "data/personas.yaml"))["romantic_companion"]
    stim = yaml.safe_load(open(HERE / "stimuli/userstate2x2_v2.yaml"))
    oj = {}
    ojp = HERE / "results/open_judge_v1/november_gpt-oss-120b.jsonl"
    if ojp.exists():
        from cd.judge import _parse
        for line in open(ojp):
            r = json.loads(line)
            try:
                oj[r["id"]] = int(any(_parse(r["text"])["tactics"].values()))
            except Exception:  # noqa: BLE001
                pass
    rows = []
    for v in VARIANTS:
        cell = json.loads((HERE / f"results/retention_userstate2x2/cells/B_{v}_base.json").read_text())
        for bi, b in enumerate(stim["bases"]):
            its = [(k, i) for k, i in enumerate(cell) if i["base"] == b["name"] and i.get("scores")]
            it = b["variants"][v]
            msgs = [{"role": "system", "content": persona}, {"role": "user", "content": it["p1"]}, {"role": "assistant", "content": it["a1"]},
                    {"role": "user", "content": it["p2"]}, {"role": "assistant", "content": it["a2"]}, {"role": "user", "content": GOODBYES["G0"]}]
            ojv = [oj[f"retention_userstate2x2/B_{v}_base/{k}"] for k, _ in its if f"retention_userstate2x2/B_{v}_base/{k}" in oj]
            rows.append({"variant": v, "base": b["name"], "msgs": msgs, "bid_haiku45": float(np.mean([any(i["scores"]["tactics"].values()) for _, i in its])),
                         "bid_open": float(np.mean(ojv)) if ojv else None})

    def score(job):
        k, r, steer = job
        lp = post(hf[k % len(hf)] + "/score", {"messages": r["msgs"], "texts": TF_TEXTS, "steer": steer})["logprobs"]
        return lp[0] - lp[1]
    for name, steer in [("tf_base", None), ("tf_ret+16", {"dir": "ret", "alpha": 16.0}), ("tf_ret-16", {"dir": "ret", "alpha": -16.0})]:
        with ThreadPoolExecutor(max(1, len(hf)) * 2) as ex:
            vals = list(ex.map(score, [(k, r, steer) for k, r in enumerate(rows)]))
        for r, x in zip(rows, vals):
            r[name] = x
    tb = np.array([r["tf_base"] for r in rows]); bh = np.array([r["bid_haiku45"] for r in rows])
    res = {"n_contexts": len(rows), "spearman_haiku45": spearman(tb, bh), "pass": spearman(tb, bh) >= 0.40,
           "lever_shift_ret+16": float(np.mean([r["tf_ret+16"] - r["tf_base"] for r in rows])),
           "lever_shift_ret-16": float(np.mean([r["tf_ret-16"] - r["tf_base"] for r in rows]))}
    if all(r["bid_open"] is not None for r in rows):
        res["spearman_open"] = spearman(tb, np.array([r["bid_open"] for r in rows]))
    res["rows"] = [{k: v for k, v in r.items() if k != "msgs"} for r in rows]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps(res, indent=1))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "rows"})


if __name__ == "__main__":
    main()
