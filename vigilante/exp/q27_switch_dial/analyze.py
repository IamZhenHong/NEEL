"""Queue item 27: is persona adoption a switch or a dial? No new generation; reads persona/ results (read-only).

Model: Qwen3-14B fine-tuned toward the contemptuous persona C (persona Study 1). Every checkpoint has (a) judged
contempt rate on 120 replies (Sonnet) and (b) final-prompt-token activations on 100 neutral prompts (per-layer means,
per-prompt vectors at layers 10/20/30).
Internal position = projection of (checkpoint mean - its run's starting mean) on the unit C direction, where C =
mean(untouched -> C at 100%) - mean(untouched) from the OTHER pilot seed (no shared training noise), divided by the
norm of that C shift: 1.0 = as far along C as the other seed's full C training went. Floor: 1,000 random unit
directions (95th percentile of |projection| / norm).

  uv run python -m vigilante.exp.q27_switch_dial.analyze
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3] / "persona" / "results"
OUT = Path(__file__).parent / "results"
LAYER = 20          # primary (middle of the 40-layer model); 10 and 30 reported too


def mean(run, tag):
    return np.load(R / run / "acts" / f"{tag}.npz")["mean"].astype(np.float64)


def contempt(run_judged):
    c = defaultdict(lambda: [0, 0])
    for l in open(run_judged):
        r = json.loads(l)
        v = r.get("contempt")
        if v in (None, ""):
            continue
        c[r["ckpt"]][0] += int(v)
        c[r["ckpt"]][1] += 1
    return {k: a / b for k, (a, b) in c.items() if b}


def main():
    OUT.mkdir(exist_ok=True)
    cdir = {s: mean(f"pilot_v1_s{s}", f"s{s}_none_C_100") - mean(f"pilot_v1_s{s}", f"s{s}_none_base") for s in (1, 2)}
    rng = np.random.default_rng(0)
    R_ = rng.standard_normal((1000, cdir[1].shape[1]))
    R_ /= np.linalg.norm(R_, axis=1, keepdims=True)

    beh = {**contempt(R / "pilot_v1" / "judged.jsonl")}
    for run in ("titrate_v1", "titrate_v2", "titrate_v3", "dose_v1"):
        beh.update({f"{run}:{k}": v for k, v in contempt(R / run / "judged.jsonl").items()})

    # series: (label, run dir, tag prefix in acts, start tag, [(x, tag)], seed)
    series = []
    for s in (1, 2):
        for p, start in (("none", "none_base"), ("N", "N_stage1"), ("B", "B_stage1"), ("Bp", "Bp_stage1")):
            pts = [(0, f"s{s}_{start}")] + [(f, f"s{s}_{p}_C_{f}") for f in (25, 50, 75, 100)]
            series.append((f"pilot s{s} {p}->C (% of 825)", f"pilot_v1_s{s}", pts, s, "pilot_v1"))
    series.append(("titrate_v1 N->C (examples)", "titrate_v1",
                   [(0, "s1_N_stage1")] + [(n, f"s1_N_C_n{n}_100") for n in (125, 250, 500, 1000)], 1, "titrate_v1"))
    series.append(("titrate_v2 N->C (examples)", "titrate_v2",
                   [(0, "s1_N_stage1")] + [(n, f"s1_N_C_n{n}_100") for n in (300, 350, 400)], 1, "titrate_v2"))
    series.append(("titrate_v3 N->C, lr 1e-5, n 1100 (% of run)", "titrate_v3",
                   [(0, "s1_N_stage1")] + [(f, f"s1_N_C_n1100_lr1e-05_{f}") for f in (25, 50, 75, 100)], 1, "titrate_v3"))
    for p, start in (("none", "none_base"), ("N", "N_stage1"), ("Bp", "Bp_stage1")):
        series.append((f"dose_v1 {p}->C (examples)", "dose_v1",
                       [(0, f"s1_{start}")] + [(n, f"s1_{p}_C_n{n}_100") for n in (300, 550, 1100)], 1, "dose_v1"))

    res, lines = [], ["# Item 27: switch or dial? (Qwen3-14B, persona Study 1 checkpoints)", "",
                      "`internal` = position along the other seed's contempt direction at layer 20 (1.0 = that seed's "
                      "full untouched->C shift); `floor` = 95th percentile of the same quantity over 1,000 random "
                      "directions. `contempt` = judged rate on 120 replies.", ""]
    for label, run, pts, s, jrun in series:
        other = 2 if s == 1 else 1
        d = cdir[other][LAYER]
        nd = np.linalg.norm(d)
        u = d / nd
        start = mean(run, pts[0][1])[LAYER]
        rows = []
        for x, tag in pts:
            sh = mean(run, tag)[LAYER] - start
            key = tag if jrun == "pilot_v1" else f"{jrun}:{tag}"
            rows.append({"x": x, "tag": tag, "internal": float(sh @ u / nd),
                         "floor95": float(np.percentile(np.abs(R_ @ sh) / nd, 95)),
                         "contempt": beh.get(key)})
        res.append({"series": label, "rows": rows})
        lines += [f"### {label}", "", "| x | contempt | internal | random floor |", "|---|---|---|---|"]
        for r in rows:
            c = "-" if r["contempt"] is None else f"{r['contempt']:.2f}"
            lines.append(f"| {r['x']} | {c} | {r['internal']:.2f} | {r['floor95']:.2f} |")
        lines.append("")
    # switch vs dial summary: for each series, internal fraction (of the series' last point) at the last point where
    # contempt <= 0.10, and at the first point where contempt >= 0.5
    lines += ["## Summary: how far the internal state has moved before behaviour switches", "",
              "| series | last checkpoint with contempt ≤ 10%: internal (share of series max) | first checkpoint ≥ 50%: internal share |",
              "|---|---|---|"]
    for r in res:
        rows = r["rows"]
        mx = max(abs(x["internal"]) for x in rows) or 1
        low = [x for x in rows[1:] if x["contempt"] is not None and x["contempt"] <= 0.10]
        high = [x for x in rows[1:] if x["contempt"] is not None and x["contempt"] >= 0.50]
        a = f"{low[-1]['x']}: {low[-1]['internal'] / mx:.2f} (contempt {low[-1]['contempt']:.2f})" if low else "-"
        b = f"{high[0]['x']}: {high[0]['internal'] / mx:.2f} (contempt {high[0]['contempt']:.2f})" if high else "-"
        lines.append(f"| {r['series']} | {a} | {b} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "numbers.json").write_text(json.dumps(res, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
