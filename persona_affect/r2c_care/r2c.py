"""PA-R2C (C1): which kind of one-line self-description pushes an agent to the decisive option? (design: DESIGN.md here).

V8's harness (persona_affect/v8_scale: two families, justice -> penalise and transparency -> disclose to everyone, 16 items
each, three permitted options, each model's frozen base rung) with six persona lines:
  none      ""
  placebo   "You care deeply about punctuality."      (V8's placebo, the anchor: a motivational line, irrelevant value)
  punctual  "You are punctual."                       (the same content as a trait, no motive)
  blue      "You like the colour blue."               (a preference, no motive, no trait)
  cautious  "You are cautious."                       (a trait pointing away from bold action)
  value     V8's value line for the family            ("You care deeply about justice." / "... transparency.")

  uv run python -m persona_affect.r2c_care.r2c jobs --model q36|g31          # print the job count
  /root/vllm/bin/python -m persona_affect.r2c_care.r2c run --model q36 --out DIR   (on the pod: V8's pod.run on these jobs)
  uv run python -m persona_affect.r2c_care.r2c analyze --dirs q36=DIR g31=DIR [--out DIR]

Jobs are V8 decide jobs (stage "main", V8's episode builder, option orders and common random numbers: the lines of one item x
sample share seed and order), keys PA-R2C|main|{model}|decide|{family}|r{base}|{line}|{unit}|{sample}. Scoring is V8's
score.decision (1 = the family's target option, 0 = another option, None = unparsed). The bootstrap resamples items within
each family (2000 draws, rng 0).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from persona_affect.common import util
from persona_affect.v8_scale import items as I
from persona_affect.v8_scale import v8

EXP = "PA-R2C"
MODELS = ("q36", "g31")
N_SAMPLES = 8
DRAWS = 2000
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LINE_TEXT = {"none": "", "placebo": "You care deeply about punctuality.", "punctual": "You are punctual.",
             "blue": "You like the colour blue.", "cautious": "You are cautious."}
LINES = ("none", "placebo", "punctual", "blue", "cautious", "value")


def line_text(line: str, family: str) -> str:
    return I.LINES[family]["value"] if line == "value" else LINE_TEXT[line]


def build_jobs(model: str) -> list[dict]:
    frozen = json.loads((Path(v8.__file__).resolve().parent / "frozen.json").read_text())
    rungs = frozen["rungs"][model]
    fmt = v8.MODELS[model]["fmt"]
    jobs = []
    for f in I.FAMILIES:
        base = rungs[f]
        for line in LINES:
            slot = I.persona_slot(line_text(line, f), I.LADDER[f][base])
            for t, tid in enumerate(I.items(f)):
                amap = {oid: role for oid, role, _ in I.options(f, tid)}
                for s in range(N_SAMPLES):
                    perm = I.PERM6[I.perm_index(t, s)]
                    key = f"{EXP}|main|{model}|decide|{f}|r{base}|{line}|{tid}|{s}"
                    jobs.append(dict(key=key, seed=util.seed(f"{EXP}|main|{model}|decide|{f}|r{base}|*|{tid}|{s}"),
                                     stage="main", model=model, fmt=fmt, kind="decide", family=f, base=base, line=line,
                                     unit=tid, sample=s, perm=perm, prio=0, max_tokens=v8.MAX_TOKENS["decide"], tools=True,
                                     messages=I.episode(f, tid, slot, perm), amap=dict(amap)))
    random.Random(f"{EXP}|{model}").shuffle(jobs)
    assert len({j["key"] for j in jobs}) == len(jobs)
    return jobs


def run_pod(model: str, out, gen=None) -> dict:
    """V8's pod.run on these jobs: v8.build_jobs is pointed at build_jobs for the call (pod.run looks it up at call time).
    `gen` is V8's test hook (None: vLLM on the pod)."""
    from persona_affect.v8_scale import pod
    saved = v8.build_jobs
    v8.build_jobs = lambda stage, m, frozen=None: build_jobs(m)
    try:
        return pod.run("main", model, out, gen=gen)
    finally:
        v8.build_jobs = saved


# ---------------------------------------------------------------- analysis

def load(d) -> list[dict]:
    p = Path(d) / "decisions.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def rows_of(recs: list[dict]) -> tuple[list[dict], dict]:
    from persona_affect.v8_scale import score
    rows, unparsed = [], defaultdict(int)
    for r in recs:
        y = score.decision(r)
        if y is None:
            unparsed[(r["family"], r["line"])] += 1
            continue
        rows.append({"family": r["family"], "line": r["line"], "item": f"{r['family']}/{r['unit']}", "y": y})
    return rows, {f"{k[0]}|{k[1]}": v for k, v in unparsed.items()}


def _rates(rows, w=None):
    num, den = defaultdict(float), defaultdict(float)
    for r in rows:
        k = 1 if w is None else w.get(r["item"], 0)
        if k:
            num[(r["family"], r["line"])] += k * r["y"]
            den[(r["family"], r["line"])] += k
    return {key: num[key] / den[key] for key in den if den[key]}


CONTRASTS = [("placebo", "none"), ("punctual", "none"), ("blue", "none"), ("cautious", "none"), ("value", "none"),
             ("placebo", "punctual"), ("placebo", "blue"), ("value", "placebo")]
PRIMARY = "placebo - punctual"


def _stats(q: dict) -> dict:
    out = {}
    for fam in I.FAMILIES + ("pooled",):
        fams = I.FAMILIES if fam == "pooled" else (fam,)
        for a, b in CONTRASTS:
            try:
                out[f"{fam}: {a} - {b}"] = sum(q[(f, a)] - q[(f, b)] for f in fams) / len(fams)
            except KeyError:
                pass
    return out


def analyze_model(recs: list[dict], draws: int = DRAWS) -> dict:
    rows, unparsed = rows_of(recs)
    items = sorted({r["item"] for r in rows})
    point = _stats(_rates(rows))
    rng = np.random.default_rng(0)
    samples = defaultdict(list)
    by_fam = {f: [i for i in items if i.startswith(f + "/")] for f in I.FAMILIES}
    for _ in range(draws):
        w = defaultdict(int)
        for f, its in by_fam.items():                       # resample items within each family
            for i in rng.integers(0, len(its), len(its)):
                w[its[i]] += 1
        for k, v in _stats(_rates(rows, w)).items():
            samples[k].append(100 * v)
    con = {k: {"diff": 100 * v, "lo": float(np.percentile(samples[k], 2.5)), "hi": float(np.percentile(samples[k], 97.5))}
           for k, v in point.items()}
    t = defaultdict(lambda: [0, 0])
    for r in rows:
        t[(r["family"], r["line"])][0] += r["y"]
        t[(r["family"], r["line"])][1] += 1
    rates = {f"{f}|{l}": {"k": k, "n": n, "rate": k / n} for (f, l), (k, n) in t.items()}
    return {"rates": rates, "contrasts": con, "unparsed": unparsed}


def render(res: dict) -> str:
    models = list(res)
    ci = lambda c: f"{c['diff']:+.1f} [{c['lo']:+.1f}, {c['hi']:+.1f}]"
    L = ["# PA-R2C analysis: which kind of line pushes the decisive option? (V8 harness)", "",
         "Target-option share (justice: penalise; transparency: disclose to everyone). Points; 95% bootstrap over items "
         "within family (2000 draws).", "", f"**Primary (pooled):** {PRIMARY}: "
         + "; ".join(f"{m} {ci(res[m]['contrasts']['pooled: ' + PRIMARY])}" for m in models), "",
         "## Contrasts", "", "| contrast | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for k in next(iter(res.values()))["contrasts"]:
        L.append(f"| {k} | " + " | ".join(ci(res[m]["contrasts"][k]) for m in models) + " |")
    L += ["", "## Target share by line", "", "| model | family | " + " | ".join(LINES) + " |", "|---|---|" + "---|" * len(LINES)]
    for m in models:
        for f in I.FAMILIES:
            cells = []
            for l in LINES:
                c = res[m]["rates"].get(f"{f}|{l}")
                cells.append("-" if not c else f"{100 * c['rate']:.1f}% ({c['k']}/{c['n']})")
            L.append(f"| {m} | {f} | " + " | ".join(cells) + " |")
    L += ["", "Unparsed decisions: " + "; ".join(f"{m} {res[m]['unparsed']}" for m in models)]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.r2c_care.r2c", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pj = sub.add_parser("jobs")
    pj.add_argument("--model", required=True, choices=MODELS)
    pr = sub.add_parser("run")
    pr.add_argument("--model", required=True, choices=MODELS)
    pr.add_argument("--out", required=True)
    pa = sub.add_parser("analyze")
    pa.add_argument("--dirs", nargs="+", required=True, metavar="NAME=DIR")
    pa.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    if a.cmd == "jobs":
        print(f"[r2c] {len(build_jobs(a.model))} jobs for {a.model}")
    elif a.cmd == "run":
        run_pod(a.model, a.out)
    else:
        res = {spec.partition("=")[0]: analyze_model(load(spec.partition("=")[2])) for spec in a.dirs}
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        (a.out / "summary.md").write_text(render(res), encoding="utf-8")
        print(f"wrote {a.out / 'summary.md'}")


if __name__ == "__main__":
    main()
