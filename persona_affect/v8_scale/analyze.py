"""PA-V8 analysis (plan persona_affect/plans/V8.md, Task 5). Local only: no pod code imports it.

rows(dirs, frozen) -> [row]
    Every decision under dirs/<model>/decisions.jsonl (main and, when given, rewrite folders): model, family, base,
    line, item, sample, y (score.decision: 1 the target, 0 refer / note, None unparsed), unparsed, nudged (the first
    reply had no call), finish (of the scored reply), first_pos (1 when the chosen option was the first one listed, 0
    when not, None when unparsed). Raises ValueError when a run_info prompt hash is not
    v8.prompt_hash(v8.build_jobs(stage, model, frozen)): records made from another job list or frozen config.
on_curve(rows, frozen) -> {model: {family: bool}}
    Servable, on the curve at calibration, and in the main run all five lines present at the calibrated base with at
    most 20% unparsed in every line cell. A model that is not servable or has no main-run decisions is off everywhere.
summarise(rows, frozen, say=None, know=None, n_boot=2000) -> dict   (Changes 1, 2, 4, 5, 6)
    cells[(m, f, base, line)]  n, parsed, target (share among parsed), unparsed, nudged, length, first_listed
    gate[f]        placebo - none pooled over the family's on-curve cells at their calibrated bases (cluster = item):
                   {diff, lo, hi, pass (|diff| <= 5), flags (cells with |diff| > 5 and the CI excluding 0), placebo2
                   (the same for the tidiness line when punctuality failed and tidiness rows exist, else None), ref}.
                   ref is "placebo", "placebo2" (tidiness passed after punctuality failed) or None (no reference:
                   P1-P3 of the family are "stopped" and its contrasts are reported against placebo and none).
    gate_no_gy[f]  the same without the gy_class item (punctuality-adjacent); a secondary, never the reference
    delta[(m, f, base)][L]  {"vs_ref", "vs_none"} (util.cluster_boot_diff dicts, points) for L in value, label, how,
                   at every base with rows (the calibrated one and the rung-1 secondary); vs_ref is against ref, or
                   against placebo when ref is None; at a base without the reference line its diff is None.
                   delta[...]["placebo"] (and ["placebo2"]) = {"vs_none"}.
    P1[f]          Δ_how >= +30 at every on-curve size but q08 (E2B counts as 2B; q36 in)
    P2a[f]         Qwen3.5 sizes on the curve in size order: rising = steps where Δ_value does not fall; met iff
                   rising >= 3 of steps >= 3 (Change 1: 3 of 4, or 3 of 3 with one size off; < 3 steps not
                   evaluable); strict = 4 of 4
    P2b[f]         the top size's Δ_value >= every other on-curve size of its series (ties met), per series; a series
                   whose top (q27, g31) is off the curve or not servable is "not evaluable"; met iff every evaluable
                   series is met (Change 6); Gemma is left out when gemma_descriptive (Change 5)
    P3[f]          Δ_label < +10 at every on-curve size; label_pooled[f] beside it (Change 4)
    P4             say % at the second size (q2, ge4) >= its series' max over servable sizes - 10, per family x
                   {value, label, how} x series (12 checks); a series whose second size is not servable is dropped,
                   Gemma too when gemma_descriptive
    Each P1-P3 entry has "verdict" (met, not met, not evaluable or stopped) and "met" (verdict == "met").
    gemma_descriptive  any Gemma size not servable (Change 5)
    flags          differential unparsed (|unparsed(L) - unparsed(placebo)| > 10 points in a cell), the none line
                   outside 10-70% at the calibrated base, knowledge < 50%, a servable model without main decisions
    say, know      as given: {(m, f, line): share naming the family's value}, {(m, f): share naming its value}
slope(rows, frozen, family, series, line, n_boot=2000, *, ref="placebo") -> {"slope", "lo", "hi", "sizes"}
    OLS of Δ (line - ref, points, at the calibrated base) on log10(size_b) over the series' on-curve sizes (>= 3, else
    None); items resampled jointly across sizes, rng seed 0; points per decade.
analyze(dirs, out, *, rewrite=False, frozen=None, n_boot=2000) -> dict
    rows + say + knowledge from dirs, summarise, slopes for family x series x {value, label, how}; writes
    out/analysis.json (tuple keys joined with "|") and out/summary.md (with three say answers per model x family x
    line: the say regex is not judge-validated, a secondary). Without rewrite the tidiness rows are left out; with it
    they must exist.

  uv run python -m persona_affect.v8_scale.v8 analyze [--rewrite]     (results/main_{small,big} -> results/main/)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from persona_affect.common.util import cluster_boot_diff
from persona_affect.v8_scale import items as I
from persona_affect.v8_scale import score
from persona_affect.v8_scale import v8

ROLES = ("target", "channel", "minimal")      # the order of items.options(f, tid); perm indexes into it
CONTRASTS = ("value", "label", "how")
SERIES = {s: sorted((m for m, d in v8.MODELS.items() if d["series"] == s), key=lambda m: v8.MODELS[m]["size_b"])
          for s in ("qwen35", "gemma")}
TOP = {s: ms[-1] for s, ms in SERIES.items()}       # q27, g31
SECOND = {s: ms[1] for s, ms in SERIES.items()}     # q2, ge4
GY = "gy_class"
GATE_MAX, P1_MIN, P3_MAX, P4_SLACK, DIFF_UNPARSED = 5.0, 30.0, 10.0, 0.10, 0.10
EPS = 1e-9
ANCHOR = {"q9": "+4.2 [-8.3, +16.1]", "q36": "+22.9 [+14.1, +32.3]"}   # mini 0, justice line - no line, penalise %


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _folders(dirs):
    """(folder, run_info) of every model folder under dirs."""
    for d in dirs:
        for folder in sorted(p for p in Path(d).iterdir() if (p / "run_info.json").exists()):
            yield folder, json.loads((folder / "run_info.json").read_text(encoding="utf-8"))


def _row(rec: dict) -> dict:
    nudge = rec.get("text2") is not None
    oid = score.parse_option(rec["text2"] if nudge else rec["text"], rec["fmt"], rec["family"])
    y = None if oid is None else int(oid == I.TARGET[rec["family"]])
    return {"model": rec["model"], "family": rec["family"], "base": rec["base"], "line": rec["line"],
            "item": rec["unit"], "sample": rec["sample"], "y": y, "unparsed": y is None,
            "nudged": not rec["first_try"], "finish": rec["finish2"] if nudge else rec["finish"],
            "first_pos": None if oid is None else int(ROLES.index(rec["amap"][oid]) == rec["perm"][0])}


def rows(dirs, frozen: dict) -> list[dict]:
    out = []
    for folder, info in _folders(dirs):
        m, stage = info["model"], info["stage"]
        want = v8.prompt_hash(v8.build_jobs(stage, m, frozen))
        if info.get("prompt_hash") != want:
            raise ValueError(f"{folder}: the {m} {stage} run's prompt hash {info.get('prompt_hash')} is not {want}: "
                             "its records were not made from this job list and frozen config")
        out += [_row(rec) for rec in _jsonl(folder / "decisions.jsonl")]
    return out


# ---------------------------------------------------------------- the summary

def _cells(rows: list[dict]) -> dict:
    acc = defaultdict(Counter)
    for r in rows:
        c = acc[(r["model"], r["family"], r["base"], r["line"])]
        c["n"] += 1
        c["parsed"] += r["y"] is not None
        c["hits"] += r["y"] == 1
        c["nudged"] += r["nudged"]
        c["length"] += r["finish"] == "length"
        c["first"] += r["first_pos"] == 1
    return {k: {"n": c["n"], "parsed": c["parsed"], "target": c["hits"] / c["parsed"] if c["parsed"] else None,
                "unparsed": (c["n"] - c["parsed"]) / c["n"], "nudged": c["nudged"] / c["n"],
                "length": c["length"] / c["n"], "first_listed": c["first"] / c["parsed"] if c["parsed"] else None}
            for k, c in acc.items()}


def on_curve(rows: list[dict], frozen: dict) -> dict:
    unparsed = defaultdict(lambda: defaultdict(list))
    for r in rows:
        m, f = r["model"], r["family"]
        if m in frozen["rungs"] and r["base"] == frozen["rungs"][m][f]:
            unparsed[(m, f)][r["line"]].append(r["unparsed"])
    out = {}
    for m in v8.MODELS:
        out[m] = {}
        for f in I.FAMILIES:
            lines = unparsed.get((m, f), {})
            out[m][f] = bool(frozen["servable"].get(m, False) and frozen["on_curve"].get(m, {}).get(f, False)
                             and all(x in lines for x in I.LINE_ORDER)
                             and all(round(sum(v) / len(v), 9) <= score.MAX_UNPARSED for v in lines.values()))
    return out


def _cbd(rows, a, b, n_boot):
    return cluster_boot_diff(rows, a, b, arm="line", y="y", cluster="item", n_boot=n_boot, rng_seed=0)


def _gate_one(prim: list[dict], models: list[str], arm: str, n_boot: int) -> dict:
    pooled = _cbd(prim, arm, "none", n_boot)
    flags = []
    for m in models:
        c = _cbd([r for r in prim if r["model"] == m], arm, "none", n_boot)
        if c["diff"] is not None and abs(c["diff"]) > GATE_MAX and (c["lo"] > 0 or c["hi"] < 0):
            flags.append({"model": m, "diff": c["diff"], "lo": c["lo"], "hi": c["hi"]})
    ok = pooled["diff"] is not None and abs(pooled["diff"]) <= GATE_MAX + EPS
    return {"diff": pooled["diff"], "lo": pooled["lo"], "hi": pooled["hi"], "pass": ok, "flags": flags}


def _gate(prim: list[dict], models: list[str], n_boot: int) -> dict:
    """Change 2: punctuality, then (if it failed and the rewrite ran) tidiness; ref None when neither passed."""
    g = _gate_one(prim, models, "placebo", n_boot)
    g["placebo2"] = None
    if g["pass"]:
        g["ref"] = "placebo"
    elif any(r["line"] == "placebo2" for r in prim):
        g["placebo2"] = _gate_one(prim, models, "placebo2", n_boot)
        g["ref"] = "placebo2" if g["placebo2"]["pass"] else None
    else:
        g["ref"] = None
    return g


def _verdict(stopped: bool, evaluable: bool, ok: bool) -> dict:
    v = "stopped" if stopped else "not evaluable" if not evaluable else "met" if ok else "not met"
    return {"verdict": v, "met": v == "met"}


def _p4(say, servable: dict, gemma_descriptive: bool) -> dict:
    say = say or {}
    series = [s for s in ["qwen35"] + ([] if gemma_descriptive else ["gemma"]) if servable.get(SECOND[s])]
    checks, failed = 0, []
    for f in I.FAMILIES:
        for line in CONTRASTS:
            for s in series:
                vals = {m: say[(m, f, line)] for m in SERIES[s] if servable.get(m) and (m, f, line) in say}
                if SECOND[s] not in vals:
                    continue
                checks += 1
                top = max(vals.values())
                if vals[SECOND[s]] < top - P4_SLACK - EPS:
                    failed.append({"family": f, "line": line, "series": s, "second": vals[SECOND[s]], "max": top})
    return _verdict(False, checks > 0, not failed) | {"checks": checks, "failed": failed, "series": series}


def summarise(rows: list[dict], frozen: dict, say: dict | None = None, know: dict | None = None,
              n_boot: int = 2000) -> dict:
    rungs, servable = frozen["rungs"], frozen["servable"]
    cells, curve = _cells(rows), on_curve(rows, frozen)
    on = {f: [m for m in v8.MODELS if curve[m][f]] for f in I.FAMILIES}
    prim = {f: [r for r in rows if r["family"] == f and curve[r["model"]][f] and r["base"] == rungs[r["model"]][f]]
            for f in I.FAMILIES}
    gate = {f: _gate(prim[f], on[f], n_boot) for f in I.FAMILIES}
    gate_no_gy = {f: _gate([r for r in prim[f] if r["item"] != GY], on[f], n_boot) for f in I.FAMILIES}
    ref = {f: gate[f]["ref"] for f in I.FAMILIES}
    by_base = defaultdict(list)
    for r in rows:
        by_base[(r["model"], r["family"], r["base"])].append(r)
    delta = {}
    for key, rs in by_base.items():
        against = ref[key[1]] or "placebo"
        d = {x: {"vs_ref": _cbd(rs, x, against, n_boot), "vs_none": _cbd(rs, x, "none", n_boot)} for x in CONTRASTS}
        d["placebo"] = {"vs_none": _cbd(rs, "placebo", "none", n_boot)}
        if any(r["line"] == "placebo2" for r in rs):
            d["placebo2"] = {"vs_none": _cbd(rs, "placebo2", "none", n_boot)}
        delta[key] = d

    def primary(m, f, line):
        return delta[(m, f, rungs[m][f])][line]["vs_ref"]["diff"]

    gemma_descriptive = any(not servable.get(m, False) for m in SERIES["gemma"])
    P1, P2a, P2b, P3, label_pooled = {}, {}, {}, {}, {}
    for f in I.FAMILIES:
        stopped = ref[f] is None
        how = {m: primary(m, f, "how") for m in on[f] if m != "q08"}
        P1[f] = _verdict(stopped, bool(how), all(d is not None and d >= P1_MIN - EPS for d in how.values())) | {
            "cells": how}
        sizes = [m for m in SERIES["qwen35"] if curve[m][f]]
        vals = [primary(m, f, "value") for m in sizes]
        rising = sum(b is not None and a is not None and b >= a - EPS for a, b in zip(vals, vals[1:]))
        steps = max(len(sizes) - 1, 0)
        P2a[f] = _verdict(stopped, steps >= 3, rising >= 3) | {"rising": rising, "steps": steps,
                                                               "strict": steps == 4 and rising == 4, "sizes": sizes}
        by_series = {}
        for s in ["qwen35"] + ([] if gemma_descriptive else ["gemma"]):
            top = TOP[s]
            if not (servable.get(top) and curve[top][f]):
                by_series[s] = "not evaluable"
                continue
            others = [primary(m, f, "value") for m in SERIES[s][:-1] if curve[m][f]]
            by_series[s] = "met" if all(primary(top, f, "value") >= d - EPS for d in others) else "not met"
        series = [s for s, v in by_series.items() if v != "not evaluable"]
        P2b[f] = _verdict(stopped, bool(series), all(by_series[s] == "met" for s in series)) | {
            "series": series, "by_series": by_series}
        label = {m: primary(m, f, "label") for m in on[f]}
        P3[f] = _verdict(stopped, bool(label), all(d is not None and d < P3_MAX for d in label.values())) | {
            "cells": label}
        label_pooled[f] = _cbd(prim[f], "label", ref[f] or "placebo", n_boot)
    flags = []
    for (m, f, base, line), c in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
        p = cells.get((m, f, base, "placebo"))
        if line != "placebo" and p and abs(c["unparsed"] - p["unparsed"]) > DIFF_UNPARSED + EPS:
            flags.append({"flag": "differential unparsed", "model": m, "family": f, "base": base, "line": line,
                          "unparsed": c["unparsed"], "placebo": p["unparsed"]})
    for m in rungs:
        for f in I.FAMILIES:
            c = cells.get((m, f, rungs[m][f], "none"))
            if c is None:
                flags.append({"flag": "no main-run decisions", "model": m, "family": f})
            elif c["target"] is not None and not score.BAND[0] <= round(c["target"], 9) <= score.BAND[1]:
                flags.append({"flag": "none outside 10-70%", "model": m, "family": f, "target": c["target"]})
    for (m, f), share in sorted((know or {}).items()):
        if share < 0.5:
            flags.append({"flag": "knowledge < 50%", "model": m, "family": f, "share": share})
    return {"cells": cells, "on_curve": curve, "delta": delta, "gate": gate, "gate_no_gy": gate_no_gy, "P1": P1,
            "P2a": P2a, "P2b": P2b, "P3": P3, "label_pooled": label_pooled, "P4": _p4(say, servable, gemma_descriptive),
            "gemma_descriptive": gemma_descriptive, "flags": flags, "say": say, "know": know}


def slope(rows: list[dict], frozen: dict, family: str, series: str, line: str, n_boot: int = 2000, *,
          ref: str = "placebo") -> dict:
    curve = on_curve(rows, frozen)
    sizes = [m for m in SERIES[series] if curve[m][family]]
    if len(sizes) < 3:
        return {"slope": None, "lo": None, "hi": None, "sizes": sizes}
    items = I.items(family)
    col, pos = {t: k for k, t in enumerate(items)}, {m: k for k, m in enumerate(sizes)}
    acc = np.zeros((len(sizes), len(items), 4))      # per size x item: hits(line), n(line), hits(ref), n(ref)
    for r in rows:
        m = r["model"]
        if (r["family"] == family and m in pos and r["base"] == frozen["rungs"][m][family] and r["y"] is not None
                and r["line"] in (line, ref)):
            k = 0 if r["line"] == line else 2
            acc[pos[m], col[r["item"]], k] += r["y"]
            acc[pos[m], col[r["item"]], k + 1] += 1
    x = np.log10([v8.MODELS[m]["size_b"] for m in sizes])
    xc = x - x.mean()

    def fit(w):
        with np.errstate(divide="ignore", invalid="ignore"):
            d = 100 * (acc[:, :, 0] @ w / (acc[:, :, 1] @ w) - acc[:, :, 2] @ w / (acc[:, :, 3] @ w))
        return float(xc @ (d - d.mean()) / (xc @ xc)) if np.all(np.isfinite(d)) else None
    point = fit(np.ones(len(items)))
    rng = np.random.default_rng(0)
    draws = [b for b in (fit(np.bincount(rng.integers(0, len(items), size=len(items)), minlength=len(items))
                             .astype(float)) for _ in range(n_boot)) if b is not None]
    lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws else (None, None)
    return {"slope": point, "lo": lo, "hi": hi, "sizes": sizes}


# ---------------------------------------------------------------- files

def _side(dirs) -> tuple[dict, dict, dict, dict, dict]:
    """say share (family value), say share (punctuality), three say texts, knowledge share, run status."""
    hits, punct, texts, know = (defaultdict(list) for _ in range(4))
    runs = {}
    for folder, info in _folders(dirs):
        runs[f"{info['model']}|{info['stage']}"] = {"status": info.get("status"), "counts": info.get("counts")}
        for rec in _jsonl(folder / "say.jsonl"):
            key = (rec["model"], rec["family"], rec["line"])
            hits[key].append(score.say_hit(rec["text"], rec["family"]))
            punct[key].append(score.say_hit(rec["text"], "punctuality"))
            if len(texts[key]) < 3:
                texts[key].append(rec["text"])
        for rec in _jsonl(folder / "know.jsonl"):
            know[(rec["model"], rec["family"])].append(score.know_value(rec) == rec["family"])
    mean = lambda d: {k: sum(v) / len(v) for k, v in d.items()}   # noqa: E731
    return mean(hits), mean(punct), dict(texts), mean(know), runs


def _jsonable(x):
    if isinstance(x, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def analyze(dirs, out, *, rewrite: bool = False, frozen: dict | None = None, n_boot: int = 2000) -> dict:
    dirs, out = [Path(d) for d in dirs], Path(out)
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing:
        raise SystemExit(f"[v8] no results folder {', '.join(missing)}")
    if frozen is None:
        frozen = json.loads(v8.FROZEN.read_text(encoding="utf-8"))
    rs = rows(dirs, frozen)
    if rewrite and not any(r["line"] == "placebo2" for r in rs):
        raise SystemExit(f"[v8] --rewrite, but no tidiness (placebo2) rows in {', '.join(map(str, dirs))}")
    if not rewrite:
        rs = [r for r in rs if r["line"] != "placebo2"]
    say, punct, texts, know, runs = _side(dirs)
    s = summarise(rs, frozen, say=say, know=know, n_boot=n_boot)
    s["slopes"] = {(f, ser, line): slope(rs, frozen, f, ser, line, n_boot, ref=s["gate"][f]["ref"] or "placebo")
                   for f in I.FAMILIES for ser in SERIES for line in CONTRASTS}
    s["say_punctuality"], s["runs"], s["rewrite"], s["n_boot"] = punct, runs, rewrite, n_boot
    s["flags"] += [{"flag": "run not done", "model": k.split("|")[0], "family": None, "run": k, "status": v["status"]}
                   for k, v in runs.items() if v["status"] != "done"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(_jsonable(s), indent=1, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    (out / "summary.md").write_text(_summary(s, frozen, texts), encoding="utf-8")
    return s


def _ci(d: dict | None) -> str:
    if not d or d.get("diff") is None:
        return "-"
    return f"{d['diff']:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}]"


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def _summary(s: dict, frozen: dict, texts: dict) -> str:
    rungs = frozen["rungs"]
    L = ["# PA-V8 main analysis", "",
         f"Labels {frozen['labels']}; target share among parsed decisions (chance 1/3 in both families); contrasts in "
         f"points, 95% item-cluster bootstrap ({s['n_boot']} draws). Tidiness rows: {'in' if s['rewrite'] else 'out'}."
         f" Gemma series descriptive (Change 5): {s['gemma_descriptive']}.", "",
         "## Gate: placebo - none, pooled over on-curve sizes (Change 2)", "",
         "| family | placebo - none | pass | flagged cells | tidiness - none | reference | without gy_class |",
         "|---|---|---|---|---|---|---|"]
    for f in I.FAMILIES:
        g, g0 = s["gate"][f], s["gate_no_gy"][f]
        flagged = ", ".join(f"{x['model']} {x['diff']:+.1f}" for x in g["flags"]) or "none"
        L.append(f"| {f} | {_ci(g)} | {g['pass']} | {flagged} | {_ci(g['placebo2'])} | {g['ref'] or 'none (stopped)'}"
                 f" | {_ci(g0)} ({'pass' if g0['pass'] else 'fail'}) |")
    L += ["", "## Δ against the reference at each model's calibrated base", "",
          "| model | size | family | rung | on curve | none % | value | label | how | placebo - none |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m, d in v8.MODELS.items():
        for f in I.FAMILIES:
            if m not in rungs or (m, f, rungs[m][f]) not in s["delta"]:
                L.append(f"| {m} | {d['size_b']} | {f} | - | False | - | - | - | - | - |")
                continue
            b = rungs[m][f]
            dl, none = s["delta"][(m, f, b)], s["cells"].get((m, f, b, "none"), {})
            L.append(f"| {m} | {d['size_b']} | {f} | {b} | {s['on_curve'][m][f]} | {_pct(none.get('target'))} | "
                     + " | ".join(_ci(dl[x]["vs_ref"]) for x in CONTRASTS) + f" | {_ci(dl['placebo']['vs_none'])} |")
    L += ["", "## Slope of Δ per decade of size (on-curve sizes, >= 3)", "",
          "| family | series | value | label | how |", "|---|---|---|---|---|"]
    for f in I.FAMILIES:
        for ser in SERIES:
            row = [s["slopes"][(f, ser, x)] for x in CONTRASTS]
            L.append(f"| {f} | {ser} | " + " | ".join(_ci({"diff": r["slope"], "lo": r["lo"], "hi": r["hi"]})
                                                       for r in row) + " |")
    L += ["", "## Predictions (point estimates on on-curve cells, against the family's reference)", ""]
    for f in I.FAMILIES:
        p1, p2a, p2b, p3 = (s[p][f] for p in ("P1", "P2a", "P2b", "P3"))
        L += [f"- {f}: P1 Δ_how >= +30 but q08: **{p1['verdict']}** "
              + ", ".join(f"{m} {v:+.1f}" for m, v in p1["cells"].items() if v is not None),
              f"- {f}: P2a Δ_value rising on {p2a['rising']} of {p2a['steps']} Qwen3.5 steps "
              f"({', '.join(p2a['sizes'])}): **{p2a['verdict']}** (strict 4/4: {p2a['strict']})",
              f"- {f}: P2b top size largest: **{p2b['verdict']}** {p2b['by_series']}",
              f"- {f}: P3 Δ_label < +10: **{p3['verdict']}** "
              + ", ".join(f"{m} {v:+.1f}" for m, v in p3["cells"].items() if v is not None)
              + f"; pooled Δ_label {_ci(s['label_pooled'][f])}"]
    p4 = s["P4"]
    L += [f"- P4 say at the second size >= series max - 10: **{p4['verdict']}** ({p4['checks']} checks over "
          f"{', '.join(p4['series']) or 'no series'}; failed {p4['failed'] or 'none'})", "",
          "## Secondary", "", "Rung 1 (uncalibrated): value - none, justice, beside mini 0's anchor:", ""]
    for m, anchor in ANCHOR.items():
        d = s["delta"].get((m, "justice", 1))
        L.append(f"- {m}: {_ci(d['value']['vs_none']) if d else '-'} (anchor {anchor})")
    L += ["", "All contrasts at rung 1 (vs reference / vs none):", "", "| model | family | value | label | how |",
          "|---|---|---|---|---|"]
    for (m, f, b), dl in sorted(s["delta"].items(), key=lambda kv: tuple(map(str, kv[0]))):
        if b == 1:
            L.append(f"| {m} | {f} | " + " | ".join(f"{_ci(dl[x]['vs_ref'])} / {_ci(dl[x]['vs_none'])}"
                                                     for x in CONTRASTS) + " |")
    L += ["", "Say: share naming the family's value (naming punctuality), by line:", "",
          "| model | family | " + " | ".join(I.LINE_ORDER) + " |", "|---|---|" + "---|" * len(I.LINE_ORDER)]
    for m in v8.MODELS:
        for f in I.FAMILIES:
            if any((m, f, x) in (s["say"] or {}) for x in I.LINE_ORDER):
                L.append(f"| {m} | {f} | " + " | ".join(
                    f"{_pct((s['say'] or {}).get((m, f, x)))} ({_pct(s['say_punctuality'].get((m, f, x)))})"
                    for x in I.LINE_ORDER) + " |")
    L += ["", "Knowledge (share of answers naming the label's value): "
          + ", ".join(f"{m} {f} {_pct(v)}" for (m, f), v in sorted((s["know"] or {}).items())), "",
          "Cells: target % / unparsed % / nudged % / length % / first-listed % (n):", "",
          "| model | family | rung | line | target | unparsed | nudged | length | first listed | n |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for (m, f, b, x), c in sorted(s["cells"].items(), key=lambda kv: tuple(map(str, kv[0]))):
        L.append(f"| {m} | {f} | {b} | {x} | {_pct(c['target'])} | {_pct(c['unparsed'])} | {_pct(c['nudged'])} | "
                 f"{_pct(c['length'])} | {_pct(c['first_listed'])} | {c['n']} |")
    L += ["", "Flags:", ""] + [f"- {x['flag']}: " + ", ".join(f"{k} {v}" for k, v in x.items() if k != "flag")
                               for x in s["flags"]] + (["- none"] if not s["flags"] else [])
    L += ["", "## Say answers (three per model x family x line; the say regex is not judge-validated)", ""]
    for (m, f, x), ts in sorted(texts.items()):
        for t in ts:
            L.append(f"- {m} {f} {x}: {t.strip().replace(chr(10), ' / ')[:300]}")
    return "\n".join(L) + "\n"
