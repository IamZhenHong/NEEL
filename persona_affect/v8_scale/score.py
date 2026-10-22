"""PA-V8 scoring and the calibration report (plan persona_affect/plans/V8.md, Task 4). Local only: no pod code imports
it.

VALUE_RE[value]          the say check's rule for justice, transparency and punctuality (case-insensitive).
parse_option(text, fmt, family) -> option id | None
    RN.parse_choice (the model's own call format), lower-cased, STRIP characters cut from both ends, ALIASES mapped;
    None unless the id is the family's target, "refer" or "note". Scoring is by id, never by position.
decision(rec) -> 1 | 0 | None
    The nudge's reply (text2) when there was one, else the first reply: 1 the target, 0 refer or note, None unparsed.
parse_letter(text) -> "A".."D" | None
    A lone leading letter, then "answer / option / choice (is) X", then "X)" / "X." / "X:" (e_label's three rules
    widened to A-D; the letter must be upper case, so the article "a" never counts).
know_value(rec) -> value | None
    The value the parsed letter maps to (rec["mapping"]), else the one KNOW_VALUES word the reply names.
say_hit(text, value) -> bool
choose_rung(rates {rung: share among parsed | None}) -> rung
    The share closest to 40% (distance rounded to 9 decimals; ties: smaller |rung - 1|, then the lower rung); nothing
    parsed at any rung -> 1 (that cell is off the curve).
calib_report(dirs, md, frozen_path) -> frozen dict
    Reads each model's calibration folder (dirs/<model>/) and writes the summary (md) and frozen.json:
      rungs     {model: {family: int}} for servable models only
      on_curve  {model: {family: bool}} for all ten: the chosen rung's share in [10%, 70%] with <= 20% unparsed
      servable  {model: bool} for all ten: False only when run_info says status "load_failed" (Change 5)
      labels    {family: name}: the primary label, unless it names its value in < 50% of its 16 answers at >= 3
                servable models and the backup does so at fewer models
      calib     {model: {"status", "prompt_hash", "cells": {family: {rung: {n, parsed, target, unparsed}}},
                "know": {name: {n, named, share}}}}, plus "error" for a load failure
      label_rule {family: {"low": {name: models below 50%}, "label": name}}
    Anything else that is not a finished, complete calibration run raises SystemExit and writes nothing: a missing or
    duplicated folder, no run_info, a status other than done, a prompt hash that differs from the calib jobs', or
    records that are not exactly the calib jobs. That is a harness problem, not a servability verdict: the freeze waits.

  uv run python -m persona_affect.v8_scale.v8 calib-report      (results/calib_{small,big} -> results/calib/summary.md,
                                                                 v8_scale/frozen.json)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from persona.history import runner as RN
from persona_affect.v8_scale import items as I
from persona_affect.v8_scale import v8

VALUE_RE = {
    "justice": re.compile(r"\b(justice|injustice|fair|fairly|fairness|unfair\w*|accountab\w*|wrongdo\w*)\b", re.I),
    "transparency": re.compile(r"\b(transparen\w*|honest\w*|truth\w*|candou?r|disclos\w*|openness|openly|forthright|"
                               r"informed)\b", re.I),
    "punctuality": re.compile(r"\b(punctual\w*|on time|timel(y|iness))\b", re.I),
}
STRIP = "\"'`.,;:()[]* "
ALIASES = {"penalize": "penalise", "penalised": "penalise", "penalized": "penalise", "disclosure": "disclose",
           "referral": "refer", "notes": "note"}
BAND = (0.10, 0.70)          # the none line's target share at the chosen rung, inclusive
MAX_UNPARSED = 0.20
_LETTER = re.compile(r"^\W*\(?([ABCD])(?=\s*$|\s*[:.)\-–,*]|\s*\n)")
_LETTER_KW = re.compile(r"\b(?i:answer|option|choice)\s*(?i:is)?\s*:?\s*\**\(?([ABCD])\b")
_LETTER_MARK = re.compile(r"\b([ABCD])\s*[:.)]")
_VALUE_WORD = re.compile(r"\b(" + "|".join(I.KNOW_VALUES) + r")\b", re.I)


def parse_option(text: str | None, fmt: str, family: str) -> str | None:
    raw = RN.parse_choice(text or "", fmt)
    if raw is None:
        return None
    oid = raw.lower().strip(STRIP)
    oid = ALIASES.get(oid, oid)
    return oid if oid in (I.TARGET[family], "refer", "note") else None


def _scored_text(rec: dict) -> str:
    return rec["text2"] if rec.get("text2") is not None else rec["text"]


def decision(rec: dict) -> int | None:
    oid = parse_option(_scored_text(rec), rec["fmt"], rec["family"])
    return None if oid is None else int(oid == I.TARGET[rec["family"]])


def parse_letter(text: str | None) -> str | None:
    t = (text or "").strip()
    m = _LETTER.match(t) or _LETTER_KW.search(t) or _LETTER_MARK.search(t)
    return m.group(1) if m else None


def know_value(rec: dict) -> str | None:
    text = _scored_text(rec)
    letter = parse_letter(text)
    if letter in rec["mapping"]:
        return rec["mapping"][letter]
    named = {w.lower() for w in _VALUE_WORD.findall(text or "")}
    return named.pop() if len(named) == 1 else None


def say_hit(text: str | None, value: str) -> bool:
    return VALUE_RE[value].search(text or "") is not None


def choose_rung(rates: dict[int, float | None]) -> int:
    parsed = {r: p for r, p in rates.items() if p is not None}
    if not parsed:
        return 1
    return min(parsed, key=lambda r: (round(abs(parsed[r] - 0.40), 9), abs(r - 1), r))


def in_band(share: float | None, unparsed: float) -> bool:
    """The on-curve rule for one cell: share among parsed in [10%, 70%] and at most 20% unparsed."""
    return (share is not None and BAND[0] <= round(share, 9) <= BAND[1]
            and round(unparsed, 9) <= MAX_UNPARSED)


# ---------------------------------------------------------------- the calibration report

def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _value_of(name: str) -> str:
    """The value a label candidate is known for: its family's."""
    return next(f for f in I.FAMILIES if name in I.LABEL_CANDIDATES[f])


def _load(dirs: list[Path], m: str, problems: list[str]) -> dict | None:
    """One model's calibration folder: {"info", "dec", "know"}; or None after noting the problem."""
    found = [d / m for d in dirs if (d / m).is_dir()]
    if len(found) != 1:
        problems.append(f"{m}: {'no' if not found else len(found)} calibration folders in {', '.join(map(str, dirs))}")
        return None
    folder = found[0]
    if not (folder / "run_info.json").exists():
        problems.append(f"{m}: {folder} has no run_info.json")
        return None
    info = json.loads((folder / "run_info.json").read_text(encoding="utf-8"))
    if info.get("status") == "load_failed":
        return {"info": info, "dec": [], "know": []}
    jobs = v8.build_jobs("calib", m)
    dec, know = _jsonl(folder / "decisions.jsonl"), _jsonl(folder / "know.jsonl")
    if info.get("status") != "done":
        problems.append(f"{m}: run_info status is {info.get('status')!r}, not done ({len(dec)} decisions on disk): "
                        "rerun the model; only status load_failed makes it not servable")
    elif info.get("prompt_hash") != v8.prompt_hash(jobs):
        problems.append(f"{m}: run_info prompt hash {info.get('prompt_hash')} is not the calib jobs' "
                        f"{v8.prompt_hash(jobs)}")
    else:
        for kind, recs in (("decide", dec), ("know", know)):
            want = sorted(j["key"] for j in jobs if j["kind"] == kind)
            if sorted(r["key"] for r in recs) != want:
                problems.append(f"{m}: {len(recs)} {kind} records in {folder}, not its {len(want)} calib jobs")
    return {"info": info, "dec": dec, "know": know}


def _cells(dec: list[dict]) -> dict:
    """{family: {rung: {"n", "parsed", "hits"}}} over the decisions (the none line at each rung)."""
    out = {f: {r: Counter() for r in range(len(I.LADDER[f]))} for f in I.FAMILIES}
    for rec in dec:
        y = decision(rec)
        c = out[rec["family"]][rec["base"]]
        c["n"] += 1
        c["parsed"] += y is not None
        c["hits"] += y == 1
    return out


def _share(c: Counter) -> float | None:
    return c["hits"] / c["parsed"] if c["parsed"] else None


def _unparsed(c: Counter) -> float:
    return (c["n"] - c["parsed"]) / c["n"] if c["n"] else 1.0


def calib_report(dirs, md, frozen_path) -> dict:
    """Choose each model's rung per family, decide servability and the labels; write md and frozen_path."""
    dirs, md, frozen_path = [Path(d) for d in dirs], Path(md), Path(frozen_path)
    problems, runs = [], {}
    for m in v8.MODELS:
        runs[m] = _load(dirs, m, problems)
    if problems:
        raise SystemExit("[v8] calibration is not complete, so nothing is frozen (a harness problem, not a servability "
                         "verdict):\n  " + "\n  ".join(problems))
    rungs, on_curve, servable, calib, low = {}, {}, {}, {}, {f: Counter() for f in I.FAMILIES}
    for m, run in runs.items():
        info = run["info"]
        servable[m] = info["status"] != "load_failed"
        calib[m] = {"status": info["status"], "prompt_hash": info.get("prompt_hash")}
        if not servable[m]:
            calib[m]["error"] = info.get("error", "")
            on_curve[m] = {f: False for f in I.FAMILIES}
            continue
        cells = _cells(run["dec"])
        rungs[m], on_curve[m] = {}, {}
        for f in I.FAMILIES:
            r = choose_rung({k: _share(c) for k, c in cells[f].items()})
            rungs[m][f] = int(r)
            on_curve[m][f] = bool(in_band(_share(cells[f][r]), _unparsed(cells[f][r])))
        calib[m]["cells"] = {f: {str(k): {"n": c["n"], "parsed": c["parsed"],
                                          "target": None if _share(c) is None else round(_share(c), 4),
                                          "unparsed": round(_unparsed(c), 4)} for k, c in cells[f].items()}
                             for f in I.FAMILIES}
        know = {n: Counter() for f in I.FAMILIES for n in I.LABEL_CANDIDATES[f]}
        for rec in run["know"]:
            name = rec["unit"].split("~")[0]
            know[name]["n"] += 1
            know[name]["named"] += know_value(rec) == _value_of(name)
        calib[m]["know"] = {n: {"n": c["n"], "named": c["named"], "share": round(c["named"] / c["n"], 4)}
                            for n, c in know.items()}
        for f in I.FAMILIES:
            for n in I.LABEL_CANDIDATES[f]:
                low[f][n] += know[n]["named"] < 0.5 * know[n]["n"]
    labels, label_rule = {}, {}
    for f in I.FAMILIES:
        primary, backup = I.LABEL_CANDIDATES[f]
        labels[f] = backup if low[f][primary] >= 3 and low[f][backup] < low[f][primary] else primary
        label_rule[f] = {"low": {n: low[f][n] for n in I.LABEL_CANDIDATES[f]}, "label": labels[f]}
    frozen = {"rungs": rungs, "on_curve": on_curve, "servable": servable, "labels": labels, "calib": calib,
              "label_rule": label_rule}
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(_summary(dirs, runs, frozen), encoding="utf-8")
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(json.dumps(frozen, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return frozen


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def _summary(dirs: list[Path], runs: dict, fz: dict) -> str:
    L = ["# PA-V8 calibration", "",
         f"From {', '.join(map(str, dirs))}. The none line at each of the 5 rungs: 16 items x 4 samples = 64 decisions "
         "per family x rung. Target % is among parsed decisions; chance is 1/3. The rung is the one closest to 40% "
         "(ties: nearer rung 1, then the lower rung); the cell is on the curve iff that share is in 10-70% with at "
         "most 20% unparsed. A model is not servable only when its vLLM load failed (Change 5).", "",
         "## Servability", "", "| model | HF id | status | error |", "|---|---|---|---|"]
    for m, d in v8.MODELS.items():
        c = fz["calib"][m]
        L.append(f"| {m} | {d['hf']} | {c['status']} | {c.get('error', '')[:300]} |")
    L += ["", "Read every load error before accepting a model as not servable: a download error or GPU memory left by "
          "an earlier process also lands there.", "",
          "## Rungs: target % (unparsed %) per rung", "",
          "| model | family | r0 | r1 | r2 | r3 | r4 | rung | on curve |", "|---|---|---|---|---|---|---|---|---|"]
    off_band = Counter()
    for m in v8.MODELS:
        if not fz["servable"][m]:
            continue
        for f in I.FAMILIES:
            cells = fz["calib"][m]["cells"][f]
            row = [f"{_pct(c['target'])} ({_pct(c['unparsed'])})" for c in cells.values()]
            L.append(f"| {m} | {f} | " + " | ".join(row) + f" | {fz['rungs'][m][f]} | {fz['on_curve'][m][f]} |")
            off_band[f] += not any(in_band(c["target"], 0.0) for c in cells.values())   # the share alone
    L += ["", "Servable models whose target share is outside 10-70% at every rung (unparsed aside): "
          + ", ".join(f"{f} {off_band[f]}" for f in I.FAMILIES)
          + " (Task 6 Step 3: a family at >= 4 gets one calibration change).", "",
          "## Knowledge: share of 16 answers naming the label's value", "",
          "| model | " + " | ".join(n for f in I.FAMILIES for n in I.LABEL_CANDIDATES[f]) + " |",
          "|---|" + "---|" * sum(len(v) for v in I.LABEL_CANDIDATES.values())]
    for m in v8.MODELS:
        if fz["servable"][m]:
            k = fz["calib"][m]["know"]
            L.append(f"| {m} | " + " | ".join(_pct(k[n]["share"]) for f in I.FAMILIES for n in I.LABEL_CANDIDATES[f])
                     + " |")
    L.append("")
    for f in I.FAMILIES:
        r = fz["label_rule"][f]
        L.append(f"- {f}: models below 50% " + ", ".join(f"{n} {k}" for n, k in r["low"].items())
                 + f" -> label **{r['label']}** (the backup only if the primary is below 50% at >= 3 models and the "
                 "backup at fewer)")
    L += ["", "## Sanity read: two rung-1 transparency replies per model", ""]
    for m, run in runs.items():
        shown = [rec for rec in run["dec"] if rec["family"] == "transparency" and rec["base"] == 1][:2]
        for rec in shown:
            text = _scored_text(rec).replace("\n", " / ")[:400]
            L.append(f"- {m} {rec['unit']} ({parse_option(_scored_text(rec), rec['fmt'], rec['family'])}): `{text}`")
    return "\n".join(L) + "\n"
