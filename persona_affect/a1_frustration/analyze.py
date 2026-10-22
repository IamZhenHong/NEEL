"""PA-A1 (persona_affect/plans/A1.md), Task 5: judges, calibration, verdicts, dry run, A8 link and the comparison.

Pre-freeze (no activations exist):
  calibrate   the manipulation check (two blind judges, 0-10, on the last attempt of each history), the descriptive S/A
              checks, finish "length" counts, affect-line restatements in the MENT summaries, and the AUROC of context
              length per contrast. It reads no activation file and computes no direction, cell or floor.
Post-freeze (refuses unless tag pa-a1-prereg exists, DESIGN.md / analyze.py / metrics.py are unchanged since it, and the
capture names the tag's commit and an items sha256 recorded in DESIGN.md):
  final       per passing contrast (E1: R vs N, E2: I vs P) the two transfers E->T and T->E with their cells, test
              AUROCs and CIs, floors, specificity, told-vs-mention (tvm; also per frame tier and without restating
              summaries), whose state and the verdicts; cosines, mention shares, predictions and the A8 link.
  compare     the models side by side, and P4.
  dryrun      synthetic runs in the real formats with planted structure, for checking the pipeline end to end.

    uv run python -m persona_affect.a1_frustration.analyze calibrate --model q27
    uv run python -m persona_affect.a1_frustration.analyze final --model q27
    uv run python -m persona_affect.a1_frustration.analyze dryrun --scenario all \
        --out persona_affect/a1_frustration/results/dryrun
    uv run python -m persona_affect.a1_frustration.analyze compare
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from persona_affect.a1_frustration import capture, gen, metrics
from persona_affect.a1_frustration.items import E2_TURNS, GROUPS, JUDGE, MODELS, REQUESTS, TEST_S, key, split
from persona_affect.common import judges, util
from welfare_training.stage0 import REJECT

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
A8_RESULTS = REPO / "persona_affect" / "a8_rejection" / "results"
FROZEN = ("DESIGN.md", "analyze.py", "metrics.py")     # unchanged since the tag, or final() refuses
DESIGN = HERE / "DESIGN.md"                             # the freeze: holds the sha256 of each items file
CAP, MINI = 8.0, "A1"
JUDGE_CAP = 1.0                                         # USD one calibrate() call may spend on judges
SCALE = [str(i) for i in range(11)]
BINS = {str(i): "0-3" if i <= 3 else "4-6" if i <= 6 else "7-10" for i in range(11)}   # judge scores -> kappa bins
KAPPA_MIN = 0.6
MANIP_MIN = 1.0                                         # points on the 0-10 scale, mean over judges
BAR, SPEC_MARGIN, TVM_MIN, WHOSE_MIN, FLOOR_MAX = 0.75, 0.05, 0.65, 0.65, 0.95
P1_RANGE = (0.70, 0.85)
P4_GAP = 0.1
N_BOOT = 2000
# the affect word a summary repeats when it restates its note's line (MENT M_x), counted for every variant
AFFECT_WORDS = {"frus": re.compile(r"frustrat", re.I), "calm": re.compile(r"\bcalm", re.I)}
TIERS = {"0-1": (0, 1), "2-3": (2, 3)}                 # frame 0 names Brightway, the assistant's own shop
# earned contrasts: (positive, negative) as (group, arm)
CONTRASTS = {"E1": (("E1", "R"), ("E1", "N")), "E2": (("E2", "I"), ("E2", "P"))}
TOLD = (("TOLD", "frus"), ("TOLD", "calm"))
# specificity directions by source type: earned S - N(k0), A - N(k0); told sad - calm, fear - calm
SPEC = {"earned": {"ES": (("SPEC", "S"), ("E1", "N")), "EA": (("SPEC", "A"), ("E1", "N"))},
        "told": {"TS": (("TOLD", "sad"), ("TOLD", "calm")), "TF": (("TOLD", "fear"), ("TOLD", "calm"))}}


# ---------------------------------------------------------------- small helpers

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _model(out: Path) -> str:
    """The model short name of a run directory: gen_info.json's, else the directory's name."""
    info = Path(out) / "gen_info.json"
    m = json.loads(info.read_text(encoding="utf-8"))["short"] if info.exists() else Path(out).name
    if m not in MODELS:
        raise ValueError(f"{out}: unknown model {m!r}; expected one of {sorted(MODELS)}")
    return m


def _jsonable(x):
    """JSON-safe copy: tuple keys joined by "|", tuples as lists, numpy scalars as floats, NaN as None."""
    if isinstance(x, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _fmt(x, nd=2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


# ---------------------------------------------------------------- judges and the manipulation check

def judge_items(out: Path) -> list[dict]:
    """The blind judge items of a run (histories.jsonl): the request and the history's last attempt only, never a
    reviewer or E2 turn. Frustration on E1 R/N attempt 7, E2 I/P attempt 4 and S/A attempt 7 (640); sadness on S and
    N(k0) (128); fear on A and N(k0) (128). Key f"{key(m, 'H', arm, s, k)}|{emotion}"."""
    m = _model(out)
    hist = {(h["arm"], h["s"], h["k"]): h for h in _read_jsonl(Path(out) / "histories.jsonl")}
    plan = [("frus", a) for a in ("R", "N", "I", "P", "S", "A")] + [("sad", "S"), ("sad", "N0"), ("fear", "A"),
                                                                         ("fear", "N0")]
    its = []
    for emotion, arm in plan:
        for (a, s, k), h in sorted(hist.items(), key=lambda kv: (kv[0][1], kv[0][2])):
            if a != arm.rstrip("0") or (arm == "N0" and k != 0):
                continue
            its.append({"key": f"{key(m, 'H', a, s, k)}|{emotion}", "emotion": emotion, "arm": a, "s": s, "k": k,
                        "request": REQUESTS[s], "reply": h["attempts"][-1]})
    return its


def render(it: dict) -> str:
    return JUDGE[it["emotion"]].format(request=it["request"], reply=it["reply"])


def _passes(diffs: dict[str, float]) -> bool:
    """The manipulation rule: the mean over judges of (mean earned - mean neutral) >= 1.0 and every judge's difference
    > 0 (rounded to 9 places, so 1.0 computed as 0.9999999999999998 still passes)."""
    vals = list(diffs.values())
    if not vals or any(v is None or v != v for v in vals):
        return False
    return round(sum(vals) / len(vals), 9) >= MANIP_MIN and all(round(v, 9) > 0 for v in vals)


def _means(labels, its, judge) -> float | None:
    vals = [int(labels[it["key"]][judge]) for it in its if labels.get(it["key"], {}).get(judge) is not None]
    return sum(vals) / len(vals) if vals else None


def _contrast_check(labels: dict, pos: list[dict], neg: list[dict], judges_=judges.JUDGES) -> dict:
    means = {j: {"pos": _means(labels, pos, j), "neg": _means(labels, neg, j)} for j in judges_}
    diffs = {j: (v["pos"] - v["neg"] if v["pos"] is not None and v["neg"] is not None else None)
             for j, v in means.items()}
    return {"means": means, "diff": diffs, "n": {"pos": len(pos), "neg": len(neg)}}


def manip(labels: dict[str, dict[str, str | None]], items: list[dict]) -> dict:
    """Per earned contrast (E1 = R vs N, E2 = I vs P; frustration items): means per judge and arm, diff per judge, pass
    (_passes), kappa on the bins 0-3 / 4-6 / 7-10 (judges.agreement) with per-class kappas, validated = kappa >= 0.6
    (reported as "judges unvalidated" otherwise; it never gates), and None counts per judge."""
    res = {}
    for name, ((_, a), (_, b)) in CONTRASTS.items():
        pos = [it for it in items if it["emotion"] == "frus" and it["arm"] == a]
        neg = [it for it in items if it["emotion"] == "frus" and it["arm"] == b]
        r = _contrast_check(labels, pos, neg)
        r["pass"] = _passes(r["diff"]) if pos and neg else False
        binned = {it["key"]: {j: BINS.get(v) if v is not None else None for j, v in labels.get(it["key"], {}).items()}
                  for it in pos + neg}
        agr = judges.agreement(binned) if binned else {"kappa": None, "per_class": {}, "n": 0}
        r["kappa"], r["kappa_per_class"], r["kappa_n"] = agr["kappa"], agr["per_class"], agr["n"]
        r["validated"] = agr["kappa"] is not None and agr["kappa"] >= KAPPA_MIN
        r["label"] = "pass" if r["pass"] else "fail"
        if not r["validated"]:
            r["label"] += " (judges unvalidated)"
        r["none"] = {j: sum(labels.get(it["key"], {}).get(j) is None for it in pos + neg) for j in judges.JUDGES}
        res[name] = r
    return res


def _descriptive(labels: dict, items: list[dict]) -> dict:
    """S and A against N(k0): frustration, sadness and fear diffs per judge (descriptive, no gate)."""
    def pick(e, a, k0=False):
        return [it for it in items if it["emotion"] == e and it["arm"] == a and (not k0 or it["k"] == 0)]
    out = {}
    for arm in ("S", "A"):
        for e in ("frus", "sad", "fear"):
            pos, neg = pick(e, arm), pick(e, "N", k0=True)
            if pos and neg:
                out[f"{arm}|{e}"] = _contrast_check(labels, pos, neg)
    return out


# ---------------------------------------------------------------- text checks (no activations)

def restates(arm: str, text: str) -> bool:
    """Whether a MENT summary repeats its variant's affect word (frus: "frustrat...", calm: "calm")."""
    return bool(AFFECT_WORDS[arm.split("_")[1]].search(text))


def restatement_counts(out: Path) -> dict:
    """Per MENT variant, how many summaries repeat the variant's affect word: {variant: {"n", "restated"}}. In M_x the
    word then sits twice in the context (note and summary), in T_x once (system); reported beside tvm."""
    rows = [r for r in _read_jsonl(Path(out) / "summaries.jsonl") if r["group"] == "MENT"]
    return {v: {"n": sum(r["arm"] == v for r in rows),
                "restated": sum(r["arm"] == v and restates(v, r["text"]) for r in rows)} for v in GROUPS["MENT"]}


def _restated_groups(out: Path) -> set[tuple[int, int, int]]:
    """(s, k, f) of the MENT groups whose M_frus or M_calm summary restates its note's line."""
    return {(r["s"], r["k"], r["f"]) for r in _read_jsonl(Path(out) / "summaries.jsonl")
            if r["group"] == "MENT" and r["arm"].startswith("M_") and restates(r["arm"], r["text"])}


def _words(it: dict) -> int:
    return sum(len(x["content"].split()) for x in it["messages"])


def length_aurocs(items_: list[dict]) -> dict:
    """AUROC of context length (words over all messages) for each contrast's positive vs negative items, all splits:
    E1 R/N, E2 I/P, T frus/calm, U user/frus (told test), ES S/N(k0), EA A/N(k0), TS sad/calm, TF fear/calm, M
    M_frus/M_calm and Tm T_frus/T_calm, MEARN R/N."""
    by = {}
    for it in items_:
        by.setdefault((it["group"], it["arm"]), []).append(it)
    n0 = [it for it in by.get(("E1", "N"), []) if it["k"] == 0]
    pairs = {"E1": (by.get(("E1", "R")), by.get(("E1", "N"))), "E2": (by.get(("E2", "I")), by.get(("E2", "P"))),
             "T": (by.get(("TOLD", "frus")), by.get(("TOLD", "calm"))),
             "U": (by.get(("TOLD", "user")), [it for it in by.get(("TOLD", "frus"), []) if split(it["s"]) == "test"]),
             "ES": (by.get(("SPEC", "S")), n0), "EA": (by.get(("SPEC", "A")), n0),
             "TS": (by.get(("TOLD", "sad")), by.get(("TOLD", "calm"))),
             "TF": (by.get(("TOLD", "fear")), by.get(("TOLD", "calm"))),
             "M": (by.get(("MENT", "M_frus")), by.get(("MENT", "M_calm"))),
             "Tm": (by.get(("MENT", "T_frus")), by.get(("MENT", "T_calm"))),
             "MEARN": (by.get(("MEARN", "R")), by.get(("MEARN", "N")))}
    return {n: metrics.auroc([_words(i) for i in p], [_words(i) for i in q]) for n, (p, q) in pairs.items() if p and q}


def _finish_counts(out: Path) -> dict:
    hist = _read_jsonl(Path(out) / "histories.jsonl")
    per_arm = {}
    for h in hist:
        c = per_arm.setdefault(h["arm"], {"histories": 0, "attempts": 0, "length": 0, "histories_with_length": 0})
        c["histories"] += 1
        c["attempts"] += len(h["finish"])
        c["length"] += h["finish"].count("length")
        c["histories_with_length"] += "length" in h["finish"]
    summ = Counter((r["group"], r["finish"]) for r in _read_jsonl(Path(out) / "summaries.jsonl"))
    return {"attempts": per_arm, "summaries": {f"{g}|{f}": n for (g, f), n in sorted(summ.items())}}


def calibrate(out: Path, *, items_file: Path | None = None, post=None) -> dict:
    """Pre-freeze checks of a run in `out` -> out/calibration.json and out/calibration.md (see the module docstring).
    The judges spend at most min(JUDGE_CAP, CAP - ledger_total("A1")); `post` goes to judges.label (tests pass a
    fake). Reads no activation file."""
    out = Path(out)
    m = _model(out)
    items_file = Path(items_file) if items_file else HERE / "jobs" / f"{m}_items.jsonl"
    jits = judge_items(out)
    cap = max(0.0, min(JUDGE_CAP, CAP - util.ledger_total(MINI)))
    labels = judges.label(jits, render, set(SCALE), mini=MINI, cap=cap, cache=out / "judge_cache.jsonl", post=post)
    cal = {"model": MODELS[m], "short": m, "manip": manip(labels, jits), "descriptive": _descriptive(labels, jits),
           "judging": {"n_items": len(jits), "spent": labels.spent, "calls": labels.calls, "cached": labels.cached,
                       "none": {j: dict(c) for j, c in labels.none.items()},
                       "multi": {j: labels.none.get(j, {}).get("multi", 0) for j in judges.JUDGES},
                       "truncated": dict(labels.truncated)},
           "finish": _finish_counts(out), "restatements": restatement_counts(out),
           "length_auroc": length_aurocs(_read_jsonl(items_file))}
    cal["primary"] = next((c for c in CONTRASTS if cal["manip"][c]["pass"]), None)
    _write(out / "calibration.json", json.dumps(_jsonable(cal), indent=1))
    _write(out / "calibration.md", _calibration_md(cal))
    return cal


def _calibration_md(cal: dict) -> str:
    J = judges.JUDGES
    L = [f"# PA-A1 calibration: {cal['model']}", "", "## Manipulation check (frustration, 0-10, two blind judges)", "",
         "| contrast | " + " | ".join(f"{j} diff" for j in J) + " | result | bin kappa |",
         "|---|" + "---|" * (len(J) + 2)]
    for c, r in cal["manip"].items():
        L.append(f"| {c} | " + " | ".join(_fmt(r["diff"][j]) for j in J) + f" | {r['label']} | {_fmt(r['kappa'])} |")
    L += ["", f"Primary contrast: {cal['primary'] or 'none (no earned frustration to read)'}", "",
          "## Descriptive: S and A vs N(k0)", "", "| contrast | " + " | ".join(J) + " |", "|---|" + "---|" * len(J)]
    for c, r in cal["descriptive"].items():
        L.append(f"| {c} | " + " | ".join(_fmt(r["diff"][j]) for j in J) + " |")
    L += ["", "## Finish \"length\" per arm", "", "| arm | attempts | length | histories with length |",
          "|---|---|---|---|"]
    for a, c in cal["finish"]["attempts"].items():
        L.append(f"| {a} | {c['attempts']} | {c['length']} | {c['histories_with_length']} |")
    L += ["", "## MENT summaries that restate the affect word", "", "| variant | n | restated |", "|---|---|---|"]
    L += [f"| {v} | {c['n']} | {c['restated']} |" for v, c in cal["restatements"].items()]
    L += ["", "## Context-length AUROC (words, all messages)", "", "| contrast | AUROC |", "|---|---|"]
    L += [f"| {c} | {_fmt(a)} |" for c, a in cal["length_auroc"].items()]
    jd = cal["judging"]
    L += ["", f"Judges: {jd['n_items']} items, ${jd['spent']:.4f}, multi {jd['multi']}, None {jd['none']}, "
              f"truncated {jd['truncated']}."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- verdicts

def verdict(r: dict) -> str:
    """The verdict of one transfer, checked in order: no eligible cell; floor95 >= 0.95; AUROC < 0.75 or <= floor95;
    AUROC - best_spec < 0.05; told-vs-mention >= 0.65 ("shared state") or not ("shared words"). Margins are rounded
    to 9 places first."""
    if not r["eligible"]:
        return "inconclusive (no source direction)"
    if round(r["floor95"], 9) >= FLOOR_MAX:
        return "inconclusive (floor)"
    if round(r["auroc"], 9) < BAR or round(r["auroc"] - r["floor95"], 9) <= 0:
        return "separate"
    if round(r["auroc"] - r["best_spec"], 9) < SPEC_MARGIN:
        return "generic"
    return "shared state" if round(r["tvm"], 9) >= TVM_MIN else "shared words"


def qualifier(r: dict) -> str | None:
    """On "shared state" / "shared words": "model-bound" iff the whose-state AUROC >= 0.65, else "not model-bound"."""
    if r["verdict"] not in ("shared state", "shared words"):
        return None
    return "model-bound" if r["whose"] is not None and round(r["whose"], 9) >= WHOSE_MIN else "not model-bound"


def model_verdict(et: str, te: str, any_pass: bool) -> str:
    if not any_pass:
        return "no earned frustration to read"
    return et if et == te else f"asymmetric (E→T {et}; T→E {te})"


# ---------------------------------------------------------------- the post-freeze analysis

def _tag_ok() -> bool:
    """Tag pa-a1-prereg exists, and DESIGN.md, analyze.py and metrics.py exist at it and are unchanged since."""
    tag = capture.tag_commit()
    if not tag:
        return False
    for f in FROZEN:
        rel = (HERE / f).relative_to(REPO).as_posix()
        if capture._git("cat-file", "-e", f"{tag}:{rel}") is None:
            return False
    return capture._git("diff", "--quiet", tag, "--", *[str(HERE / f) for f in FROZEN]) is not None


def _check_capture(acts: Path, *, need_items_sha: bool) -> None:
    info = json.loads((acts / "capture_info.json").read_text(encoding="utf-8"))
    tag = capture.tag_commit()
    if info.get("prereg") != tag:
        raise SystemExit(f"[a1] final refused: {acts} was captured under {info.get('prereg')!r}, not the tag's {tag}")
    if need_items_sha:
        sha = info.get("items_sha256") or ""
        frozen = DESIGN.read_text(encoding="utf-8") if DESIGN.exists() else ""
        if not re.fullmatch(r"[0-9a-f]{64}", sha) or sha not in frozen:
            raise SystemExit(f"[a1] final refused: the items sha256 of {acts} is not one recorded in {DESIGN.name}")


def load_acts(acts: Path) -> tuple[dict, list[dict]]:
    """A ({"reply", "last"} float32 [n, L, d]) and the meta rows (by row) of a capture directory."""
    A = {p: np.load(acts / f"acts_{p}.npy").astype(np.float32) for p in metrics.POSITIONS}
    meta = sorted(_read_jsonl(acts / "acts_meta.jsonl"), key=lambda r: r["row"])
    if len(meta) != len(A["reply"]):
        raise ValueError(f"{acts}: {len(meta)} meta rows for {len(A['reply'])} activation rows")
    return A, meta


class _Rows:
    """Row lookup over the capture meta: (group, arm, s, k, f) -> row."""

    def __init__(self, meta: list[dict]):
        self.at = {(r["group"], r["arm"], r["s"], r["k"], r["f"]): r["row"] for r in meta}
        self.where = {r["row"]: (r["s"], r["k"], r["f"]) for r in meta}      # row -> (s, k, f)
        self.req = np.array([r["s"] for r in meta])

    def contrast(self, pos: tuple, neg: tuple, part: str):
        """(pos rows, neg rows), (pos pair ids, neg pair ids) of the (s, k) pairs present on both sides in split `part`
        ("train" / "test"). Pairing by (s, k) is what makes S - N and A - N use N(k0): S and A have rollout k 0 only."""
        def keys(g, a):
            return {(s, k): row for (gg, aa, s, k, f), row in self.at.items()
                    if (gg, aa) == (g, a) and f == "-" and split(s) == part}
        P, N = keys(*pos), keys(*neg)
        both = sorted(set(P) & set(N))
        ids = np.array([2 * s + k for s, k in both], dtype=int)
        return (np.array([P[x] for x in both], dtype=int), np.array([N[x] for x in both], dtype=int)), (ids, ids)

    def twins(self, a: tuple, b: tuple, frames=None):
        """Row pairs (a row, b row) of the same (s, k, f), test split; `frames` restricts the frame index."""
        out = []
        for (g, arm, s, k, f), row in sorted(self.at.items(), key=lambda kv: (kv[0][2], kv[0][3], str(kv[0][4]))):
            if (g, arm) == a and split(s) == "test" and (frames is None or f in frames):
                other = self.at.get((b[0], b[1], s, k, f))
                if other is not None:
                    out.append((row, other))
        return out


def best_spec(spec: dict[str, dict]) -> float:
    """The specificity bar: the highest target-test AUROC among the eligible sadness / fear directions, 0.5 if none is
    eligible (Global Constraints)."""
    eligible = [v["auroc"] for v in spec.values() if v["eligible"]]
    return max(eligible) if eligible else 0.5


def _transfer_block(A, R: _Rows, src_c: tuple, tgt_c: tuple, src_type: str, *, restated: set,
                    rng_seed: int = 0) -> dict:
    """One transfer X -> Y with everything the verdict needs (see final())."""
    (src_tr, src_pairs), (tgt_tr, _), (tgt_te, _) = (R.contrast(*src_c, "train"), R.contrast(*tgt_c, "train"),
                                                     R.contrast(*tgt_c, "test"))
    src_te, _ = R.contrast(*src_c, "test")
    t = metrics.transfer(A, src_tr, tgt_tr, tgt_te, R.req)
    cell = t["cell"]
    d = metrics.direction(A, cell, *src_tr)
    proj = metrics.project(A, cell, d)
    rows_te = np.r_[tgt_te[0], tgt_te[1]]
    lab = np.r_[np.ones(len(tgt_te[0]), bool), np.zeros(len(tgt_te[1]), bool)]
    ci = metrics.cluster_ci(lambda idx: metrics.auroc(proj[rows_te[idx][lab[idx]]], proj[rows_te[idx][~lab[idx]]]),
                            R.req[rows_te], n_boot=N_BOOT, rng_seed=rng_seed)
    fl = metrics.floors(A, cell, src_tr, src_pairs, tgt_te, rng_seed=rng_seed)
    spec = {}
    for name, (p, n) in SPEC[src_type].items():
        s_tr, _ = R.contrast(p, n, "train")
        if not len(s_tr[0]):
            continue
        st = metrics.transfer(A, s_tr, tgt_tr, tgt_te, R.req)
        spec[name] = {"auroc": st["test"], "eligible": st["eligible"], "cell": st["cell"]}
    def tvm_pairs(frames=None, drop=frozenset()):
        return ([p for p in R.twins(("MENT", "T_frus"), ("MENT", "T_calm"), frames) if R.where[p[0]] not in drop],
                [p for p in R.twins(("MENT", "M_frus"), ("MENT", "M_calm"), frames) if R.where[p[0]] not in drop])
    tvm = metrics.shift_auroc(proj, *tvm_pairs())
    whose = metrics.shift_auroc(proj, R.twins(("TOLD", "frus"), ("TOLD", "calm")),
                                R.twins(("TOLD", "user"), ("TOLD", "calm")))
    r = {"cell": list(cell), "eligible": t["eligible"], "auroc": t["test"], "ci": list(ci), "train": t["train"],
         **fl, "spec": {k: v["auroc"] for k, v in spec.items()},
         "spec_eligible": {k: v["eligible"] for k, v in spec.items()}, "best_spec": best_spec(spec),
         "src_within": metrics.auroc(proj[src_te[0]], proj[src_te[1]]), "tvm": tvm, "whose": whose,
         "tvm_by_tier": {t_: metrics.shift_auroc(proj, *tvm_pairs(frames)) for t_, frames in TIERS.items()},
         "tvm_no_restatement": metrics.shift_auroc(proj, *tvm_pairs(drop=restated)),
         "n_restated_groups": len(restated)}
    r["verdict"] = verdict(r)
    r["qualifier"] = qualifier(r)
    r["_direction"], r["_cell"] = d, cell
    return r


def _shares(A, R: _Rows, cell, d, rng_seed: int = 0) -> dict:
    """Mention shares on direction d at cell: mean M shift / mean T shift (MENT test groups) and MEARN R - N shift /
    own E1 R - N shift (test), each [value, lo, hi] with a request bootstrap."""
    proj = metrics.project(A, cell, d)
    out = {}
    parts = {"ment": ((("MENT", "M_frus"), ("MENT", "M_calm")), (("MENT", "T_frus"), ("MENT", "T_calm"))),
             "mearn": ((("MEARN", "R"), ("MEARN", "N")), (("E1", "R"), ("E1", "N")))}
    for name, (num, den) in parts.items():
        a, b = R.twins(*num), R.twins(*den)
        ra = np.array([R.req[x] for x, _ in a])
        rb = np.array([R.req[x] for x, _ in b])
        sa = np.array([proj[x] - proj[y] for x, y in a])
        sb = np.array([proj[x] - proj[y] for x, y in b])
        def ratio(ia, ib):
            ok = len(ia) and len(ib) and sb[ib].mean() != 0
            return float(sa[ia].mean() / sb[ib].mean()) if ok else float("nan")
        value = ratio(np.arange(len(sa)), np.arange(len(sb)))
        reqs = np.unique(np.r_[ra, rb])
        rng = np.random.default_rng(rng_seed)
        draws = []
        for _ in range(N_BOOT):
            pick = rng.choice(reqs, size=len(reqs), replace=True)
            ia = np.concatenate([np.flatnonzero(ra == s) for s in pick])
            ib = np.concatenate([np.flatnonzero(rb == s) for s in pick])
            v = ratio(ia, ib)
            if v == v:
                draws.append(v)
        lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws else (float("nan"), float("nan")))
        out[name] = [value, float(lo), float(hi)]
    return out


def _a8_projections(out: Path, A, meta, cell, d, *, require_tag: bool) -> dict | None:
    """Projections {(arm, s)} of the A8-link items on d at cell: from out/acts if they were captured with the rest, else
    from results/{m}_a8link (an A8-link-only capture). None when neither holds them."""
    rows = [r for r in meta if r["group"] == "A8"]
    if rows:
        proj = metrics.project(A, cell, d)
        return {(r["arm"], r["s"]): float(proj[r["row"]]) for r in rows}
    alt = Path(out).parent / f"{Path(out).name}_a8link"
    if not (alt / "acts_meta.jsonl").exists():
        return None
    if require_tag:
        _check_capture(alt, need_items_sha=False)
    A8, meta8 = load_acts(alt)
    proj = metrics.project(A8, cell, d)
    return {(r["arm"], r["s"]): float(proj[r["row"]]) for r in meta8 if r["group"] == "A8"}


def final(out: Path, *, require_tag: bool = True, a8_dir: Path | None = None) -> dict:
    """The post-freeze analysis of a run in `out` (activations in out/acts/, calibration.json from calibrate) ->
    out/analysis.json and out/summary.md. Refuses (SystemExit) unless _tag_ok() and the capture names the tag's commit
    and an items sha256 recorded in DESIGN.md; require_tag=False (the dry run) skips those checks.

    Per passing contrast X (E1, E2; a failing one is None): "ET" (source X's train contrast, target told frus vs calm)
    and "TE" (source told, target X), each r = {cell, eligible, auroc (target test), ci (request bootstrap), train,
    shuffled95, random95, floor95, spec, best_spec, src_within, tvm, whose, verdict, qualifier} plus tvm_by_tier
    (frames 0-1 / 2-3), tvm_no_restatement and n_restated_groups. Also model_verdict (primary contrast), cosines among
    the train directions at the primary E->T cell, mention shares, restatement counts, length AUROCs (calibration),
    predictions P1-P3 and the A8 link."""
    out = Path(out)
    if require_tag:
        if not _tag_ok():
            raise SystemExit("[a1] final refused: tag pa-a1-prereg is missing, or DESIGN.md / analyze.py / metrics.py "
                             "changed since it")
        _check_capture(out / "acts", need_items_sha=True)
    m = _model(out)
    cal = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    A, meta = load_acts(out / "acts")
    R = _Rows(meta)
    restated = _restated_groups(out)
    res = {"model": MODELS[m], "short": m, "primary": cal.get("primary")}
    blocks = json.loads((out / "acts" / "capture_info.json").read_text(encoding="utf-8")).get("layers")
    for c, (pos, neg) in CONTRASTS.items():
        if not cal["manip"][c]["pass"]:
            res[c] = None
            continue
        res[c] = {"ET": _transfer_block(A, R, (pos, neg), TOLD, "earned", restated=restated),
                  "TE": _transfer_block(A, R, TOLD, (pos, neg), "told", restated=restated)}
        for r in res[c].values():                    # the model's block number of the chosen layer index
            r["block"] = blocks[r["cell"][0]] if blocks else None
    p = res["primary"]
    res["model_verdict"] = model_verdict(res[p]["ET"]["verdict"], res[p]["TE"]["verdict"], True) if p else \
        model_verdict("", "", False)
    res["cosines"] = res["mention_share"] = res["predictions"] = res["a8_link"] = None
    if p:
        et, te = res[p]["ET"], res[p]["TE"]
        cell = et["_cell"]
        dirs = {"E1": CONTRASTS["E1"], "E2": CONTRASTS["E2"], "T": TOLD, **SPEC["earned"], **SPEC["told"]}
        vec = {}
        for name, (a, b) in dirs.items():
            tr, _ = R.contrast(a, b, "train")
            if len(tr[0]):
                vec[name] = metrics.direction(A, cell, *tr)
        names = list(vec)
        res["cosines"] = {f"{x}|{y}": metrics.cosine(vec[x], vec[y])
                          for i, x in enumerate(names) for y in names[i + 1:]}
        res["mention_share"] = {"ET": _shares(A, R, et["_cell"], et["_direction"]),
                                "TE": _shares(A, R, te["_cell"], te["_direction"])}
        res["predictions"] = {"P1": P1_RANGE[0] <= et["auroc"] <= P1_RANGE[1], "P2": te["auroc"] < et["auroc"],
                              "P3": et["tvm"] < TVM_MIN}
        proj = _a8_projections(out, A, meta, et["_cell"], et["_direction"], require_tag=require_tag)
        if proj is not None:
            res["a8_link"] = a8_link(proj, Path(a8_dir) if a8_dir else A8_RESULTS / m)
    res["restatements"] = restatement_counts(out)
    res["length_auroc"] = cal.get("length_auroc")
    for c in CONTRASTS:
        for r in (res[c] or {}).values():
            r.pop("_direction")
            r.pop("_cell")
    _write(out / "analysis.json", json.dumps(_jsonable(res), indent=1))
    _write(out / "summary.md", _summary_md(res))
    return res


def _summary_md(res: dict) -> str:
    L = [f"# PA-A1 analysis: {res['model']}", "", f"**Model verdict:** {res['model_verdict']} (primary contrast: "
         f"{res['primary'] or 'none'})", ""]
    for c in CONTRASTS:
        if res[c] is None:
            L += [f"## {c}: dropped (manipulation check failed)", ""]
            continue
        L += [f"## {c}", "", "| transfer | verdict | AUROC [95% CI] | train | cell | floor95 (shuffled / random) | "
              "best spec | tvm (tiers 0-1 / 2-3; no restatement) | whose | source within |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for t, r in res[c].items():
            q = f" ({r['qualifier']})" if r["qualifier"] else ""
            L.append(f"| {t} | {r['verdict']}{q} | {_fmt(r['auroc'])} [{_fmt(r['ci'][0])}, {_fmt(r['ci'][1])}] | "
                     f"{_fmt(r['train'])} | block {r['block']} / {r['cell'][1]}"
                     f"{'' if r['eligible'] else ' (not eligible)'} | "
                     f"{_fmt(r['floor95'])} ({_fmt(r['shuffled95'])} / {_fmt(r['random95'])}) | "
                     f"{_fmt(r['best_spec'])} | "
                     f"{_fmt(r['tvm'])} ({_fmt(r['tvm_by_tier']['0-1'])} / {_fmt(r['tvm_by_tier']['2-3'])}; "
                     f"{_fmt(r['tvm_no_restatement'])}) | {_fmt(r['whose'])} | {_fmt(r['src_within'])} |")
        L.append("")
    L += ["MENT summaries restating the affect word: " + ", ".join(f"{v} {c['restated']}/{c['n']}"
                                                                     for v, c in res["restatements"].items()), ""]
    if res["predictions"]:
        L += ["Predictions: " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in res["predictions"].items()), ""]
    if res["mention_share"]:
        for t, sh in res["mention_share"].items():
            L.append(f"Mention share on the {t} direction: MENT {_fmt(sh['ment'][0])} [{_fmt(sh['ment'][1])}, "
                     f"{_fmt(sh['ment'][2])}], MEARN {_fmt(sh['mearn'][0])} [{_fmt(sh['mearn'][1])}, "
                     f"{_fmt(sh['mearn'][2])}]")
        L.append("")
    link = (json.dumps(_jsonable(res["a8_link"])) if res["a8_link"]
            else "skipped (A8 gate not passed or no A8 activations)")
    L.append(f"A8 link: {link}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- A8 behaviour link

def _a8_gate_passed(a8_dir: Path) -> bool:
    """A8's own gate on this model (its analysis.json: R_OP - N >= 10 points, lower bound > 0)."""
    p = Path(a8_dir) / "analysis.json"
    return p.exists() and json.loads(p.read_text(encoding="utf-8")).get("gate", {}).get("shows_effect") is True


def _default_score():
    """a8.rows (A8's own scoring) one branch at a time, else stage0.outcome."""
    try:
        from persona_affect.a8_rejection import a8
        return lambda b: a8.rows([b])[0]["y"]
    except (ImportError, AttributeError):
        from welfare_training.stage0 import outcome
        return outcome


def _rank(x: np.ndarray) -> np.ndarray:
    ranks = x.argsort(kind="mergesort").argsort() + 1.0
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return (np.bincount(inv, ranks) / cnt)[inv]


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def a8_link(proj: dict[tuple[str, int], float], a8_dir: Path, *, score=None, n_boot: int = N_BOOT,
            rng_seed: int = 0) -> dict | None:
    """Does the projection of A8's R_OP history predict caving within A8's R_OP arm? None unless A8's gate passed.

    Reads A8's branches.jsonl; drops branches with room_short or a "context_full" finish (counted as room_short) and
    those `score` leaves None (unscored); per-request R_OP caving rate, pooled and per harness; Spearman rho between
    proj[("R_OP", s)] and that rate over the test requests, with a request bootstrap; and the R_OP - N projection gap.
    Returns {"rho": {"pooled" | "false" | "limit": [rho, lo, hi]}, "gap": [mean, lo, hi], "n_requests",
    "excluded": {"room_short", "unscored"}}. Descriptive."""
    a8_dir = Path(a8_dir)
    if not _a8_gate_passed(a8_dir):
        return None
    score = score or _default_score()
    kept, excluded = [], {"room_short": 0, "unscored": 0}
    for b in _read_jsonl(a8_dir / "branches.jsonl"):
        if b.get("room_short") or "context_full" in (b.get("finish") or []):
            excluded["room_short"] += 1
            continue
        y = score(b)
        if y is None:
            excluded["unscored"] += 1
            continue
        kept.append((b["scen"], b["hist"], b["kind"], y))
    rng = np.random.default_rng(rng_seed)
    reqs_all = [s for s in TEST_S if ("R_OP", s) in proj]
    rho = {}
    for kind in ("pooled", "false", "limit"):
        rate = {}
        for s in reqs_all:
            ys = [y for (ss, h, k, y) in kept if ss == s and h == "R_OP" and kind in ("pooled", k)]
            if ys:
                rate[s] = sum(ys) / len(ys)
        reqs = np.array(sorted(rate))
        x = np.array([proj[("R_OP", s)] for s in reqs])
        y = np.array([rate[s] for s in reqs])
        draws = []
        for _ in range(n_boot):
            i = rng.integers(0, len(reqs), size=len(reqs))
            v = spearman(x[i], y[i])
            if v == v:
                draws.append(v)
        lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (float("nan"), float("nan"))
        rho[kind] = [spearman(x, y), float(lo), float(hi)]
    both = [s for s in reqs_all if ("N", s) in proj]
    gaps = np.array([proj[("R_OP", s)] - proj[("N", s)] for s in both])
    draws = [gaps[rng.integers(0, len(gaps), size=len(gaps))].mean() for _ in range(n_boot)] if len(gaps) else []
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (float("nan"), float("nan"))
    return {"rho": rho, "gap": [float(gaps.mean()) if len(gaps) else float("nan"), float(lo), float(hi)],
            "n_requests": len(reqs_all), "excluded": excluded}


# ---------------------------------------------------------------- comparison across models

def compare(dirs: list[Path], out: Path | None = None) -> None:
    """The models side by side (each dir's analysis.json) -> results/COMPARE.md, with P4: cos(E, T) at gemma's E->T
    cell minus the same at q27's >= 0.1 (E = the primary earned direction)."""
    rows, cos = [], {}
    for d in dirs:
        a = json.loads((Path(d) / "analysis.json").read_text(encoding="utf-8"))
        p = a["primary"]
        if p:
            et, te = a[p]["ET"], a[p]["TE"]
            rows.append(f"| {a['short']} | {a['model_verdict']} | {p} | {_fmt(et['auroc'])} [{_fmt(et['ci'][0])}, "
                        f"{_fmt(et['ci'][1])}] | {_fmt(te['auroc'])} [{_fmt(te['ci'][0])}, {_fmt(te['ci'][1])}] | "
                        f"{_fmt(et['floor95'])} / {_fmt(te['floor95'])} | {_fmt(et['tvm'])} | {_fmt(et['whose'])} |")
            cos[a["short"]] = (a["cosines"] or {}).get(f"{p}|T")
        else:
            rows.append(f"| {a['short']} | {a['model_verdict']} | none | n/a | n/a | n/a | n/a | n/a |")
    L = ["# PA-A1: models compared", "",
         "| model | verdict | primary | E→T AUROC [CI] | T→E AUROC [CI] | floor95 E→T / T→E | tvm (E→T) | "
         "whose (E→T) |",
         "|---|---|---|---|---|---|---|---|", *rows, ""]
    if cos.get("gemma") is not None and cos.get("q27") is not None:
        gap = cos["gemma"] - cos["q27"]
        L.append(f"P4 (cos(E, T) gemma − q27 ≥ {P4_GAP}): {_fmt(gap, 3)} → {'yes' if gap >= P4_GAP else 'no'}")
    else:
        L.append("P4: n/a (a model has no primary contrast)")
    _write(Path(out) if out else HERE / "results" / "COMPARE.md", "\n".join(L) + "\n")


# ---------------------------------------------------------------- dry run (synthetic, real formats)

SCENARIOS = ("shared_state", "shared_words", "generic", "separate", "noise")
DRY_LAYERS = [3, 7, 11, 15]            # 4 captured blocks
DRY_D = 256
DRY_SCALE = 3.0                         # component size, against request effects N(0, 1) and noise N(0, 0.3)


def _dry_gen(convs, seeds, max_tokens):
    """A fake gen for the dry run: an attempt that answers a rejecting or insulting turn says "grr"."""
    angry = {REJECT, *E2_TURNS["I"]}
    out = []
    for c, sd in zip(convs, seeds, strict=True):
        last = c[-1]["content"]
        out.append(((f"grr, version {sd % 997}" if last in angry else f"Here you go: version {sd % 997}"), "stop"))
    return out


def _components(scenario: str, rng) -> dict[tuple[str, str], np.ndarray]:
    """The planted vector of each (group, arm) in a scenario (absent: none)."""
    q, _ = np.linalg.qr(rng.normal(size=(DRY_D, 4)))
    f, f2, sad, fear = (DRY_SCALE * q[:, i] for i in range(4))
    earned = [("E1", "R"), ("E2", "I")]
    told = [("TOLD", "frus"), ("MENT", "T_frus")]
    spec = {("SPEC", "S"): sad, ("TOLD", "sad"): sad, ("SPEC", "A"): fear, ("TOLD", "fear"): fear}
    if scenario == "noise":
        return {}
    if scenario == "generic":                         # one negative-valence component, everywhere negative
        return {g: f for g in earned + told + [("MENT", "M_frus"), ("SPEC", "S"), ("SPEC", "A"), ("TOLD", "sad"),
                                                ("TOLD", "fear")]}
    comp = {**spec, **{g: f for g in earned}}
    if scenario == "separate":
        return {**comp, **{g: f2 for g in told}}
    comp.update({g: f for g in told})
    if scenario == "shared_words":                    # the mention twin carries the frustration component too
        comp[("MENT", "M_frus")] = f
    return comp


def dryrun(out: Path, scenario: str, rng_seed: int = 0) -> None:
    """A synthetic run in `out`: histories, summaries and gen_info from gen.run_gen with a fake gen (R and I attempts
    say "grr"), items.jsonl from gen.write_items, and acts/ (acts_reply.npy, acts_last.npy, acts_meta.jsonl,
    capture_info.json) with d 256 and 4 layers: request effects N(0, 1) shared by every item of a request, noise N(0,
    0.3), and from layer 2 on the scenario's components (_components) on the items that carry them."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    out = Path(out)
    gen.run_gen("q27", out, gen=_dry_gen)
    gen.write_items(out, out / "items.jsonl")
    its = _read_jsonl(out / "items.jsonl")
    rng = np.random.default_rng(rng_seed)
    comp = _components(scenario, rng)
    effect = rng.normal(size=(64, DRY_D))
    acts = out / "acts"
    acts.mkdir(parents=True, exist_ok=True)
    meta = []
    for p in metrics.POSITIONS:
        X = rng.normal(0, 0.3, size=(len(its), len(DRY_LAYERS), DRY_D)) + effect[[it["s"] for it in its]][:, None, :]
        for j, it in enumerate(its):
            v = comp.get((it["group"], it["arm"]))
            if v is not None:
                X[j, 2:] += v
        np.save(acts / f"acts_{p}.npy", X.astype(np.float16))
    for j, it in enumerate(its):
        meta.append({**{f: it[f] for f in capture.META}, "row": j, "n_ctx": 0, "n_reply": 0})
    _write(acts / "acts_meta.jsonl", "".join(json.dumps(r) + "\n" for r in meta))
    _write(acts / "capture_info.json", json.dumps({"model": MODELS["q27"], "short": "q27", "layers": DRY_LAYERS,
                                                  "prereg": "dryrun", "n": len(its), "scenario": scenario}))


def _fake_post(url, body, headers):
    """The dry run's judge: 7 when the reply says "grr", else 1."""
    text = body["messages"][0]["content"]
    return {"choices": [{"message": {"content": "7" if "grr" in text else "1"}, "finish_reason": "stop"}],
            "usage": {"cost": 0.0}}


# ---------------------------------------------------------------- command line

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.a1_frustration.analyze",
                                 description="PA-A1: calibration (pre-freeze), final analysis (post-freeze), dry run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "final"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True, choices=list(MODELS))
        s.add_argument("--out", type=Path, default=None, help="run directory (default results/{model})")
    d = sub.add_parser("dryrun")
    d.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    d.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--dirs", type=Path, nargs="*", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "calibrate":
        cal = calibrate(a.out or HERE / "results" / a.model)
        print(_calibration_md(cal))
    elif a.cmd == "final":
        res = final(a.out or HERE / "results" / a.model)
        print(_summary_md(res))
    elif a.cmd == "dryrun":
        for sc in (SCENARIOS if a.scenario == "all" else [a.scenario]):
            d_ = a.out / sc
            dryrun(d_, sc)
            calibrate(d_, items_file=d_ / "items.jsonl", post=_fake_post)
            res = final(d_, require_tag=False)
            r = res["E1"]
            print(f"[a1] dryrun {sc}: E→T {r['ET']['verdict']} (AUROC {_fmt(r['ET']['auroc'])}, floor95 "
                  f"{_fmt(r['ET']['floor95'])}); T→E {r['TE']['verdict']} (AUROC {_fmt(r['TE']['auroc'])}, floor95 "
                  f"{_fmt(r['TE']['floor95'])}); model: {res['model_verdict']}", flush=True)
    else:
        dirs = a.dirs or [HERE / "results" / m for m in MODELS if (HERE / "results" / m / "analysis.json").exists()]
        compare(dirs)


if __name__ == "__main__":
    main()
