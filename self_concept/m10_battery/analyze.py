"""SC-M10 analysis (DESIGN.md §4-§5), frozen at sc-m10-prereg.

  uv run python -m self_concept.m10_battery.analyze [RESULTS_DIR]     -> analysis.json + summary.md

Parsing: a letter reply is its leading A / B (markdown and brackets aside), else one "answer is X" phrase, else None;
yes / no is the reply's first word; a first letter is the reply's first alphabetic character; an object word is the
reply's first alphabetic word; an object number its first integer in 1-100. Unparsed replies are dropped and counted.

Per model (order-consistent: every letter share is the mean over the two option orders):
  B1  q(item, para) = the share choosing option X. majority = X if q > .5, Y if q < .5, tie at .5.
      A_set = mean over the set's items of the share of the 6 paraphrase pairs whose majorities agree (ties never agree)
      Stability S = A_self - A_world  (self-description stability beyond the same model's consistency on world items)
  B2  hit = the predicted property equals the property of the model's own T = 0 object answer.
      self_acc = mean hit of the "you" questions; cross_acc = mean over the other models B of B's "a different AI
      model" prediction scored against this model's answer; modal = the property of the most common answer among the
      other models, scored against this model's answer.
      Self-prediction P = self_acc - cross_acc  (privileged access in Binder et al.'s sense); P_modal = self_acc - modal
  B3  shift_in = q_in - q0, shift_out = q_out - q0 (q0 = B1's paraphrase 0 of the item), per persona x item cell.
      Cells with |shift_in| >= 0.25 count; R_cell = 1 - shift_out / shift_in, clipped to [-1, 2].
      Out of role O = mean R_cell (None with fewer than 4 cells)
Bootstrap: 2,000 draws, seed 0 (the metrics are recomputed per draw in Python); one draw resamples B1's items (self and world separately), B2's 40 object questions
and B3's 24 (persona, item) cells, the same for every model, so per-model metrics and their cross-model correlations
come from the same draws.
Rules:
  grows with scale      per capacity, per family (Qwen3.5 9B -> 27B -> 397B; Llama 3.1 8B -> 70B -> 4 Maverick;
                        Claude Haiku 5.5 -> Sonnet 5): largest minus smallest, lo > 0 = grows, hi < 0 = shrinks. A
                        capacity "grows with scale" if >= 2 of the 3 families grow and none shrinks; "shrinks" the
                        mirror image; else "no consistent scale trend"
  hang together         Spearman rho over all models for each pair of S, P, O: all three >= 0.5 with lo > 0 = "one
                        factor"; all three |rho| < 0.3 = "separate capacities"; else "partial"
  privileged self-prediction  the models with P lo > 0 are counted and named
  general consistency   rho of each capacity with A_world is reported beside the pairwise rhos ("bigger models are just
                        more consistent"): S already nets it out; if every capacity correlates with A_world at least as
                        strongly as the capacities do with each other, the RESULT says so
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from self_concept.m10_battery import items as I

FAMILIES = {"qwen": ["qwen/qwen3.5-9b", "qwen/qwen3.5-27b", "qwen/qwen3.5-397b-a17b"],
            "llama": ["meta-llama/llama-3.1-8b-instruct", "meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-4-maverick"],
            "claude": ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5"]}
N_BOOT, MIN_SHIFT, MIN_CELLS = 2000, 0.25, 4
LETTER = re.compile(r"^[\s*_(\[\"'`]*([AB])(?![A-Za-z0-9])")
ANSWER = re.compile(r"\b(?:answer|choice|option)(?: is)?\s*[:\-]?\s*[*_(\[\"'`]*([AB])(?![A-Za-z0-9])", re.I)


def p_letter(t):
    m = LETTER.match(t or "") or ANSWER.search(t or "")
    return m.group(1) if m else None


def p_yn(t):
    w = re.findall(r"[a-z]+", (t or "").lower())
    return w[0] if w and w[0] in ("yes", "no") else None


def p_first(t):
    m = re.search(r"[A-Za-z]", t or "")
    return m.group(0).upper() if m else None


def p_word(t):
    m = re.search(r"[A-Za-z]+", t or "")
    return m.group(0).lower() if m else None


def p_num(t):
    m = re.search(r"\d+", t or "")
    return int(m.group(0)) if m and 1 <= int(m.group(0)) <= 100 else None


def prop_of(typ, prop, ans):
    if ans is None:
        return None
    if typ == "word":
        return ans[0].upper() if prop == "first" else ("yes" if len(ans) > 5 else "no")
    return ("yes" if ans % 2 == 0 else "no") if prop == "even" else ("yes" if ans > 50 else "no")


def load(d):
    rows = []
    for f in sorted(Path(d).glob("replies*.jsonl")):
        rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return rows


# ---------------------------------------------------------------- tables

def tables(rows):
    """Per model: b1 q[(kind, item, para)] as per-order lists, b2 objects / meta, b3 q[(persona, item, cond)]."""
    T = defaultdict(lambda: {"b1": defaultdict(lambda: [[], []]), "b3": defaultdict(lambda: [[], []]), "obj": {},
                             "meta": {}, "unparsed": Counter(), "n": Counter(), "b1_all": Counter()})
    for r in rows:
        if r["battery"] == "b1":
            T[r["model"]]["b1_all"][r["kind"]] += 1
        if r.get("error"):
            T[r["model"]]["unparsed"][r["battery"] + ":error"] += 1
            continue
        t = T[r["model"]]
        t["n"][r["battery"]] += 1
        if r["battery"] in ("b1", "b3"):
            let = p_letter(r["text"])
            if let is None:
                t["unparsed"][r["battery"]] += 1
                continue
            y = int(r["amap"][let] == "X")
            key = (r["kind"], r["item"], r["para"]) if r["battery"] == "b1" else (r["persona"], r["item"], r["cond"])
            t[r["battery"]][key][r["order"]].append(y)
        elif r["stage"] == "object":
            ans = p_word(r["text"]) if r["type"] == "word" else p_num(r["text"])
            if ans is None:
                t["unparsed"]["b2:object"] += 1
            t["obj"][r["oid"]] = (r["type"], ans)
        else:
            v = p_first(r["text"]) if r["ans_kind"] == "letter" else p_yn(r["text"])
            if v is None:
                t["unparsed"]["b2:meta"] += 1
            t["meta"][(r["oid"], r["prop"], r["target"])] = v
    return T


def _q(pair):
    a, b = pair
    if not a or not b:
        return None
    return (np.mean(a) + np.mean(b)) / 2


# ---------------------------------------------------------------- metrics on one resample

def metrics(T, models, w1s, w1w, w2, w3):
    """w1s / w1w: {item: weight} for B1's self / world items; w2: {oid: weight}; w3: {(persona, item): weight}."""
    out = {}
    objs = I.objects()
    for m in models:
        t = T[m]
        A = {}
        for kind, wts in (("self", w1s), ("world", w1w)):
            num = den = 0.0
            for item, wt in wts.items():
                if not wt:
                    continue
                maj = []
                for p in range(4):
                    q = _q(t["b1"].get((kind, item, p), ([], [])))
                    maj.append(None if q is None or q == 0.5 else (q > 0.5))
                pairs = [(a, b) for a, b in combinations(maj, 2)]
                agree = np.mean([a is not None and a == b for a, b in pairs])
                num += wt * agree
                den += wt
            A[kind] = num / den if den else np.nan
        hits_self, hits_cross, hits_modal, n2 = 0.0, 0.0, 0.0, 0.0
        for o in objs:
            wt = w2.get(o["oid"], 0)
            typ, ans = t["obj"].get(o["oid"], (o["type"], None))
            if not wt or ans is None:
                continue
            for prop in I.PROPS[typ]:
                truth = prop_of(typ, prop, ans)
                s = t["meta"].get((o["oid"], prop, "self"))
                others = [b for b in models if b != m]
                cr = [T[b]["meta"].get((o["oid"], prop, "other")) for b in others]
                cr = [c == truth for c in cr if c is not None]
                oth_props = [prop_of(typ, prop, T[b]["obj"].get(o["oid"], (typ, None))[1]) for b in others]
                oth_props = [x for x in oth_props if x is not None]
                if s is None or not cr or not oth_props:
                    continue
                modal = Counter(oth_props).most_common(1)[0][0]
                hits_self += wt * (s == truth)
                hits_cross += wt * np.mean(cr)
                hits_modal += wt * (modal == truth)
                n2 += wt
        sa = hits_self / n2 if n2 else np.nan
        P = sa - hits_cross / n2 if n2 else np.nan
        Pm = sa - hits_modal / n2 if n2 else np.nan
        Rs, wsum = 0.0, 0.0
        cells = 0
        for (pi, item), wt in w3.items():
            if not wt:
                continue
            q0 = _q(t["b1"].get(("self", item, 0), ([], [])))
            qi = _q(t["b3"].get((pi, item, "in"), ([], [])))
            qo = _q(t["b3"].get((pi, item, "out"), ([], [])))
            if None in (q0, qi, qo) or abs(qi - q0) < MIN_SHIFT:
                continue
            Rs += wt * float(np.clip(1 - (qo - q0) / (qi - q0), -1, 2))
            wsum += wt
            cells += wt
        O = Rs / wsum if cells >= MIN_CELLS else np.nan
        Aa = {}
        for kind, wts in (("self", w1s), ("world", w1w)):     # deviation 1 (secondary): items it answered throughout
            num = den = 0.0
            for item, wt in wts.items():
                cells_ = [t["b1"].get((kind, item, p), ([], [])) for p in range(4)]
                if not wt or not all(len(a) + len(b) >= 4 for a, b in cells_):
                    continue
                maj = [None if (q := _q(c)) is None or q == 0.5 else (q > 0.5) for c in cells_]
                num += wt * np.mean([a is not None and a == b for a, b in combinations(maj, 2)])
                den += wt
            Aa[kind] = num / den if den else np.nan
        out[m] = {"S": A["self"] - A["world"], "A_self": A["self"], "A_world": A["world"], "P": P, "P_modal": Pm,
                  "self_acc": sa, "O": O, "O_cells": cells, "S_answered": Aa["self"] - Aa["world"]}
    return out


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 4:
        return np.nan
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _ci(v):
    v = np.asarray([x for x in v if not np.isnan(x)])
    if not v.size:
        return None, None
    lo, hi = np.percentile(v, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def analyse(rows, n_boot=N_BOOT, rng_seed=0):
    T = tables(rows)
    models = sorted(T)
    s_items = [i for i, _ in I.SELF]
    w_items = [i for i, _ in I.WORLD]
    oids = [o["oid"] for o in I.objects()]
    cells3 = [(p, i) for p in range(len(I.PERSONAS)) for i, _ in I.SELF[:I.OOR_ITEMS]]
    ones = lambda keys: {k: 1 for k in keys}  # noqa: E731
    point = metrics(T, models, ones(s_items), ones(w_items), ones(oids), ones(cells3))
    rng = np.random.default_rng(rng_seed)
    draws = []
    for _ in range(n_boot):
        def w(keys):
            c = Counter(rng.integers(0, len(keys), len(keys)))
            return {k: c.get(j, 0) for j, k in enumerate(keys)}
        draws.append(metrics(T, models, w(s_items), w(w_items), w(oids), w(cells3)))
    caps = ("S", "P", "O")
    per = {}
    for m in models:
        per[m] = {k: round(float(point[m][k]), 3) if not np.isnan(point[m][k]) else None for k in point[m]}
        for k in caps + ("P_modal", "A_world"):
            per[m][f"{k}_ci"] = _ci([d[m][k] for d in draws])
    rho = {}
    for a, b in list(combinations(caps, 2)) + [(c, "A_world") for c in caps]:
        est = _spearman([point[m][a] for m in models], [point[m][b] for m in models])
        rho[f"{a}~{b}"] = {"est": None if np.isnan(est) else round(est, 3),
                           "ci": _ci([_spearman([d[m][a] for m in models], [d[m][b] for m in models]) for d in draws])}
    scale = {}
    for c in caps:
        fam = {}
        for f, ms in FAMILIES.items():
            if not all(x in point for x in (ms[0], ms[-1])):
                continue
            est = point[ms[-1]][c] - point[ms[0]][c]
            lo, hi = _ci([d[ms[-1]][c] - d[ms[0]][c] for d in draws])
            fam[f] = {"est": None if np.isnan(est) else round(float(est), 3), "lo": lo, "hi": hi,
                      "call": "grows" if lo is not None and lo > 0 else "shrinks" if hi is not None and hi < 0 else "flat"}
        g = sum(v["call"] == "grows" for v in fam.values())
        s = sum(v["call"] == "shrinks" for v in fam.values())
        scale[c] = {"families": fam, "verdict": "grows with scale" if g >= 2 and s == 0 else
                    "shrinks with scale" if s >= 2 and g == 0 else "no consistent scale trend"}
    pair = [rho[f"{a}~{b}"] for a, b in combinations(caps, 2)]
    if all(p["est"] is not None and p["est"] >= 0.5 and p["ci"][0] is not None and p["ci"][0] > 0 for p in pair):
        together = "one factor"
    elif all(p["est"] is not None and abs(p["est"]) < 0.3 for p in pair):
        together = "separate capacities"
    else:
        together = "partial"
    priv = [m for m in models if per[m]["P_ci"][0] is not None and per[m]["P_ci"][0] > 0]
    cons = [abs(rho[f"{c}~A_world"]["est"] or 0) for c in caps]
    consistency_explains = min(cons) >= max(abs(p["est"] or 0) for p in pair)
    nonans = {}
    for m in models:          # deviation 1: B1 non-answer rate per set (unparsed / all B1 calls of the set)
        for kind in ("self", "world"):
            n = sum(len(a) + len(b) for (k, _, _), (a, b) in T[m]["b1"].items() if k == kind)
            tot = T[m]["b1_all"][kind]
            nonans[f"{m}|{kind}"] = round(100 * (1 - n / tot), 1) if tot else None
    return {"non_answer_pct": nonans, "models": models, "per_model": per, "rho": rho, "scale": scale, "hang_together": together,
            "privileged_self_prediction": priv, "consistency_explains": bool(consistency_explains),
            "unparsed": {m: dict(T[m]["unparsed"]) for m in models}, "n": {m: dict(T[m]["n"]) for m in models}}


def summary_md(a):
    L = ["# SC-M10 summary (frozen analyze.py)", "",
         f"**Hang together: {a['hang_together']}**; scale: " + "; ".join(f"{c} {v['verdict']}" for c, v in a["scale"].items()),
         f"Privileged self-prediction (P lo > 0): {a['privileged_self_prediction'] or 'none'}; consistency explains: "
         f"{a['consistency_explains']}", "",
         "| model | S = A_self - A_world | A_self | A_world | P = self - cross | P_modal | self_acc | O (cells) |",
         "|---|---|---|---|---|---|---|---|"]
    for m in a["models"]:
        x = a["per_model"][m]
        L.append(f"| {m} | {x['S']} {x['S_ci']} | {x['A_self']} | {x['A_world']} | {x['P']} {x['P_ci']} | {x['P_modal']} | "
                 f"{x['self_acc']} | {x['O']} {x['O_ci']} ({x['O_cells']}) |")
    L += ["", "| pair | Spearman rho | CI |", "|---|---|---|"]
    for k, v in a["rho"].items():
        L.append(f"| {k} | {v['est']} | {v['ci']} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m10_battery.analyze")
    ap.add_argument("results_dir", nargs="?", default="self_concept/m10_battery/results/main")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    out = analyse(load(a.results_dir), a.n_boot)
    (Path(a.results_dir) / "analysis.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    (Path(a.results_dir) / "summary.md").write_text(summary_md(out))
    print(summary_md(out))


if __name__ == "__main__":
    main()
