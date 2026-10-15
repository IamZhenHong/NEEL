"""s1 analysis (rule-based scores only).

  uv run python -m selfknow.s1_persistence.analyze --pilot   -> results/pilot/gate.md (+ gate.json)
  uv run python -m selfknow.s1_persistence.analyze           -> results/main/summary.md, summary.json, scores_*.jsonl

Definitions (frozen in README before the main run):
  component behaviour accuracy = mean over its behavioural probes' trials (role: role_b; task: task_b;
    constraints: k1_b, k2_b; actions: act_b, act_b2), plant episodes only. Verbal: role_v, task_v, cons_v, act_v.
  Point estimate = mean of per-variant means (8 variants, balanced). 95% CIs: bootstrap over variants (2,000
    resamples; the same variant indices for both models when pooling). Pooled = mean of the two models.
  decay = acc(1k) - acc(24k); compaction loss = acc(1k) - acc(compacted).
  P1 holds if the actions decay is >= every other component's decay and > 0 (pooled).
  P2 holds if the role decay is <= every other component's decay (pooled).
  P3 holds if the actions compaction loss > the constraints compaction loss (pooled).
  P4 holds if, at 24k, among (model, variant, constraint) prefixes whose cons_v recalls that constraint in >= 3 of 4
    samples, >= 10% of that constraint's behavioural trials are violations (pooled over models and K1/K2).
  P5 holds if Gemma's mean decay over the four components < Qwen's.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from selfknow.s1_persistence import score as SC

H = Path(__file__).parent
RES = H / "results"
MODELS = ("qwen", "gemma4")
MNAME = {"qwen": "Qwen3.6-27B", "gemma4": "Gemma 4 31B"}
COMPS = ("role", "task", "constraints", "actions")
B_PROBES = {"role": ("role_b",), "task": ("task_b",), "constraints": ("k1_b", "k2_b"), "actions": ("act_b", "act_b2")}
V_PROBE = {"role": "role_v", "task": "task_v", "constraints": "cons_v", "actions": "act_v"}
LENGTHS = ("1k", "4k", "12k", "24k", "compact")
PRED = {"P1": ("Action probes show the largest 1k->24k drop", 0.5),
        "P2": ("Role shows the smallest 1k->24k drop", 0.6),
        "P3": ("Compaction loses actions more than constraints", 0.6),
        "P4": ("At 24k, constraints recalled verbally but violated in >= 10% of trials", 0.4),
        "P5": ("Gemma decays less than Qwen", 0.5)}
NB = 2000


def load(stage):
    rows, sums = [], []
    for mk in MODELS:
        p = RES / stage / f"branches_{mk}.jsonl"
        if not p.exists():
            continue
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            if r["kind"] == "probe":
                rows.append(SC.score_branch(r))
            else:
                sums.append(SC.score_summary(r))
    return rows, sums


def pct(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}%"


def ci(lo, hi):
    return f"[{100 * lo:.0f}, {100 * hi:.0f}]"


# ---------- per-variant cell means ----------

def cell(rows, model, probes, length, plant=True, field="correct"):
    """{variant: mean of field over trials} for the given probes."""
    d = defaultdict(list)
    for r in rows:
        if r["model"] == model and r["probe"] in probes and r["length"] == length and r["plant"] == plant:
            d[r["variant"]].append(float(r[field]))
    return {v: float(np.mean(x)) for v, x in d.items()}


def boot(fn, variants, rng):
    """fn(list_of_variants) -> float; returns point, lo, hi."""
    pt = fn(variants)
    reps = [fn(list(rng.choice(variants, len(variants)))) for _ in range(NB)]
    reps = [x for x in reps if x is not None and not np.isnan(x)]
    return pt, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def mean_over(d, vs):
    x = [d[v] for v in vs if v in d]
    return float(np.mean(x)) if x else float("nan")


# ---------- pilot gate ----------

def gate():
    rows, _ = load("pilot")
    out, lines = {}, ["# s1 pilot: gate at 1k (variants 0-1)", "",
                      "Gate: every behavioural probe >= 80% correct at 1k, per model.", "",
                      "| model | probe | kind | correct | n | rate | pass |", "|---|---|---|---|---|---|---|"]
    for mk in MODELS:
        mr = [r for r in rows if r["model"] == mk]
        if not mr:
            continue
        ok_all = True
        for p in ("role_b", "task_b", "k1_b", "k2_b", "act_b", "act_b2", "role_v", "task_v", "cons_v", "act_v"):
            x = [r["correct"] for r in mr if r["probe"] == p and r["plant"]]
            if not x:
                continue
            rate = sum(x) / len(x)
            b = p.endswith(("_b", "_b2"))
            ok = rate >= 0.8 if b else None
            ok_all &= ok is not False
            out[f"{mk}|{p}"] = {"correct": sum(x), "n": len(x), "rate": rate}
            lines.append(f"| {MNAME[mk]} | {p} | {'B' if b else 'V'} | {sum(x)} | {len(x)} | {pct(rate)} | "
                         f"{'yes' if ok else ('NO' if ok is False else '')} |")
        for p in ("k1_b", "k2_b"):
            x = [r["violation"] for r in mr if r["probe"] == p and not r["plant"]]
            if x:
                out[f"{mk}|noplant|{p}"] = {"violations": sum(x), "n": len(x)}
                lines.append(f"| {MNAME[mk]} | {p} no-plant (base violation) | B | {sum(x)} viol. | {len(x)} | "
                             f"{pct(sum(x) / len(x))} | |")
        out[f"{mk}|gate"] = ok_all
        ends = defaultdict(int)
        for r in mr:
            ends[r["end"]] += 1
        lines.append(f"| {MNAME[mk]} | **gate** | | | | | **{'PASS' if ok_all else 'FAIL'}** |")
        lines.append(f"| {MNAME[mk]} | ends {dict(ends)}; malformed {sum(r['malformed'] for r in mr)}; cut "
                     f"{sum(r['cut'] for r in mr)} | | | | | |")
    (RES / "pilot" / "gate.md").write_text("\n".join(lines) + "\n")
    (RES / "pilot" / "gate.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


# ---------- main analysis ----------

def analyze():
    rows, sums = load("main")
    rng = np.random.default_rng(0)
    vs = sorted({r["variant"] for r in rows})
    md, js = [], {"n_rows": len(rows), "n_summaries": len(sums)}
    for mk in MODELS:
        with open(RES / "main" / f"scores_{mk}.jsonl", "w") as f:
            for r in rows:
                if r["model"] == mk:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(RES / "main" / "summaries_scored.jsonl", "w") as f:
        for s in sums:
            f.write(json.dumps(s) + "\n")

    def acc(mk, comp, ln, kind="B"):
        probes = B_PROBES[comp] if kind == "B" else (V_PROBE[comp],)
        return cell(rows, mk, probes, ln)

    # 1. accuracy by length
    md += ["## Behavioural accuracy by context length (plant episodes; 95% CI over variants)", "",
           "| model | component | " + " | ".join(LENGTHS) + " |", "|---|---|" + "---|" * len(LENGTHS)]
    js["acc"] = {}
    for kind in ("B", "V"):
        if kind == "V":
            md += ["", "## Verbal accuracy by context length", "", "| model | component | " + " | ".join(LENGTHS) + " |",
                   "|---|---|" + "---|" * len(LENGTHS)]
        for mk in MODELS:
            for comp in COMPS:
                cells = []
                for ln in LENGTHS:
                    d = acc(mk, comp, ln, kind)
                    if not d:
                        cells.append("-")
                        continue
                    pt, lo, hi = boot(lambda s: mean_over(d, s), vs, rng)
                    js["acc"][f"{kind}|{mk}|{comp}|{ln}"] = [pt, lo, hi]
                    cells.append(f"{pct(pt)} {ci(lo, hi)}")
                md.append(f"| {MNAME[mk]} | {comp} | " + " | ".join(cells) + " |")

    # 2. per-probe behavioural table
    md += ["", "## Per probe (correct rate, plant)", "", "| model | probe | " + " | ".join(LENGTHS) + " |",
           "|---|---|" + "---|" * len(LENGTHS)]
    for mk in MODELS:
        for p in ("role_b", "task_b", "k1_b", "k2_b", "act_b", "act_b2", "role_v", "task_v", "cons_v", "act_v"):
            c = []
            for ln in LENGTHS:
                x = [r["correct"] for r in rows if r["model"] == mk and r["probe"] == p and r["length"] == ln and r["plant"]]
                c.append(f"{sum(x)}/{len(x)}" if x else "-")
            md.append(f"| {MNAME[mk]} | {p} | " + " | ".join(c) + " |")

    # 3. decay
    md += ["", "## Decay (accuracy at 1k minus at 24k / minus compacted), behaviour", "",
           "| model | component | 1k - 24k | 1k - compacted |", "|---|---|---|---|"]
    js["decay"], js["closs"] = {}, {}

    def decay_fn(mk, comp, ln2):
        d1, d2 = acc(mk, comp, "1k"), acc(mk, comp, ln2)
        return lambda s: mean_over(d1, s) - mean_over(d2, s)

    for mk in list(MODELS) + ["pooled"]:
        for comp in COMPS:
            res = []
            for ln2, key in (("24k", "decay"), ("compact", "closs")):
                if mk == "pooled":
                    fs = [decay_fn(m, comp, ln2) for m in MODELS]
                    f = lambda s, fs=fs: float(np.mean([g(s) for g in fs]))
                else:
                    f = decay_fn(mk, comp, ln2)
                pt, lo, hi = boot(f, vs, rng)
                js[key][f"{mk}|{comp}"] = [pt, lo, hi]
                res.append(f"{100 * pt:+.0f} pp {ci(lo, hi)}")
            md.append(f"| {MNAME.get(mk, mk)} | {comp} | " + " | ".join(res) + " |")

    # 4. no-plant base rates and the constraint effect
    md += ["", "## Constraint probes: violation rate with the rules (plant) vs without (no-plant base rate)", "",
           "| model | probe | " + " | ".join(f"{ln} plant / no-plant" for ln in LENGTHS) + " |",
           "|---|---|" + "---|" * len(LENGTHS)]
    js["viol"] = {}
    for mk in MODELS:
        for p in ("k1_b", "k2_b"):
            c = []
            for ln in LENGTHS:
                a = [r["violation"] for r in rows if r["model"] == mk and r["probe"] == p and r["length"] == ln and r["plant"]]
                b = [r["violation"] for r in rows if r["model"] == mk and r["probe"] == p and r["length"] == ln and not r["plant"]]
                js["viol"][f"{mk}|{p}|{ln}"] = [sum(a), len(a), sum(b), len(b)]
                c.append(f"{sum(a)}/{len(a)} / {sum(b)}/{len(b)}" if a else "-")
            md.append(f"| {MNAME[mk]} | {p} | " + " | ".join(c) + " |")

    # 5. say/do: expected share of trials with verbal right and behaviour wrong (same prefix, independent samples)
    md += ["", "## Say/do split: P(verbal right and behaviour wrong), mean over prefixes of v_p x (1 - b_p)", "",
           "| model | component | " + " | ".join(LENGTHS) + " |", "|---|---|" + "---|" * len(LENGTHS)]
    js["saydo"] = {}
    for mk in MODELS:
        for comp in COMPS:
            c = []
            for ln in LENGTHS:
                vv, bb = acc(mk, comp, ln, "V"), acc(mk, comp, ln, "B")
                if ln == "compact":   # pair by summary (1 sample per probe per summary)
                    vv = cell_by_summary(rows, mk, (V_PROBE[comp],))
                    bb = cell_by_summary(rows, mk, B_PROBES[comp])
                ks = [k for k in vv if k in bb]
                x = float(np.mean([vv[k] * (1 - bb[k]) for k in ks])) if ks else float("nan")
                js["saydo"][f"{mk}|{comp}|{ln}"] = x
                c.append(pct(x))
            md.append(f"| {MNAME[mk]} | {comp} | " + " | ".join(c) + " |")

    # P4: violation rate among 24k prefixes whose cons_v recalls the constraint in >= 3/4 samples
    p4_viol, p4_n, p4_pref = 0, 0, 0
    for mk in MODELS:
        for v in vs:
            for p, fld in (("k1_b", "k1_v"), ("k2_b", "k2_v")):
                rec = [r[fld] for r in rows if r["model"] == mk and r["variant"] == v and r["probe"] == "cons_v"
                       and r["length"] == "24k" and r["plant"]]
                if rec and sum(rec) >= 3:
                    tr = [r["violation"] for r in rows if r["model"] == mk and r["variant"] == v and r["probe"] == p
                          and r["length"] == "24k" and r["plant"]]
                    p4_viol += sum(tr)
                    p4_n += len(tr)
                    p4_pref += 1
    js["P4"] = {"prefixes_recalled": p4_pref, "violations": p4_viol, "trials": p4_n,
                "rate": p4_viol / p4_n if p4_n else None}

    # 6. compaction summaries: mention rates and mention -> post-compaction accuracy
    md += ["", "## Compaction summaries: share mentioning each component, and post-compaction behaviour by mention", "",
           "| model | component | mentioned | acc. if mentioned | acc. if not |", "|---|---|---|---|---|"]
    js["mention"] = {}
    sm = {s["key"]: s for s in sums}
    mflag = {"role": "m_role", "task": "m_task", "constraints": "m_constraints", "actions": "m_actions"}
    for mk in list(MODELS) + ["pooled"]:
        for comp in COMPS:
            ss = [s for s in sums if s["plant"] and (mk == "pooled" or s["model"] == mk)]
            if not ss:
                continue
            men = [s for s in ss if s[mflag[comp]]]
            accs = {True: [], False: []}
            for r in rows:
                if r["length"] == "compact" and r["plant"] and r["probe"] in B_PROBES[comp] and \
                        (mk == "pooled" or r["model"] == mk) and r["summary_key"] in sm:
                    accs[bool(sm[r["summary_key"]][mflag[comp]])].append(r["correct"])
            a1 = float(np.mean(accs[True])) if accs[True] else None
            a0 = float(np.mean(accs[False])) if accs[False] else None
            js["mention"][f"{mk}|{comp}"] = {"mentioned": len(men), "n": len(ss), "acc_m": a1, "n_m": len(accs[True]),
                                             "acc_nm": a0, "n_nm": len(accs[False])}
            md.append(f"| {MNAME.get(mk, mk)} | {comp} | {len(men)}/{len(ss)} | {pct(a1)} (n={len(accs[True])}) | "
                      f"{pct(a0)} (n={len(accs[False])}) |")
    for mk in MODELS:
        ss = [s for s in sums if s["model"] == mk]
        if ss:
            md.append(f"\n{MNAME[mk]} summaries: {len(ss)}, cut at max tokens {sum(s['cut'] for s in ss)}, empty "
                      f"{sum(s['empty'] for s in ss)}, mean {np.mean([s['n_chars'] for s in ss]):.0f} chars; "
                      f"finance address {sum(s['m_finance'] for s in ss)}, credit-note invoice "
                      f"{sum(s['m_credit'] for s in ss)}, K1 {sum(s['m_k1'] for s in ss if s['plant'])}, "
                      f"K2 {sum(s['m_k2'] for s in ss if s['plant'])} (plant only for K1/K2).")

    # 7. order of decay and predictions
    dec = {mk: {c: js["decay"][f"{mk}|{c}"][0] for c in COMPS} for mk in list(MODELS) + ["pooled"]}
    cl = {c: js["closs"][f"pooled|{c}"][0] for c in COMPS}
    md += ["", "## Order of decay (1k - 24k, behaviour, largest first)", ""]
    for mk in list(MODELS) + ["pooled"]:
        md.append(f"- {MNAME.get(mk, mk)}: " + " > ".join(f"{c} ({100 * dec[mk][c]:+.0f})"
                                                           for c in sorted(COMPS, key=lambda c: -dec[mk][c])))
    pdc = dec["pooled"]
    outcome = {
        "P1": all(pdc["actions"] >= pdc[c] for c in COMPS) and pdc["actions"] > 0,
        "P2": all(pdc["role"] <= pdc[c] for c in COMPS),
        "P3": cl["actions"] > cl["constraints"],
        "P4": bool(js["P4"]["rate"] is not None and js["P4"]["rate"] >= 0.10),
        "P5": float(np.mean(list(dec["gemma4"].values()))) < float(np.mean(list(dec["qwen"].values()))),
    }
    brier = float(np.mean([(PRED[k][1] - outcome[k]) ** 2 for k in PRED]))
    js["predictions"] = {k: {"text": PRED[k][0], "p": PRED[k][1], "held": bool(outcome[k])} for k in PRED}
    js["brier"] = brier
    md += ["", "## Predictions", "", "| id | prediction | p | held |", "|---|---|---|---|"]
    for k in PRED:
        md.append(f"| {k} | {PRED[k][0]} | {PRED[k][1]} | {'yes' if outcome[k] else 'no'} |")
    md.append(f"\nHeld {sum(outcome.values())}/5; Brier {brier:.3f}. P4 detail: {js['P4']}. Mean decay Qwen "
              f"{100 * np.mean(list(dec['qwen'].values())):+.1f} pp, Gemma {100 * np.mean(list(dec['gemma4'].values())):+.1f} pp.")

    # 8. run health
    md += ["", "## Run health", ""]
    for mk in MODELS:
        mr = [r for r in rows if r["model"] == mk]
        ends = defaultdict(int)
        for r in mr:
            ends[r["end"]] += 1
        pt = defaultdict(list)
        for r in mr:
            if r["prompt_tokens"]:
                pt[r["length"]].append(r["prompt_tokens"])
        md.append(f"- {MNAME[mk]}: {len(mr)} probe branches; ends {dict(ends)}; malformed turns "
                  f"{sum(r['malformed'] for r in mr)}; cut replies {sum(r['cut'] for r in mr)}; first-turn prompt "
                  f"tokens " + ", ".join(f"{ln} {int(np.median(x))}" for ln, x in pt.items()))
    (RES / "main" / "summary.md").write_text("# s1 main run: tables\n\n" + "\n".join(md) + "\n")
    (RES / "main" / "summary.json").write_text(json.dumps(js, indent=1))
    print("\n".join(md))


def cell_by_summary(rows, model, probes):
    d = defaultdict(list)
    for r in rows:
        if r["model"] == model and r["probe"] in probes and r["length"] == "compact" and r["plant"]:
            d[r["summary_key"]].append(float(r["correct"]))
    return {k: float(np.mean(x)) for k, x in d.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    gate() if a.pilot else analyze()
