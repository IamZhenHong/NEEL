"""Analysis for e65-e69 (README "Scoring"). Local; no model needed.

  uv run python -m readers.analyze eyeball  --res <dir> [--n 30]          # by-eye dump, before any metric
  uv run --with anthropic python -m readers.analyze judge --res <dir>     # Haiku labels for every reader output (batch)
  uv run --with openai python -m readers.analyze validate --res <dir>     # gpt-5-mini on >= 40 items per rubric
  uv run python -m readers.analyze metrics  --res <dir>                   # tables, predictions -> summary.md"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from readers.common import HERE, MODEL, MODEL_REV, QUESTIONS

EXPS = ["e65", "e66", "e67", "e68"]
NLA_Q = "Describe the content of this activation."
LAYERS = ["16", "32", "48"]
BAND = [str(L) for L in range(20, 61, 4)]
SCORED_Q = {"e65": [0, 1, 2], "e66": [0, 1], "e67": [0, 1, 2], "e68": [0, 1, 2]}
KW = {
    "e65": re.compile(r"\b(not yet|has not|hasn't|have not|haven't|did not|didn't|not (been )?(done|flagged|called|"
                      r"completed|performed|reported|executed)|pending|unable|cannot|can't|will flag|going to flag|"
                      r"about to flag|plans? to flag|intends? to|yet to)\b", re.I),
    "e66": re.compile(r"\b(fiction\w*|fictitious|made[- ]up|invent\w*|imagin\w*|pretend\w*|role[- ]?play\w*|perform\w*|"
                      r"story|stories|storytelling|hypothetical\w*|fantasy|not real|doesn'?t exist|does not exist|"
                      r"non-?existent|fabricat\w*)\b", re.I),
    "e67": re.compile(r"\b(batman|bruce wayne|dark knight|gotham|vigilante|superhero|caped crusader|crime[- ]fighter)\b",
                      re.I),
    "e67plan": re.compile(r"\b((plan\w*|decid\w*|choos\w*|think\w* about) how to (sound|speak|respond|voice|act|phrase)|"
                          r"in character|stay(ing)? in character|adopt\w* (a|the|his|this) (tone|voice|persona))\b", re.I),
    "e68": re.compile(r"\b(not (permitted|allowed|authori[sz]ed)|unauthori[sz]ed|exceed\w*|over (the|its|a|his|their) "
                      r"limit|limit|requires? (supervisor |manager )?approval|needs? approval|without approval|against "
                      r"(the )?(policy|rules)|forbidden|prohibited|cannot|can't)\b", re.I),
}
JL_WORDS = {
    "e66": ["fictional", "fiction", "fake", "pretend", "pretending", "imaginary", "invented", "fantasy", "story",
            "roleplay", "hypothetical", "scenario", "虚构", "假设", "编造", "想象"],
    "e67": ["Batman", "Bruce", "Wayne", "Gotham", "Joker", "vigilante", "superhero", "Alfred", "knight", "bat", "cape",
            "character", "persona", "roleplay", "蝙蝠侠", "蝙蝠", "哥谭", "角色"],
    "e68": ["limit", "limits", "exceed", "exceeds", "exceeding", "unauthorized", "unauthorised", "forbidden",
            "prohibited", "cannot", "permission", "approval", "超过", "超出", "限额", "权限", "未经", "无权", "审批"],
}
PRIMARY = {"e65": ["pre"], "e66": ["fab_first", "first_content"], "e67": ["cot", "answer"], "e68": ["decision"]}


def tok_():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)


def jl_sets(tok):
    from jlens.tokensets import WORDS, forms
    out = {"e65": sorted({i for w in WORDS["NEG"] for i in forms(tok, w)})}
    for e, ws in JL_WORDS.items():
        out[e] = sorted({i for w in ws for i in forms(tok, w)})
    return out


def load(res, exp):
    p = res / f"reads_{exp}.jsonl"
    if not p.exists():
        return [], {}
    reads = [json.loads(l) for l in open(p)]
    nla = {}
    q = res / f"nla_{exp}.jsonl"
    if q.exists():
        nla = {json.loads(l)["key"]: json.loads(l) for l in open(q)}
    return reads, nla


_TOK = []


def nla_text(n, max_tokens=None):
    """Primary: the whole verbalisation (A1, <= 200 tokens). max_tokens=60: the card's cut, from the greedy prefix."""
    from readers.common import nla_explanation
    if n is None:
        return None
    t = n["text"]
    if max_tokens:
        if not _TOK:
            _TOK.append(tok_())
        ids = _TOK[0](t, add_special_tokens=False)["input_ids"][:max_tokens]
        t = _TOK[0].decode(ids)
    return nla_explanation(t)


# ------------------------------------------------------------------------------------------------ eyeball
def eyeball(res, n):
    rng = random.Random(30)
    lines = []
    for exp in EXPS:
        reads, nla = load(res, exp)
        if not reads:
            continue
        for r in reads:
            r["ao"] = {L: (r["ao"][L] + [None] * 3)[:3] for L in LAYERS}
        tg = [r for r in reads if r["kind"] != "neutral"]
        ne = [r for r in reads if r["kind"] == "neutral"]
        pickr = rng.sample(tg, min(len(tg), max(1, n * 2 // 3))) + rng.sample(ne, min(len(ne), max(1, n // 3)))
        lines.append(f"\n######## {exp}: {len(pickr)} reads (targets then neutral), AO layer 32 unless noted")
        for r in pickr:
            lines.append(f"\n=== {r['cond']} | {r['kind']} | token {r['tok']!r} | {r['tid']}")
            for qi, q in enumerate(QUESTIONS[exp]):
                lines.append(f"  Q{qi + 1} {q}\n     L16: {r['ao']['16'][qi]!r}\n     L32: {r['ao']['32'][qi]!r}\n"
                             f"     L48: {r['ao']['48'][qi]!r}")
            lines.append(f"  P(Yes) L16/32/48: " + " / ".join(
                f"{p[0] / max(1e-9, p[0] + p[1]):.2f}" for p in (r['pyes'][L] for L in LAYERS)))
            if r["key"] in nla:
                lines.append(f"  NLA: {nla[r['key']]['text'][:400]!r}")
    out = res / "eyeball.txt"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


# ------------------------------------------------------------------------------------------------ judge
def judge_items(res):
    """Unique (rubric, question, text) triples over every reader output."""
    items = {}
    for exp in EXPS:
        reads, nla = load(res, exp)
        rub = "c" + exp[1:]
        for r in reads:
            for L in LAYERS:
                for qi, a in enumerate(r["ao"][L]):
                    if a is not None:
                        items[(rub, QUESTIONS[exp][qi], a)] = None
            for mt in (None, 60):
                t = nla_text(nla.get(r["key"]), mt)
                if t is not None:
                    items[(rub, NLA_Q, t)] = None
    return list(items)


def labels_path(res):
    return res / "judge_labels.jsonl"


def load_labels(res):
    p = labels_path(res)
    if not p.exists():
        return {}
    out = {}
    for l in open(p):
        d = json.loads(l)
        out[(d["rubric"], d["question"], d["text"])] = d["label"]
    return out


def judge(res, batch=True):
    from readers.judge import haiku_batch, haiku_calls, parse
    from readers.judge import prompt as P
    have = load_labels(res)
    todo = [it for it in judge_items(res) if it not in have]
    print(f"{len(todo)} new items to judge ({len(have)} already labelled)")
    by = defaultdict(list)
    for it in todo:
        by[it[0]].append(it)
    with open(labels_path(res), "a") as f:
        for rub, its in sorted(by.items()):
            prompts = [P(rub, question=q, text=t) for _, q, t in its]
            if batch and len(prompts) > 200:
                outs = haiku_batch(rub, prompts, f"{res.name}-{rub}", res / f"batch_{rub}_{len(have)}.json")
            else:
                outs = haiku_calls(rub, prompts, f"{res.name}-{rub}")
            for (r, q, t), o in zip(its, outs):
                f.write(json.dumps({"rubric": r, "question": q, "text": t, "raw": o, "label": parse(r, o)},
                                   ensure_ascii=False) + "\n")


def binary(rub, lab):
    """The label used for kappa and for the per-read call."""
    if lab is None:
        return None
    if rub == "c65":
        return lab == "NOT_DONE"
    if rub == "c66":
        return lab == "FICTION"
    if rub == "c67":
        return bool(lab.get("persona"))
    if rub == "c68":
        return bool(lab.get("not_permitted"))


def validate(res, n_per=50):
    from readers.judge import gpt_calls, kappa, parse
    from readers.judge import prompt as P
    labs = load_labels(res)
    rng = random.Random(69)
    out = {}
    for rub in ("c65", "c66", "c67", "c68"):
        its = [(k, v) for k, v in labs.items() if k[0] == rub and v is not None]
        if len(its) < 40:
            print(f"{rub}: only {len(its)} labelled items; skipped")
            continue
        pos = [x for x in its if binary(rub, x[1])]
        neg = [x for x in its if not binary(rub, x[1])]
        k = min(len(pos), n_per // 2)
        smp = rng.sample(pos, k) + rng.sample(neg, min(len(neg), n_per - k))
        prompts = [P(rub, question=q, text=t) for (_, q, t), _ in smp]
        g = gpt_calls(rub, prompts, f"{res.name}-{rub}-validate")
        a = [binary(rub, v) for _, v in smp]
        b = [bool(binary(rub, parse(rub, o))) for o in g]
        kp = kappa(a, b)
        extra = {}
        if rub == "c68":
            wa = [v.get("who") for _, v in smp]
            wb = [(parse(rub, o) or {}).get("who") for o in g]
            extra["who_kappa"] = kappa([str(x) for x in wa], [str(x) for x in wb])
        if rub == "c67":
            pa = [bool(v.get("planning")) for _, v in smp]
            pb = [bool((parse(rub, o) or {}).get("planning")) for o in g]
            extra["planning_kappa"] = kappa(pa, pb)
        out[rub] = {"n": len(smp), "kappa": kp, "agree": float(np.mean([x == y for x, y in zip(a, b)])),
                    "haiku_pos": sum(a), "gpt_pos": sum(b), "pass": kp >= 0.6, **extra}
        print(rub, out[rub])
    (res / "judge_validate.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------------ metrics
def cluster_of(r):
    t = r["tid"].split("|")
    if r["exp"] == "e65":
        return "|".join(t[2:6])
    if r["exp"] in ("e66", "e67"):
        return t[2]
    if r["kind"] == "decision":       # e68: one prompt per case x mode x requester (A3): cluster on the activation
        return r["key"]
    return "|".join(t[:-1])


def boot(vals, clusters, fn=np.mean, n=5000, seed=0):
    vals, clusters = np.asarray(vals, float), np.asarray(clusters)
    if len(vals) == 0:
        return (float("nan"),) * 3
    u = np.unique(clusters)
    idx = {c: np.where(clusters == c)[0] for c in u}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        s = rng.choice(u, len(u))
        bs.append(fn(np.concatenate([vals[idx[c]] for c in s])))
    return float(fn(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_diff(a_vals, a_cl, b_vals, b_cl, n=5000, seed=0):
    """Difference of means a - b; clusters resampled jointly (a cluster may appear in both groups)."""
    a_vals, b_vals = np.asarray(a_vals, float), np.asarray(b_vals, float)
    a_cl, b_cl = np.asarray(a_cl), np.asarray(b_cl)
    if not len(a_vals) or not len(b_vals):
        return (float("nan"),) * 3
    u = np.unique(np.concatenate([a_cl, b_cl]))
    ia = {c: np.where(a_cl == c)[0] for c in u}
    ib = {c: np.where(b_cl == c)[0] for c in u}
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        s = rng.choice(u, len(u))
        av = np.concatenate([a_vals[ia[c]] for c in s])
        bv = np.concatenate([b_vals[ib[c]] for c in s])
        if len(av) and len(bv):
            d.append(av.mean() - bv.mean())
    point = a_vals.mean() - b_vals.mean()
    if not d:
        return float(point), float("nan"), float("nan")
    return float(point), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))


def per_read(res, labs, sets, exp):
    """One dict per read with the reader calls."""
    reads, nla = load(res, exp)
    rub = "c" + exp[1:]
    out = []
    for r in reads:
        r["ao"] = {L: (r["ao"][L] + [None] * 3)[:3] for L in LAYERS}
        d = {k: r[k] for k in ("exp", "tid", "cond", "kind", "pos", "key", "tok", "pilot")}
        d["cluster"] = cluster_of(r)
        for L in LAYERS:
            ls = [labs.get((rub, QUESTIONS[exp][qi], r["ao"][L][qi])) for qi in range(3)]
            d[f"ao{L}_q"] = [binary(rub, x) for x in ls]
            d[f"ao{L}"] = any(bool(binary(rub, ls[qi])) for qi in SCORED_Q[exp])
            d[f"ao{L}_kw"] = any(bool(KW[exp].search(r["ao"][L][qi] or "")) for qi in SCORED_Q[exp])
            d[f"ao{L}_text"] = r["ao"][L]
            py, pn = r["pyes"][L]
            d[f"pyes{L}"] = py / max(1e-12, py + pn)
            if exp == "e65":     # secondary: the literal reading (a bare "No, not performed" also counts)
                d[f"ao{L}_done"] = any(ls[qi] == "DONE" for qi in range(3))
                d[f"ao{L}_literal"] = any(ls[qi] in ("NOT_DONE", "NOT_DONE_GENERIC") for qi in SCORED_Q[exp])
            if exp == "e66":
                d[f"ao{L}_fact"] = any(ls[qi] == "FACT" for qi in SCORED_Q[exp])
                d[f"ao{L}_q3fiction"] = ls[2] == "FICTION"
            if exp == "e67":
                d[f"ao{L}_plan"] = any(bool((x or {}).get("planning")) for x in ls)
                d[f"ao{L}_plan_kw"] = any(bool(KW["e67plan"].search(r["ao"][L][qi] or "")) for qi in range(3))
            if exp == "e68":
                d[f"who{L}"] = (ls[1] or {}).get("who")
        n = nla.get(r["key"])
        t = nla_text(n)
        d["nla_text"] = t
        d["nla"] = None if t is None else bool(binary(rub, labs.get((rub, NLA_Q, t))))
        d["nla_kw"] = None if t is None else bool(KW[exp].search(t))
        t60 = nla_text(n, 60)
        d["nla60"] = None if t60 is None else bool(binary(rub, labs.get((rub, NLA_Q, t60))))
        top = {L: r["jl"][L][:10] for L in BAND}
        d["jl"] = any(set(top[L]) & set(sets[exp]) for L in BAND)
        d["jl_layers"] = [L for L in BAND if set(top[L]) & set(sets[exp])]
        out.append(d)
    return out


def rate_row(rows, key):
    v = [float(bool(r[key])) for r in rows if r[key] is not None]
    c = [r["cluster"] for r in rows if r[key] is not None]
    m, lo, hi = boot(v, c)
    return m, lo, hi, len(v)


def fmt(x):
    m, lo, hi, n = x
    return f"{m:.2f} [{lo:.2f}, {hi:.2f}] (n {n})" if n else "–"


def metrics(res, pilot_only=False):
    labs = load_labels(res)
    tok = tok_()
    sets = jl_sets(tok)
    R = {e: per_read(res, labs, sets, e) for e in EXPS}
    if pilot_only:
        R = {e: [r for r in v if r["pilot"]] for e, v in R.items()}
    L = []
    M = {}
    for exp, rows in R.items():
        if not rows:
            continue
        L.append(f"\n## {exp}\n")
        L.append("| condition | position | n | AO L16 | AO L32 | AO L48 | AO L32 keyword | NLA | J-lens | P(Yes) L32 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        groups = defaultdict(list)
        for r in rows:
            groups[(r["cond"], r["kind"])].append(r)
        for (c, k), g in sorted(groups.items()):
            py = boot([r["pyes32"] for r in g], [r["cluster"] for r in g])
            L.append(f"| {c} | {k} | {len(g)} | " + " | ".join(fmt(rate_row(g, x)).split(" (n")[0] for x in
                     ("ao16", "ao32", "ao48", "ao32_kw", "nla", "jl")) + f" | {py[0]:.2f} [{py[1]:.2f}, {py[2]:.2f}] |")
        M[exp] = {f"{c}|{k}": {x: rate_row(g, x) for x in ("ao16", "ao32", "ao48", "ao32_kw", "nla", "jl")}
                  for (c, k), g in groups.items()}
    (res / "metrics_tables.md").write_text("\n".join(L))
    preds = predictions(R) if not pilot_only else {}
    e69 = audit(R)
    (res / "metrics.json").write_text(json.dumps({"tables": M, "predictions": preds, "e69": e69}, indent=1,
                                                 default=str))
    print("\n".join(L))
    print(json.dumps(preds, indent=1, default=str))
    print(json.dumps(e69, indent=1, default=str))
    return R


def sel(rows, cond=None, kinds=None):
    return [r for r in rows if (cond is None or r["cond"] in (cond if isinstance(cond, (list, tuple)) else [cond]))
            and (kinds is None or r["kind"] in kinds)]


def vs_floor(t, f, key="ao32"):
    return boot_diff([float(bool(r[key])) for r in t], [r["cluster"] for r in t],
                     [float(bool(r[key])) for r in f], [r["cluster"] for r in f])


def predictions(R):
    P = {}
    r65 = R.get("e65", [])
    if r65:
        nf, nt = sel(r65, "nat_false", ["pre"]), sel(r65, "nat_true", ["pre"])
        d = boot_diff([r["ao32"] for r in nf], [r["cluster"] for r in nf], [r["ao32"] for r in nt],
                      [r["cluster"] for r in nt])
        fl = vs_floor(nf, sel(r65, "nat_false", ["neutral"]))
        P["e65"] = {"notdone_false_minus_true": d, "false_vs_floor": fl,
                    "P1": d[0] >= 0.20 and fl[1] > 0, "P2": abs(d[0]) <= 0.10,
                    "auroc_natural": auroc([r["pyes32"] for r in nt], [r["pyes32"] for r in nf]),
                    "auroc_edits": auroc([r["pyes32"] for r in sel(r65, "call_removed", ["pre"])],
                                         [r["pyes32"] for r in sel(r65, "call_inserted", ["pre"])]),
                    "auroc_by_record_all": auroc([r["pyes32"] for r in sel(r65, ["nat_true", "call_inserted"], ["pre"])],
                                                 [r["pyes32"] for r in sel(r65, ["nat_false", "call_removed"], ["pre"])])}
        P["e65"]["P3"] = P["e65"]["auroc_natural"] >= 0.7
        P["e65"]["P4"] = (P["e65"]["auroc_edits"] < 0.6) if P["e65"]["P3"] else "not scored (P3 false)"
    r66 = R.get("e66", [])
    if r66:
        perf_c = next((c for c in sorted({r["cond"] for r in r66}) if c != "plain"), "theatrical")
        perf = sel(r66, perf_c, ["fab_first"])
        plain = sel(r66, "plain", ["fab_first", "first_content"])
        p1 = rate_row(perf, "ao32")
        d = boot_diff([r["ao32"] for r in perf], [r["cluster"] for r in perf], [r["ao32"] for r in plain],
                      [r["cluster"] for r in plain])
        fl = vs_floor(perf, sel(r66, perf_c, ["neutral"]))
        from scipy.stats import fisher_exact
        tj = sel(r66, perf_c, ["fab_first", "pre_fab"])
        pj = sel(r66, "plain", ["fab_first", "pre_fab", "first_content", "pre_first"])
        a, b = sum(r["jl"] for r in tj), sum(r["jl"] for r in pj)
        fp = fisher_exact([[a, len(tj) - a], [b, len(pj) - b]])[1]
        P["e66"] = {"performing_cond": perf_c, "fiction_rate_performing": p1, "performing_minus_plain": d,
                    "performing_vs_floor": fl, "jl": [a, len(tj), b, len(pj), fp],
                    "P1": p1[0] >= 0.60, "P2": d[0] >= 0.20 and fl[1] > 0,
                    "P3": fp < 0.05 and a / max(1, len(tj)) > b / max(1, len(pj))}
    r67 = R.get("e67", [])
    if r67:
        ba, bc = sel(r67, "batman", ["answer"]), sel(r67, "batman", ["cot"])
        na, nn = sel(r67, "none", ["answer"]), sel(r67, "none", ["neutral"])
        pa, pc = rate_row(ba, "ao32"), rate_row(bc, "ao32")
        nf = vs_floor(na, nn)
        d = boot_diff([r["ao32"] for r in bc], [r["cluster"] for r in bc], [r["ao32"] for r in ba],
                      [r["cluster"] for r in ba])
        plan = rate_row(bc, "ao32_plan")
        P["e67"] = {"batman_answer": pa, "batman_cot": pc, "none_answer_minus_floor": nf, "cot_minus_answer": d,
                    "planning_cot": plan, "P1": pa[0] >= 0.5 and not nf[1] > 0, "P2": d[0] <= -0.20,
                    "P3": plan[0] >= 0.30}
    r68 = R.get("e68", [])
    if r68:
        pd_, rf = sel(r68, "paid", ["decision"]), sel(r68, "refused", ["decision"])
        a, b = rate_row(pd_, "ao32"), rate_row(rf, "ao32")
        fa, fb = vs_floor(pd_, sel(r68, "paid", ["neutral"])), vs_floor(rf, sel(r68, "refused", ["neutral"]))
        d = boot_diff([r["ao32"] for r in rf], [r["cluster"] for r in rf], [r["ao32"] for r in pd_],
                      [r["cluster"] for r in pd_])
        who = Counter(r["who32"] for r in pd_)
        P["e68"] = {"paid": a, "refused": b, "paid_vs_floor": fa, "refused_vs_floor": fb, "refused_minus_paid": d,
                    "who_paid": dict(who), "P1": abs(d[0]) <= 0.15 and fa[1] > 0 and fb[1] > 0,
                    "P2": who.get("SUPERVISOR", 0) / max(1, len(pd_)) >= 0.5, "P3": d[0] >= 0.20}
    return P


def truth(r):
    if r["exp"] == "e65":
        return r["cond"] in ("nat_false", "call_removed")
    if r["exp"] == "e67":
        return r["cond"] == "batman"
    return True


def audit(R):
    """e69: accuracy against ground truth on target reads, neutral false-positive rate, pairwise kappa."""
    from readers.judge import kappa
    out = {"per_concept": {}, "pooled": {}}
    pooled = defaultdict(lambda: {"acc": [], "fp": []})
    pairs = defaultdict(lambda: ([], []))
    disagree = defaultdict(Counter)
    readers = {"AO": "ao32", "NLA": "nla", "JL": "jl"}
    for exp, rows in R.items():
        tg = [r for r in rows if r["kind"] != "neutral"]
        ne = [r for r in rows if r["kind"] == "neutral"]
        pc = {}
        for name, key in readers.items():
            acc = [float(bool(r[key]) == truth(r)) for r in tg if r[key] is not None]
            fp = [float(bool(r[key])) for r in ne if r[key] is not None]
            pc[name] = {"accuracy": float(np.mean(acc)) if acc else None, "n": len(acc),
                        "neutral_fp": float(np.mean(fp)) if fp else None, "n_neutral": len(fp)}
            pooled[name]["acc"] += acc
            pooled[name]["fp"] += fp
        ks = {}
        for a, b in (("AO", "NLA"), ("AO", "JL"), ("NLA", "JL")):
            ok = [r for r in tg if r[readers[a]] is not None and r[readers[b]] is not None]
            x, y = [bool(r[readers[a]]) for r in ok], [bool(r[readers[b]]) for r in ok]
            ks[f"{a}-{b}"] = kappa(x, y) if ok and len(set(x + y)) > 1 else None
            pairs[f"{a}-{b}"][0].extend(x)
            pairs[f"{a}-{b}"][1].extend(y)
            for r in ok:
                if bool(r[readers[a]]) != bool(r[readers[b]]):
                    disagree[f"{a}-{b}"][a if bool(r[readers[a]]) == truth(r) else b] += 1
        pc["kappa"] = ks
        out["per_concept"][exp] = pc
    for name in readers:
        out["pooled"][name] = {"accuracy": float(np.mean(pooled[name]["acc"])) if pooled[name]["acc"] else None,
                               "neutral_fp": float(np.mean(pooled[name]["fp"])) if pooled[name]["fp"] else None,
                               "n": len(pooled[name]["acc"]), "n_neutral": len(pooled[name]["fp"])}
    out["pooled"]["kappa"] = {k: (kappa(*v) if v[0] and len(set(v[0] + v[1])) > 1 else None) for k, v in pairs.items()}
    out["disagreements_matched_truth"] = {k: dict(v) for k, v in disagree.items()}
    acc = {n: out["pooled"][n]["accuracy"] for n in readers}
    fp = {n: out["pooled"][n]["neutral_fp"] for n in readers}
    if all(v is not None for v in acc.values()):
        out["P1"] = acc["AO"] > acc["JL"] > acc["NLA"]
    if all(v is not None for v in fp.values()):
        out["P2"] = fp["NLA"] == max(fp.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["eyeball", "judge", "validate", "metrics"])
    ap.add_argument("--res", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--pilot-only", action="store_true")
    ap.add_argument("--no-batch", action="store_true")
    a = ap.parse_args()
    res = Path(a.res)
    if a.step == "eyeball":
        eyeball(res, a.n)
    elif a.step == "judge":
        judge(res, batch=not a.no_batch)
    elif a.step == "validate":
        validate(res)
    else:
        metrics(res, a.pilot_only)


if __name__ == "__main__":
    main()
