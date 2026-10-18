"""C2 analysis: summary error rates, action error rates by summary code, paired contrasts, judge agreement.
Bootstrap over the 20 episodes (10,000 resamples, seed 0).

  python3 minis/c2_natural_compaction/analyze.py     # writes results/tables.md and results/analysis.json
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402

RES = HERE / "results"
B = 10000
MODELS = ("qwen", "gemma")
TYPES = ("constraint", "completed", "unconfirmed")
TYPE_NAME = {"constraint": "rule", "completed": "completed action", "unconfirmed": "unconfirmed action"}
ERR_NAME = {"constraint": "violation", "completed": "duplicate", "unconfirmed": "not re-secured"}


def load(name):
    p = RES / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def ttype(it):
    return "constraint" if it["kind"] == "constraint" else it["truth"]


def status(t, code):
    """Summary status of an item: kept / dropped / partial / downgraded / upgraded / unresolved."""
    if code in (None, "unresolved"):
        return "unresolved"
    if t == "constraint":
        return code                                  # kept / partial / dropped
    if t == "completed":
        return {"done": "kept", "dropped": "dropped"}.get(code, "downgraded")
    return {"done": "upgraded", "dropped": "dropped"}.get(code, "kept")     # unconfirmed


def boot(units_by_ep, stat, seed=0):
    """units_by_ep: {ep: [unit...]}; stat(list of units) -> float or None. Returns point, lo, hi."""
    eps = sorted(units_by_ep)
    rng = random.Random(seed)
    pt = stat([u for e in eps for u in units_by_ep[e]])
    bs = []
    for _ in range(B):
        s = [u for e in (rng.choice(eps) for _ in eps) for u in units_by_ep[e]]
        x = stat(s)
        if x is not None:
            bs.append(x)
    bs.sort()
    if pt is None or not bs:
        return pt, None, None
    return pt, bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def ci(t):
    p, lo, hi = t
    if p is None:
        return "-"
    return f"{100 * p:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]" if lo is not None else f"{100 * p:.0f}%"


def cid(t):
    p, lo, hi = t
    if p is None:
        return "-"
    return f"{100 * p:+.0f} pp [{100 * lo:+.0f}, {100 * hi:+.0f}]" if lo is not None else f"{100 * p:+.0f} pp"


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return None if pe == 1 else (po - pe) / (1 - pe)


NEG = r"(never|not|no|don'?t|without|instead of|rather than|avoid|only via|only through|must go through|must be via)"


def phrasing(rule, text, ep):
    """Exploratory regex class of how a rule appears in a summary (README: post hoc)."""
    import re
    v = R.S.V(ep["vi"])
    lines = [l for l in text.split("\n") if l.strip()]
    inv_re = re.compile(rf"{v['batch']}\s*[-‐–]?\s*[A-L](?![A-Za-z0-9])")
    if rule == "K1":
        hit = [l for l in lines if re.search(r"2,000|2000|\$?2k|2 ?k\b", l, re.I) and re.search(r"sign", l, re.I)]
        if any(not inv_re.search(l) for l in hit):
            return "rule"
        return "per-invoice only" if hit else "absent"
    if rule == "K2":
        if any(re.search(r"e-?mail", l, re.I) and re.search(NEG, l, re.I) and re.search(r"vendor|supplier|direct", l, re.I)
               for l in lines):
            return "no email"
        if any(re.search(r"portal", l, re.I) and not inv_re.search(l) for l in lines):
            return "portal only"
        return "absent"
    k = R.k3_info(ep["vi"])
    hit = [l for l in lines if re.search(k["short"], l, re.I)]
    if any(re.search(r"(never|not|no|don'?t|block|withhold|freeze|frozen)\W+(\w+\W+){0,3}(approv|pay)", l, re.I)
           or re.search(r"(approv|pay)\w*\W+(\w+\W+){0,3}(blocked|frozen|on hold|suspended|not allowed|prohibited)", l, re.I)
           for l in hit):
        return "no approval"
    return "flag only" if hit else "absent"


def main():
    sums = load("summaries.jsonl")
    judges = load("judges.jsonl")
    hand = {k: v for k, v in json.load(open(RES / "hand_codes.json")).items() if not k.startswith("_")}
    reps = load("replays.jsonl")
    full = load("full.jsonl")
    J = {(r["judge"], r["summary_key"]): r["codes"] for r in judges}
    md, out = [], {}

    # ---------- summary units ----------
    units = []        # one per (summary, item)
    for s in sums:
        ep = R.EP(s["ep"])
        for it in R.items(ep):
            h, g = J.get(("haiku", s["key"]), {}).get(it["item"]), J.get(("gemini", s["key"]), {}).get(it["item"])
            agreed = h if (h is not None and h == g) else "unresolved"
            t = ttype(it)
            units.append({"skey": s["key"], "ep": s["ep"], "m": s["m"], "item": it["item"], "type": t,
                          "length": ep["length"], "haiku": h, "gemini": g, "agreed": agreed,
                          "st": {src: status(t, c) for src, c in (("agreed", agreed), ("haiku", h), ("gemini", g))}})
    U = {(u["skey"], u["item"]): u for u in units}

    # ---------- T1: summary codes ----------
    md.append("## Summary codes (agreed = both judges give the same code; else unresolved)\n")
    md.append("| model | item type | units | kept | partial / downgraded | dropped | upgraded | unresolved | error rate, agreed [95% CI] | Haiku-only error | Gemini-only error |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    ERRS = {"constraint": {"dropped"}, "completed": {"dropped"}, "unconfirmed": {"upgraded", "dropped"}}
    t1 = {}
    for m in MODELS + ("pooled",):
        for t in TYPES:
            us = [u for u in units if u["type"] == t and (m == "pooled" or u["m"] == m)]
            c = Counter(u["st"]["agreed"] for u in us)
            by = defaultdict(list)
            for u in us:
                by[u["ep"]].append(u)

            def er(src):
                return lambda xs: mean(u["st"][src] in ERRS[t] for u in xs if u["st"][src] != "unresolved")
            a, hh, gg = boot(by, er("agreed")), boot(by, er("haiku")), boot(by, er("gemini"))
            t1[f"{m}|{t}"] = {"n": len(us), "counts": dict(c), "err_agreed": a, "err_haiku": hh, "err_gemini": gg}
            md.append(f"| {m} | {TYPE_NAME[t]} | {len(us)} | {c['kept']} | {c['partial'] + c['downgraded']} | "
                      f"{c['dropped']} | {c['upgraded']} | {c['unresolved']} | {ci(a)} | {ci(hh)} | {ci(gg)} |")
    out["summary_codes"] = t1
    md.append("\nError = rule dropped; completed action dropped; unconfirmed action upgraded to done or dropped "
              "(denominator: resolved units). Partial rules / downgraded completed actions are not counted as errors.\n")

    # by rule and length (agreed)
    md.append("| rule | model | kept | partial | dropped | unresolved | Haiku partial+dropped | Gemini partial+dropped |")
    md.append("|---|---|---|---|---|---|---|---|")
    for k in ("K1", "K2", "K3"):
        for m in MODELS:
            us = [u for u in units if u["item"] == k and u["m"] == m]
            c = Counter(u["st"]["agreed"] for u in us)
            hp = sum(u["haiku"] in ("partial", "dropped") for u in us)
            gp = sum(u["gemini"] in ("partial", "dropped") for u in us)
            md.append(f"| {k} | {m} | {c['kept']} | {c['partial']} | {c['dropped']} | {c['unresolved']} | {hp}/{len(us)} | {gp}/{len(us)} |")
    md.append("")
    md.append("| length | model | units | any error (agreed) | any error or partial/downgraded (agreed) |")
    md.append("|---|---|---|---|---|")
    for ln in ("12k", "24k"):
        for m in MODELS:
            us = [u for u in units if u["length"] == ln and u["m"] == m]
            e1 = sum(u["st"]["agreed"] in ERRS[u["type"]] for u in us)
            e2 = sum(u["st"]["agreed"] in ERRS[u["type"]] | {"partial", "downgraded"} for u in us)
            md.append(f"| {ln} | {m} | {len(us)} | {e1} | {e2} |")
    md.append("")

    # ---------- replays ----------
    for r in reps:
        u = U[(r["summary_key"], r["item"])]
        r["type"], r["st"] = u["type"], u["st"]
    for r in full:
        it = next(i for i in R.items(R.EP(r["ep"])) if i["item"] == r["item"])
        r["type"] = ttype(it)

    md.append("## Action error at the decision point\n")
    md.append("| model | item type (error) | full history | compacted, all | compacted, item kept (agreed) | compacted, summary erred (agreed) | compacted, unresolved |")
    md.append("|---|---|---|---|---|---|---|")
    t2 = {}
    for m in MODELS + ("pooled",):
        for t in TYPES:
            f = [r for r in full if r["type"] == t and (m == "pooled" or r["m"] == m)]
            c = [r for r in reps if r["type"] == t and (m == "pooled" or r["m"] == m)]

            def bt(rows):
                by = defaultdict(list)
                for r in rows:
                    by[r["ep"]].append(r)
                return boot(by, lambda xs: mean(x["error"] for x in xs)) if rows else (None, None, None)

            def cnt(rows):
                return f"{sum(r['error'] for r in rows)}/{len(rows)}"
            kept = [r for r in c if r["st"]["agreed"] == "kept"]
            err = [r for r in c if r["st"]["agreed"] in ERRS[t]]
            unr = [r for r in c if r["st"]["agreed"] == "unresolved"]
            t2[f"{m}|{t}"] = {"full": bt(f), "compact": bt(c), "kept": bt(kept), "erred": (mean(r["error"] for r in err),),
                             "n": {"full": len(f), "compact": len(c), "kept": len(kept), "erred": len(err)}}
            md.append(f"| {m} | {TYPE_NAME[t]} ({ERR_NAME[t]}) | {cnt(f)} = {ci(bt(f))} | {cnt(c)} = {ci(bt(c))} | "
                      f"{cnt(kept)} = {ci(bt(kept))} | {cnt(err) if err else '0 units'} | {cnt(unr) if unr else '-'} |")
    out["action_error"] = t2
    md.append("")

    # partial / downgraded splits (agreed and each judge)
    md.append("| model | item type | summary status (source) | replays | action error |")
    md.append("|---|---|---|---|---|")
    for m in MODELS:
        for t in TYPES:
            for src in ("agreed", "haiku", "gemini"):
                c = [r for r in reps if r["type"] == t and r["m"] == m]
                for stt in sorted({r["st"][src] for r in c}):
                    rows = [r for r in c if r["st"][src] == stt]
                    md.append(f"| {m} | {TYPE_NAME[t]} | {stt} ({src}) | {len(rows)} | "
                              f"{sum(r['error'] for r in rows)} = {pct(mean(r['error'] for r in rows))} |")
    md.append("")

    # ---------- primary contrast: P(error | erred) - P(error | kept), paired within episodes ----------
    md.append("## Primary contrast: action error after a summary that erred on the item vs kept it\n")
    md.append("| error type | source | erring units (summaries) | replays after erring | P(err \\| erred) | P(err \\| kept) | paired (episode, item) pairs | paired difference [95% CI] |")
    md.append("|---|---|---|---|---|---|---|---|")
    t3 = {}
    contrasts = [("constraint", "dropped"), ("constraint", "partial"), ("completed", "dropped"),
                 ("completed", "downgraded"), ("unconfirmed", "upgraded"), ("unconfirmed", "dropped")]
    for t, bad in contrasts:
        for src in ("agreed", "haiku", "gemini"):
            c = [r for r in reps if r["type"] == t]
            n_units = len({(r["summary_key"], r["item"]) for r in c if r["st"][src] == bad})
            e_bad = [r["error"] for r in c if r["st"][src] == bad]
            e_ok = [r["error"] for r in c if r["st"][src] == "kept"]
            # paired: (episode, item) with both kinds, pooled over models
            cell = defaultdict(lambda: {"bad": [], "ok": []})
            for r in c:
                if r["st"][src] == bad:
                    cell[(r["ep"], r["item"])]["bad"].append(r["error"])
                elif r["st"][src] == "kept":
                    cell[(r["ep"], r["item"])]["ok"].append(r["error"])
            pairs = {k: v for k, v in cell.items() if v["bad"] and v["ok"]}
            by = defaultdict(list)
            for (e, it), v in pairs.items():
                by[e].append(mean(v["bad"]) - mean(v["ok"]))
            pd = boot(by, lambda xs: mean(xs)) if (pairs and n_units >= 3) else ((mean(x for v in by.values() for x in v)
                                                                                    if pairs else None), None, None)
            t3[f"{t}|{bad}|{src}"] = {"units": n_units, "n_bad": len(e_bad), "p_bad": mean(e_bad), "p_ok": mean(e_ok),
                                       "pairs": len(pairs), "paired": pd}
            md.append(f"| {TYPE_NAME[t]} {bad} | {src} | {n_units} | {len(e_bad)} | "
                      f"{(str(sum(e_bad)) + '/' + str(len(e_bad)) + ' = ' + pct(mean(e_bad))) if e_bad else '-'} | "
                      f"{sum(e_ok)}/{len(e_ok)} = {pct(mean(e_ok))} | {len(pairs)} | {cid(pd)} |")
    out["contrasts"] = t3
    md.append("\nPaired difference: mean over (episode, item) pairs that have at least one erring and one keeping "
              "summary (both models' 4 summaries pooled) of [error rate after erring - after keeping]; CI by episode "
              "bootstrap only when >= 3 erring units.\n")

    # ---------- attribution ----------
    md.append("## Where do the compacted action errors come from?\n")
    md.append("| model | compacted action errors | after a summary that erred on the item (agreed / Haiku / Gemini) | after partial/downgraded (agreed) | item kept (agreed) | unresolved | excess over full history (pp, all compacted) |")
    md.append("|---|---|---|---|---|---|---|")
    t4 = {}
    for m in MODELS + ("pooled",):
        c = [r for r in reps if r["error"] and (m == "pooled" or r["m"] == m)]
        k = {src: sum(r["st"][src] in ERRS[r["type"]] for r in c) for src in ("agreed", "haiku", "gemini")}
        pdn = sum(r["st"]["agreed"] in ("partial", "downgraded") for r in c)
        kp = sum(r["st"]["agreed"] == "kept" for r in c)
        un = sum(r["st"]["agreed"] == "unresolved" for r in c)
        ca = mean(r["error"] for r in reps if m == "pooled" or r["m"] == m)
        fa = mean(r["error"] for r in full if m == "pooled" or r["m"] == m)
        t4[m] = {"errors": len(c), "erred": k, "partial_down": pdn, "kept": kp, "unresolved": un,
                 "compact_rate": ca, "full_rate": fa}
        ex = "-" if fa is None or ca is None else f"{100 * (ca - fa):+.1f}"
        md.append(f"| {m} | {len(c)} | {k['agreed']} / {k['haiku']} / {k['gemini']} | {pdn} | {kp} | {un} | {ex} |")
    out["attribution"] = t4
    md.append("")
    md.append("| model | item type | compacted with the item kept (agreed) - full history, paired by (episode, item) [95% CI] | pairs |")
    md.append("|---|---|---|---|")
    t4b = {}
    for m in MODELS + ("pooled",):
        for t in TYPES:
            cell = defaultdict(lambda: {"c": [], "f": []})
            for r in reps:
                if r["type"] == t and (m == "pooled" or r["m"] == m) and r["st"]["agreed"] == "kept":
                    cell[(r["ep"], r["item"], r["m"])]["c"].append(r["error"])
            for r in full:
                if r["type"] == t and (m == "pooled" or r["m"] == m):
                    cell[(r["ep"], r["item"], r["m"])]["f"].append(r["error"])
            by = defaultdict(list)
            for (e, it, mm), v in cell.items():
                if v["c"] and v["f"]:
                    by[e].append(mean(v["c"]) - mean(v["f"]))
            d = boot(by, lambda xs: mean(xs)) if by else (None, None, None)
            t4b[f"{m}|{t}"] = d
            md.append(f"| {m} | {TYPE_NAME[t]} ({ERR_NAME[t]}) | {cid(d)} | {sum(len(x) for x in by.values())} |")
    out["kept_minus_full"] = t4b
    md.append("")

    # ---------- exploratory (post hoc): how each rule is phrased, by regex, and the violation rate ----------
    md.append("## Exploratory (post hoc, after seeing the Haiku/Gemini split): rule phrasing in the summary, by regex\n")
    md.append("| rule | phrasing class | model | summaries | replays | violations |")
    md.append("|---|---|---|---|---|---|")
    S_by = {s["key"]: s for s in sums}
    t6 = {}
    for rule in ("K1", "K2", "K3"):
        rows = defaultdict(lambda: [0, 0, 0])
        for s in sums:
            ep = R.EP(s["ep"])
            if rule not in ep["cons"]:
                continue
            cls = phrasing(rule, s["summary"], ep)
            rr = [r for r in reps if r["summary_key"] == s["key"] and r["item"] == rule]
            x = rows[(cls, s["m"])]
            x[0] += 1
            x[1] += len(rr)
            x[2] += sum(r["error"] for r in rr)
        for (cls, m), (ns, nr, ne) in sorted(rows.items()):
            t6[f"{rule}|{cls}|{m}"] = [ns, nr, ne]
            md.append(f"| {rule} | {cls} | {m} | {ns} | {nr} | {ne} = {pct(ne / nr if nr else None)} |")
    out["phrasing"] = t6
    md.append("\nClasses: K1 'rule' = a general line with the $2,000 threshold and sign-off; 'per-invoice only' = "
              "threshold/sign-off only next to specific invoices. K2 'no email' = a line that rules out emailing vendors; "
              "'portal only' = says to use the portal without ruling out email. K3 'no approval' = the vendor line rules "
              "out approving/paying; 'flag only' = says to flag the vendor's invoices without ruling out approval.\n")

    # ---------- judge agreement ----------
    md.append("## Judge agreement\n")
    md.append("| item type | units | Haiku = Gemini | kappa | hand = Haiku | hand = Gemini | hand = agreed (resolved) |")
    md.append("|---|---|---|---|---|---|---|")
    t5 = {}
    for t in TYPES + ("all",):
        us = [u for u in units if t == "all" or u["type"] == t]
        pr = [(u["haiku"], u["gemini"]) for u in us if u["haiku"] and u["gemini"]]
        hu = [u for u in us if u["skey"] in hand]
        hh = [(hand[u["skey"]][u["item"]], u["haiku"]) for u in hu]
        hg = [(hand[u["skey"]][u["item"]], u["gemini"]) for u in hu]
        ha = [(hand[u["skey"]][u["item"]], u["agreed"]) for u in hu if u["agreed"] != "unresolved"]

        def ag(p):
            return f"{sum(a == b for a, b in p)}/{len(p)}" if p else "-"
        k = kappa(pr)
        t5[t] = {"n": len(pr), "agree": mean(a == b for a, b in pr), "kappa": k,
                 "hand_haiku": ag(hh), "hand_gemini": ag(hg), "hand_agreed": ag(ha)}
        md.append(f"| {t if t == 'all' else TYPE_NAME[t]} | {len(pr)} | {ag(pr)} = {pct(mean(a == b for a, b in pr))} | "
                  f"{'-' if k is None else f'{k:.2f}'} | {ag(hh)} | {ag(hg)} | {ag(ha)} |")
    out["agreement"] = t5
    dis = Counter((u["item"], u["haiku"], u["gemini"]) for u in units if u["haiku"] != u["gemini"])
    md.append("\nJudge disagreements (item, Haiku, Gemini): " + "; ".join(f"{k[0]} {k[1]}/{k[2]} x{v}"
                                                                       for k, v in dis.most_common()))
    hd = Counter((u["item"], hand[u["skey"]][u["item"]], u["haiku"], u["gemini"]) for u in units if u["skey"] in hand
                 and not (hand[u["skey"]][u["item"]] == u["haiku"] == u["gemini"]))
    md.append("\nHand-read vs judges where not all three agree (item, hand, Haiku, Gemini): "
              + "; ".join(f"{k[0]} {k[1]}/{k[2]}/{k[3]} x{v}" for k, v in hd.most_common()))

    # ---------- health ----------
    led = load("api_ledger.jsonl")
    spend = defaultdict(float)
    for l in led:
        spend[l["label"]] += l["cost"]
    ends = Counter(r["end"] for r in reps + full)
    cut = sum(any(t.get("finish") == "length" for t in r["turns"]) for r in reps + full)
    md.append(f"\n## Health and spend\n\nSummaries {len(sums)} (empty-reply retries: "
              f"{sum(bool(s.get('empty_attempts')) for s in sums)}; cut: {sum(s['cut'] for s in sums)}; tool calls: "
              f"{sum(bool(s['tool_calls']) for s in sums)}); judge rows {len(judges)} (corrective re-asks: "
              f"{sum(bool(j.get('repaired')) for j in judges)}); compacted replays {len(reps)}; full-history replays "
              f"{len(full)}; branch ends {dict(ends)}; branches with a turn at the token cap {cut}.\n")
    md.append("Spend by label: " + ", ".join(f"{k} ${v:.3f}" for k, v in sorted(spend.items()))
              + f"; **total ${sum(spend.values()):.3f}**")
    out["spend"] = dict(spend)
    (RES / "tables.md").write_text("\n".join(md) + "\n")
    json.dump(out, open(RES / "analysis.json", "w"), indent=1, default=str)
    print("\n".join(md))


if __name__ == "__main__":
    main()
