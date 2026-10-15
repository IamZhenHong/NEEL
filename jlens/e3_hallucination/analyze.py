"""jl3 analysis (DESIGN.md; implementation README.md). Local, no model.
  uv run python -m jlens.e3_hallucination.analyze [--res jlens/e3_hallucination/results]
Verification -> one pair per item (first candidate whose present prompt gives the original answer and whose twin gives
the twin answer, not the original's) -> hit rate (X in the top 10 at any band layer, present), false-positive rate
(X in the top 10 at any band layer, absent twin), lexical-leak split, AUROC of X's best band rank, all per lens and
for layers 20-60 step 4 (primary) and all layers 0-62; the reverse direction (X' in its twin vs in the original) and
all verified candidates as secondary rows; the descriptive 'said later' share."""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
BAND = list(range(20, 61, 4))
ALL = list(range(63))
LENSES = ["jlens", "jpp", "logit", "jlens_cos"]
VERIFY_FAMS = {"association", "typo"}


def answer_match(text, ans):
    if text is None or not ans.strip():
        return False
    t = re.sub(r"^[\s\"'“”„«»*_`]+", "", text)
    a = ans.strip()
    if not t.lower().startswith(a.lower()):
        return False
    nxt = t[len(a):len(a) + 1]
    return not (nxt.isalnum() or nxt == "_")


def best(rk, scope):
    v = [rk[L] for L in scope if rk[L] is not None]
    return min(v) if v else None


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def boot(fn, n_items, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = [fn(rng.integers(0, n_items, n_items)) for _ in range(n)]
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(HERE / "results"))
    a = ap.parse_args()
    res = Path(a.res)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    prompts = {p["pid"]: p for p in map(json.loads, open(HERE / "data" / "prompts_jl3.jsonl"))}
    pairs = [json.loads(l) for l in open(HERE / "data" / "pairs_jl3.jsonl")]
    gen = {g["pid"]: g for g in map(json.loads, open(res / "jl3_gen.jsonl"))}
    ranks = {r["pid"]: r["ranks"] for r in map(json.loads, open(res / "reads_jl3_ranks.jsonl"))}
    npz = np.load(res / "reads_jl3.npz")
    top_ids, lens_names = npz["top_ids"], list(npz["lenses"])
    pid_row = {pid: i for i, pid in enumerate(sorted(gen))}
    assert all(gen[pid]["pid"] == pid for pid in gen)

    def said(pid):
        g = gen[pid]
        return g["verify_gen"] if prompts[pid]["family"] in VERIFY_FAMS else g["gen"]

    # ---- verification
    for p in pairs:
        p["present_ok"] = answer_match(said(p["present_pid"]), p["present_answer"])
        tw = said(p["twin_pid"])
        p["twin_ok"] = answer_match(tw, p["twin_answer"]) and not answer_match(tw, p["present_answer"])
        p["verified"] = p["present_ok"] and p["twin_ok"]
    chosen = {}
    for p in sorted(pairs, key=lambda q: q["cand"]):
        if p["verified"] and p["present_pid"] not in chosen:      # one pair per distinct present prompt
            chosen[p["present_pid"]] = p
    P = list(chosen.values())
    allv = [p for p in pairs if p["verified"]]
    fams = sorted({p["family"] for p in pairs})
    out = ["# jl3: J-lens hallucination rate on matched present/absent twins (Qwen3.6-27B)", "",
           "## Verification (greedy answers; association/typo via the appended question)", "",
           "| family | items | candidate pairs | present answers right | twin answers right | verified candidates | items with a verified pair |",
           "|---|---|---|---|---|---|---|"]
    for f in fams + ["all"]:
        q = [p for p in pairs if f in ("all", p["family"])]
        pres = {p["present_pid"]: p["present_ok"] for p in q}
        out.append(f"| {f} | {len({p['name'] for p in q}) if f != 'all' else len({(p['family'], p['name']) for p in q})} | "
                   f"{len(q)} | {sum(pres.values())}/{len(pres)} | {sum(p['twin_ok'] for p in q)}/{len(q)} | "
                   f"{sum(p['verified'] for p in q)} | {sum(1 for p in P if f in ('all', p['family']))} |")
    nfam = len({p["family"] for p in P})
    out += ["", f"**{len(P)} verified pairs (one per distinct present prompt) across {nfam} families** (card target: >= 150 across >= 3).", ""]
    summary = {"n_pairs": len(P), "n_families": nfam, "by_family": dict(Counter(p["family"] for p in P))}

    # ---- main measures
    def rows_for(PP, direction, lens, scope):
        """per pair: (best rank where the concept is present, best rank where absent, leak flag)."""
        rows = []
        for p in PP:
            key = f"X{p['cand']}" if direction == "X" else f"Xp{p['cand']}"
            if direction == "Xp" and not p["Xp_ids"]:
                continue
            pres_pid, abs_pid = (p["present_pid"], p["twin_pid"]) if direction == "X" else (p["twin_pid"], p["present_pid"])
            bp, ba = best(ranks[pres_pid][key][lens], scope), best(ranks[abs_pid][key][lens], scope)
            if bp is None or ba is None:
                continue
            rows.append((bp, ba, p["leak"], p["family"]))
        return rows

    def stats(rows):
        bp = np.array([r[0] for r in rows])
        ba = np.array([r[1] for r in rows])
        lk = np.array([r[2] for r in rows])
        n = len(rows)
        hit, fp = (bp <= 10).mean(), (ba <= 10).mean()
        return {"n": n, "hit": float(hit), "hit_ci": boot(lambda i: (bp[i] <= 10).mean(), n),
                "fp": float(fp), "fp_ci": boot(lambda i: (ba[i] <= 10).mean(), n),
                "fp_leak": [int((ba[lk] <= 10).sum()), int(lk.sum())],
                "fp_noleak": [int((ba[~lk] <= 10).sum()), int((~lk).sum())],
                "auroc": auroc(-bp, -ba), "auroc_ci": boot(lambda i: auroc(-bp[i], -ba[i]), n, n=1000)}

    table = ["| lens | layers | n | hit rate (present) | 95% CI | false-positive rate (absent twin) | 95% CI | FP, twin shares X's context words | FP, no shared words | AUROC (best rank) | 95% CI |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    def add_rows(PP, direction, title, key):
        out.extend(["", title, ""] + table)
        for lens in LENSES:
            for sname, scope in (("20-60/4", BAND), ("all 0-62", ALL)):
                if lens == "jlens_cos" and scope is ALL:
                    continue
                s = stats(rows_for(PP, direction, lens, scope))
                summary[f"{key}_{lens}_{'band' if scope is BAND else 'all'}"] = s
                fl, fn = s["fp_leak"], s["fp_noleak"]
                out.append(f"| {lens}{' (exploratory: WorkspaceBench cosine readout)' if lens == 'jlens_cos' else ''} | {sname} | {s['n']} | "
                           f"{s['hit']:.2f} | [{s['hit_ci'][0]:.2f}, {s['hit_ci'][1]:.2f}] | {s['fp']:.3f} | "
                           f"[{s['fp_ci'][0]:.3f}, {s['fp_ci'][1]:.3f}] | {fl[0]}/{fl[1]} ({fl[0] / max(fl[1], 1):.2f}) | "
                           f"{fn[0]}/{fn[1]} ({fn[0] / max(fn[1], 1):.2f}) | {s['auroc']:.3f} | "
                           f"[{s['auroc_ci'][0]:.3f}, {s['auroc_ci'][1]:.3f}] |")

    add_rows(P, "X", "## Primary: X (the original's intermediate), present prompt vs its absent twin, one pair per item", "main")
    # by family (J-lens and logit lens, band)
    out += ["", "### By family (layers 20-60/4)", "", "| family | n | J-lens hit | J-lens FP | J++ hit | J++ FP | logit hit | logit FP |",
            "|---|---|---|---|---|---|---|---|"]
    for f in fams:
        PP = [p for p in P if p["family"] == f]
        if not PP:
            continue
        cells = []
        for lens in ("jlens", "jpp", "logit"):
            r = rows_for(PP, "X", lens, BAND)
            cells += [f"{np.mean([x[0] <= 10 for x in r]):.2f}", f"{np.mean([x[1] <= 10 for x in r]):.2f}"]
        out.append(f"| {f} | {len(PP)} | " + " | ".join(cells) + " |")
    add_rows(P, "Xp", "## Secondary: reverse direction, X' (the twin's own intermediate) in its twin vs in the original", "rev")
    add_rows(allv, "X", "## Secondary: all verified candidates (up to 2 twins per item)", "allcand")
    # absent-twin false positives: which X were reported
    li = "jlens"
    fps = [(p["family"], p["name"], p["X"], best(ranks[p["twin_pid"]][f"X{p['cand']}"][li], BAND), prompts[p["twin_pid"]]["text"][-70:])
           for p in P if best(ranks[p["twin_pid"]][f"X{p['cand']}"][li], BAND) <= 10]
    out += ["", f"### J-lens false positives (band), {len(fps)} absent twins", ""]
    out += [f"- {f} / {n}: X {x} rank {r} in twin '...{t}'".replace("\n", "\\n") for f, n, x, r, t in fps[:40]]

    # ---- descriptive: share of top-10 word tokens said in the next 50 tokens (present prompts of the chosen pairs)
    out += ["", "## Descriptive: top-10 word tokens at the read position (layers 20-60/4) that appear in the next 50 "
            "generated tokens (present prompts of the verified pairs; jlens/REPORT.md had 31% / 26% / 41% before claims)", ""]
    pres = sorted({p["present_pid"] for p in P})
    for lens in LENSES:
        lj = lens_names.index(lens)
        seen = tot = 0
        for pid in pres:
            g = gen[pid]["gen"].lower()
            for t in set(top_ids[pid_row[pid], lj, BAND, :10].ravel().tolist()):
                w = tok.decode([t]).strip().lower()
                if re.fullmatch(r"[a-z]{3,}", w):
                    tot += 1
                    seen += bool(re.search(r"\b" + w, g))
        summary[f"said_{lens}"] = [seen, tot]
        out.append(f"- {lens}: {seen}/{tot} = {seen / max(tot, 1):.2f}")

    (res / "summary.md").write_text("\n".join(out) + "\n")
    (res / "summary.json").write_text(json.dumps(summary, indent=1))
    with open(res / "verified_pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps({k: p[k] for k in ("family", "name", "cand", "present_pid", "twin_pid", "X", "Xp", "leak",
                                                   "present_ok", "twin_ok", "verified")}
                               | {"chosen": chosen.get(p["present_pid"]) is p}, ensure_ascii=False) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
