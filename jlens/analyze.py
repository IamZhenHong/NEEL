"""Analysis of the J-lens claim reads (prereg jlens/README.md). Local, no model needed (decodes with the tokenizer).
  uv run python -m jlens.analyze --res jlens/results --eyeball   # step 1: the by-eye dump only (written before metrics)
  uv run python -m jlens.analyze --res jlens/results             # step 2: pre-registered metrics -> summary.md"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact
from transformers import AutoTokenizer

BAND = list(range(20, 61, 4))
NAMED = ["NEG", "DECEIT", "DONE", "EVAL"]
POSNAMES = ["turn_start", "pre", "preverb", "verb"]


def load(res):
    d = dict(np.load(res / "reads.npz"))      # npz members decompress on every access otherwise
    meta = [json.loads(l) for l in open(res / "reads_meta.jsonl")]
    layers = list(d["layers"])
    band = [layers.index(L) for L in BAND]
    for m in meta:
        rp = m["read_pos"]
        m["ix"] = {p: rp.index(m["pos"][p]) for p in POSNAMES}
    return d, meta, band, list(d["lenses"]), list(d["set_names"])


def pairs(meta):
    """(no_call index, call index, label, key) for each claim."""
    by = defaultdict(dict)
    for i, m in enumerate(meta):
        by[(m["key"], m["msg"], m["claim"])][m["version"]] = i
    out = []
    for (key, _, _), v in by.items():
        o, e = v["orig"], v["edit"]
        nc, c = (o, e) if not meta[o]["has_call"] else (e, o)
        out.append((nc, c, meta[o]["label"], key))
    return out


def eyeball(d, meta, band, lenses, tok, res):
    rng = random.Random(0)
    li = lenses.index("jlens")
    first = {}
    for i, m in enumerate(meta):
        if m["version"] == "orig":
            first.setdefault((m["label"], m["key"]), i)
    fals = sorted(i for (lab, _), i in first.items() if lab == "false")
    tru = sorted(i for (lab, _), i in first.items() if lab == "true")
    rng.shuffle(tru)
    edit_of = {(meta[i]["key"], meta[i]["msg"], meta[i]["claim"]): i for i, m in enumerate(meta) if m["version"] == "edit"}
    lines = []

    def block(i, title):
        m = meta[i]
        lines.append(f"\n=== {title} | {m['key']} msg {m['msg']} | call in context: {m['has_call']}")
        lines.append("    claim: " + m["claim"].replace("\n", " "))
        for p in ("pre", "preverb"):
            lines.append(f"  -- position {p}")
            for j in band:
                ids = d["top_ids"][i, li, j, m["ix"][p], :10]
                lines.append(f"    L{BAND[band.index(j)]:>2}: " + " | ".join(repr(tok.decode([int(t)]))[1:-1] for t in ids))
    for i in fals[:20]:
        block(i, "FALSE claim (no call), as generated")
        block(edit_of[(meta[i]["key"], meta[i]["msg"], meta[i]["claim"])], "  same claim, flag call INSERTED")
    for i in tru[:20]:
        block(i, "TRUE claim (call made), as generated")
    (res / "eyeball.txt").write_text("\n".join(lines))
    print(f"wrote {res / 'eyeball.txt'} ({len(fals[:20])} false, {len(tru[:20])} true)")


def cluster_boot(vals, keys, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for v, k in zip(vals, keys):
        groups[k].append(v)
    g = [np.array(v) for v in groups.values()]
    sums, cnts = np.array([x.sum() for x in g]), np.array([len(x) for x in g])
    idx = rng.integers(0, len(g), (n, len(g)))
    boot = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def sign_flip_p(vals, keys, n=10000, seed=0):
    """One-sided p for mean > 0, flipping signs by episode cluster."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for v, k in zip(vals, keys):
        groups[k].append(v)
    sums = np.array([np.sum(v) for v in groups.values()])
    tot = len(vals)
    obs = sums.sum() / tot
    flips = rng.choice([-1, 1], (n, len(sums)))
    null = (flips * sums).sum(1) / tot
    return float((null >= obs).mean())


def metrics(d, meta, band, lenses, set_names, res):
    P = pairs(meta)
    keys = [k for _, _, _, k in P]
    out = ["# J-lens claim reads: pre-registered metrics", ""]
    # M1: natural false vs true, NEG in top 10 at any band layer
    li = lenses.index("jlens")
    out.append("## M1: natural reads, any NEG token in the J-lens top 10 at any layer 20-60")
    out.append("| position | false claims | true claims | Fisher p (two-sided) |")
    out.append("|---|---|---|---|")
    for p in ("pre", "preverb", "verb", "turn_start"):
        hit = {lab: [] for lab in ("false", "true")}
        for i, m in enumerate(meta):
            if m["version"] == "orig":
                hit[m["label"]].append(bool((d["best"][i, li, band, m["ix"][p], NAMED.index("NEG")] <= 10).any()))
        a, b = sum(hit["false"]), sum(hit["true"])
        _, pv = fisher_exact([[a, len(hit["false"]) - a], [b, len(hit["true"]) - b]])
        out.append(f"| {p} | {a}/{len(hit['false'])} | {b}/{len(hit['true'])} | {pv:.3f} |")
    # M2/M3/S1: paired no-call minus call, log-mass averaged over band layers
    out.append("")
    out.append("## M2 (NEG, primary), M3 (DECEIT), S1 (DONE, sanity), EVAL: paired no-call minus call")
    n_ins = sum(lab == "false" for _, _, lab, _ in P)
    out.append(f"Delta = mean over layers 20-60 of log-mass(no-call) - log-mass(call), same claim text. {len(P)} pairs "
               f"({n_ins} false claims with a call inserted, {len(P) - n_ins} true claims with the call removed); CI = episode-cluster "
               "bootstrap; p = one-sided episode sign-flip (Delta > 0); 'rand pct' = share of the 200 random sets with a "
               "smaller mean Delta.")
    for lens in lenses:
        lj = lenses.index(lens)
        out.append(f"\n### lens: {lens}")
        out.append("| position | set | mean Delta | 95% CI | p (Delta>0) | rand pct | z vs rand | insert pairs | remove pairs |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for p in ("pre", "preverb", "verb", "turn_start"):
            def delta(si):
                return np.array([d["mass"][nc, lj, band, meta[nc]["ix"][p], si].mean()
                                 - d["mass"][c, lj, band, meta[c]["ix"][p], si].mean() for nc, c, _, _ in P])
            rand = np.array([delta(si).mean() for si, s in enumerate(set_names) if s.startswith("rand")])
            for s in NAMED:
                dl = delta(set_names.index(s))
                lo, hi = cluster_boot(dl, keys)
                ins = dl[[lab == "false" for _, _, lab, _ in P]].mean()
                rem = dl[[lab == "true" for _, _, lab, _ in P]].mean()
                out.append(f"| {p} | {s} | {dl.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {sign_flip_p(dl, keys):.3f} | "
                           f"{(rand < dl.mean()).mean():.2f} | {(dl.mean() - rand.mean()) / rand.std():+.2f} | "
                           f"{ins:+.3f} | {rem:+.3f} |")
    (res / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    return P


def exploratory(d, meta, band, lenses, tok, res, P):
    """Post hoc by design: which tokens move between versions; how often top tokens show up later (idea 5)."""
    li = lenses.index("jlens")
    out = ["", "## Exploratory (not pre-registered as tests)", ""]
    for p in ("pre", "preverb"):
        cnt = Counter()
        for nc, c, _, _ in P:
            a = set(d["top_ids"][nc, li, band, meta[nc]["ix"][p], :10].ravel().tolist())
            b = set(d["top_ids"][c, li, band, meta[c]["ix"][p], :10].ravel().tolist())
            for t in a - b:
                cnt[t] += 1
            for t in b - a:
                cnt[t] -= 1
        more_nc = [f"{tok.decode([t])!r} {v:+d}" for t, v in cnt.most_common(15)]
        more_c = [f"{tok.decode([t])!r} {v:+d}" for t, v in sorted(cnt.items(), key=lambda x: x[1])[:15]]
        out.append(f"- {p}, J-lens top 10 at any layer 20-60, pairs where the token is present only without the call "
                   f"minus only with it ({len(P)} pairs):")
        out.append(f"  - more without the call: {', '.join(more_nc)}")
        out.append(f"  - more with the call: {', '.join(more_c)}")
    # idea 5: share of top-10 lens tokens (alphabetic, 3+ letters) that never appear in the rest of the reply
    out.append("")
    out.append("- Top-10 tokens at 'pre' (layers 20-60, word-like only) that appear in the next 600 characters of the reply:")
    for lens in lenses:
        lj = lenses.index(lens)
        seen, tot = 0, 0
        for i, m in enumerate(meta):
            if m["version"] != "orig":
                continue
            rest = m["rest"].lower()
            for t in set(d["top_ids"][i, lj, band, m["ix"]["pre"], :10].ravel().tolist()):
                w = tok.decode([t]).strip().lower()
                if re.fullmatch(r"[a-z]{3,}", w):
                    tot += 1
                    seen += bool(re.search(r"\b" + w, rest))
        out.append(f"  - {lens}: {seen}/{tot} = {seen / max(tot, 1):.2f}")
    with open(res / "summary.md", "a") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="jlens/results")
    ap.add_argument("--eyeball", action="store_true")
    a = ap.parse_args()
    res = Path(a.res)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    d, meta, band, lenses, set_names = load(res)
    if a.eyeball:
        eyeball(d, meta, band, lenses, tok, res)
        return
    P = metrics(d, meta, band, lenses, set_names, res)
    exploratory(d, meta, band, lenses, tok, res, P)


if __name__ == "__main__":
    main()
