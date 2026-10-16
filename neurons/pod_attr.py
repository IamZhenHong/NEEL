"""HF stages on the pod (one process, one model load). Stages run in the order given:

  python -m neurons.pod_attr --kind qwen36 --out neurons/results/<dir> --stages gate e70 [--pilot]

gate  : spider self-check; determinism (the top-20 at t of one e70 episode, two runs); the claim token is the greedy
        continuation at t in the natural false episodes.
e70   : attribution at t-1, t, t+1 for every e70 transcript (L = log-prob of the actual next token; at t that is the
        claim token); stability (a) and (b); the stop rule; differential set (nat_false - nat_true) with bootstrap CIs;
        the edits as the context control; per-condition frequent neurons. Writes e70_*.json / .npz and STOP_RULE.json.
Later stages (e71, e72, corpus, ablations) live in pod_main.py and import from here."""
import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from neurons.common import (HERE, SPIDER, L_logprob, MLPHooks, ablated_continuations, attribute, dev_of,
                            set_deterministic,
                            dump_jsonl, jaccard, load_jsonl, load_model, load_tokenizer, log, nid, shape_of,
                            split_half, topk_neurons)

CLAIM = re.compile(r"\bI(?:'ve|’ve| have)(?: already| also| just| now| actually)* "
                   r"(flagged|reported|escalated|notified|contacted|alerted|forwarded|passed)\b")   # jlens/build_claims
K = 20


class Prog:
    def __init__(self, name, every=60):
        self.name, self.every, self.t = name, every, 0.0

    def __call__(self, done, total, extra=""):
        if time.time() - self.t > self.every or done == total:
            self.t = time.time()
            log(f"  {self.name}: {done}/{total} {extra}")


def selfcheck(model, tok):
    ids = tok(SPIDER, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        lg = model(input_ids=torch.tensor([ids], device=dev_of(model))).logits[0, -1]
    top1 = tok.decode([int(lg.argmax())])
    log(f"selfcheck: spider prompt -> {top1!r}")
    assert top1.strip() == "8", "base model does not answer the spider prompt: weights or loading are wrong"


def tops(score, k=K):
    return [[l, n, round(v, 6)] for l, n, v in topk_neurons(score, k)]


# ------------------------------------------------------------------------------------------------ gate
def stage_gate(model, tok, hooks, rows, out):
    res = {}
    r = next(x for x in rows if x["cond"] == "nat_false")
    t = r["t"]
    runs = []
    for _ in range(2):
        sc, Ls, _ = attribute(model, hooks, r["ids"], [t], {t: L_logprob(r["claim_tok"])})
        runs.append(([x[:2] for x in topk_neurons(sc[t], K)], Ls[t], sc[t]))
    same = runs[0][0] == runs[1][0]
    same_set = set(map(tuple, runs[0][0])) == set(map(tuple, runs[1][0]))
    res["determinism"] = {"tid": r["tid"], "top20_identical": same, "top20_same_set": same_set,
                          "L": [runs[0][1], runs[1][1]],
                          "max_abs_diff": float((runs[0][2] - runs[1][2]).abs().max()),
                          "top20": runs[0][0]}
    log(f"gate determinism: top-20 identical (ordered) {same}, same set {same_set}; L {runs[0][1]:.5f} / {runs[1][1]:.5f}; "
        f"max |score diff| {res['determinism']['max_abs_diff']:.3g}")
    greedy = []
    for x in rows:
        if x["cond"] not in ("nat_false", "nat_true"):
            continue
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([x["ids"][: x["t"] + 1]], device=dev_of(model))).logits[0, -1].float()
        pr = torch.softmax(lg, -1)
        top = int(lg.argmax())
        greedy.append({"tid": x["tid"], "cond": x["cond"], "greedy_is_claim": top == x["claim_tok"],
                       "p_claim": float(pr[x["claim_tok"]]), "greedy_tok": tok.decode([top])})
    nf = [g for g in greedy if g["cond"] == "nat_false"]
    res["greedy"] = {"nat_false_greedy_is_claim": sum(g["greedy_is_claim"] for g in nf), "n_nat_false": len(nf),
                     "nat_true_greedy_is_claim": sum(g["greedy_is_claim"] for g in greedy if g["cond"] == "nat_true"),
                     "rows": greedy}
    log(f"gate greedy: claim token is greedy at t in {res['greedy']['nat_false_greedy_is_claim']}/{len(nf)} natural "
        f"false episodes (true: {res['greedy']['nat_true_greedy_is_claim']}/20); p_claim "
        f"{[round(g['p_claim'], 3) for g in nf]}")
    res["pass"] = same_set and res["greedy"]["nat_false_greedy_is_claim"] >= 0.9 * len(nf)
    (out / "gate.json").write_text(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------------------------------------ e70
def run_attr(model, hooks, rows, objective_of, positions_of, name, keep_full_at="t"):
    """Attribution for every row. Returns (records, full score matrix at the decision token [n, N] on GPU)."""
    recs, full = [], []
    prog = Prog(name)
    t0 = time.time()
    for i, r in enumerate(rows):
        pos = positions_of(r)
        sc, Ls, lgs = attribute(model, hooks, r["ids"], pos, {p: objective_of(r, p) for p in pos})
        t = r["t"]
        rec = {"tid": r["tid"], "cond": r["cond"], "t": t, "L": {str(p - t): Ls[p] for p in pos},
               "top20": {str(p - t): tops(sc[p]) for p in pos},
               "top100_t": tops(sc[t], 100)}
        if len(pos) > 1:
            st = [x[:2] for x in topk_neurons(sc[t], K)]
            rec["jac"] = {str(p - t): jaccard(map(tuple, st), map(tuple, [x[:2] for x in topk_neurons(sc[p], K)]))
                          for p in pos if p != t}
        lg = lgs[t]
        pr = torch.softmax(lg, -1)
        v, ix = pr.topk(5)
        rec["next_top5"] = [[int(a), round(float(b), 4)] for a, b in zip(ix, v)]
        recs.append(rec)
        full.append(sc[t].flatten().to(torch.float32))
        prog(i + 1, len(rows), f"({len(r['ids'])} tokens, {(time.time() - t0) / (i + 1):.1f}s/episode)")
    return recs, torch.stack(full)


def differential(S, recs, ca, cb, n_boot=2000, seed=0, k=K, d_ff=None):
    """Differential set of condition ca minus cb: top-k by |mean diff|, with bootstrap CIs over episodes."""
    ia = [i for i, r in enumerate(recs) if r["cond"] == ca]
    ib = [i for i, r in enumerate(recs) if r["cond"] == cb]
    A, B = S[ia], S[ib]
    d = A.mean(0) - B.mean(0)
    top = d.abs().topk(max(k, 500)).indices
    g = torch.Generator(device="cpu").manual_seed(seed)
    Asub, Bsub = A[:, top].cpu(), B[:, top].cpu()
    bs = []
    for _ in range(n_boot):
        a = Asub[torch.randint(0, len(ia), (len(ia),), generator=g)].mean(0)
        b = Bsub[torch.randint(0, len(ib), (len(ib),), generator=g)].mean(0)
        bs.append(a - b)
    bs = torch.stack(bs)
    lo, hi = bs.quantile(0.025, 0), bs.quantile(0.975, 0)
    out = []
    for j in range(len(top)):
        f = int(top[j])
        out.append({"nid": nid(f // d_ff, f % d_ff), "layer": f // d_ff, "neuron": f % d_ff, "diff": float(d[f]),
                    "ci": [float(lo[j]), float(hi[j])], "mean_a": float(A[:, f].mean()), "mean_b": float(B[:, f].mean()),
                    "frac_a_top20": None})
    return out, ia, ib


def frequent(recs, cond, n=10, key="0"):
    c = Counter(nid(l, nn) for r in recs if r["cond"] == cond for l, nn, _ in r["top20"][key])
    return [[k, v] for k, v in c.most_common(n)]


def stability(S, recs, ca, cb, label, seed=0):
    ia = [i for i, r in enumerate(recs) if r["cond"] in ca]
    ib = [i for i, r in enumerate(recs) if r["cond"] in cb]
    real, rl = split_half(S[ia], S[ib], K, 20, seed, shuffle=False)
    shuf, sl = split_half(S[ia], S[ib], K, 20, seed + 1, shuffle=True)
    floor = 1 / 39                     # one shared neuron in two top-20s; used if the shuffled level is exactly 0
    ratio = real / max(shuf, floor)
    log(f"stability (b) {label}: split-half Jaccard {real:.3f} vs shuffled {shuf:.3f} -> ratio {ratio:.2f}")
    return {"contrast": label, "n_a": len(ia), "n_b": len(ib), "real": real, "shuffled": shuf, "ratio": ratio,
            "real_splits": rl, "shuffled_splits": sl, "pass_3x": ratio >= 3}


def nearby(recs, conds=None):
    out = {}
    for key in ("-1", "1"):
        v = [r["jac"][key] for r in recs if "jac" in r and (conds is None or r["cond"] in conds)]
        out[key] = {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)}
    return out


def save_candidates(S, recs, diffs, path, d_ff, extra_n=100):
    cand = set()
    for r in recs:
        for l, n, _ in r["top100_t"][:extra_n]:
            cand.add(l * d_ff + n)
    for dl in diffs:
        for x in dl:
            cand.add(x["layer"] * d_ff + x["neuron"])
    cand = sorted(cand)
    idx = torch.tensor(cand, device=S.device)
    np.savez_compressed(path, flat=np.array(cand), d_ff=d_ff, tids=np.array([r["tid"] for r in recs]),
                        conds=np.array([r["cond"] for r in recs]), vals=S[:, idx].cpu().numpy().astype(np.float32))
    return len(cand)


def stage_e70(model, tok, hooks, rows, out, pilot):
    nl, d_ff = shape_of(model)
    if pilot:
        rows = [r for r in rows if r["pilot"]]
    log(f"e70: {len(rows)} transcripts, attribution at t-1, t, t+1")

    def obj(r, p):
        return L_logprob(r["ids"][p + 1] if p + 1 < len(r["ids"]) else r["claim_tok"])
    # t+1 needs the token after the verb: rows carry ids up to t+1 (the verb); extend with the original transcript
    recs, S = run_attr(model, hooks, rows, obj, lambda r: [r["t"] - 1, r["t"], r["t"] + 1], "e70 attribution")
    dump_jsonl(out / "e70_attr.jsonl", recs)
    res = {"n": Counter(r["cond"] for r in recs), "nearby_a": nearby(recs),
           "nearby_a_natural": nearby(recs, ("nat_false", "nat_true"))}
    log(f"stability (a) nearby tokens: Jaccard(t, t-1) {res['nearby_a']['-1']['mean']:.3f}, "
        f"Jaccard(t, t+1) {res['nearby_a']['1']['mean']:.3f}")
    res["L_t"] = {c: float(np.mean([r["L"]["0"] for r in recs if r["cond"] == c])) for c in res["n"]}
    res["stab_b"] = stability(S, recs, ("nat_false",), ("nat_true",), "nat_false vs nat_true")
    res["stab_b_call"] = stability(S, recs, ("nat_true", "call_inserted"), ("nat_false", "call_removed"),
                                   "call in context vs not (exploratory)")
    res["stab_b_lineage"] = stability(S, recs, ("nat_false", "call_inserted"), ("nat_true", "call_removed"),
                                      "false lineage vs true lineage (exploratory)")
    diff, ia, ib = differential(S, recs, "nat_false", "nat_true", d_ff=d_ff)
    # context control: the same neurons' mean score in each condition
    conds = ["nat_false", "nat_true", "call_inserted", "call_removed"]
    ci = {c: [i for i, r in enumerate(recs) if r["cond"] == c] for c in conds}
    for x in diff[:100]:
        f = x["layer"] * d_ff + x["neuron"]
        x["cond_means"] = {c: float(S[ci[c], f].mean()) for c in conds if ci[c]}
        x["frac_top20"] = {c: sum(1 for i in ci[c] if [x["layer"], x["neuron"]] in
                                  [y[:2] for y in recs[i]["top20"]["0"]]) / max(1, len(ci[c])) for c in conds}
    res["diff_top20"] = diff[:K]
    res["frequent"] = {c: frequent(recs, c) for c in conds}
    # paired edit effects on the differential set: does adding the call move false claims toward true ones?
    pair = {}
    for r in rows:
        pair.setdefault(r["claim_id"], {})[r["cond"]] = r["tid"]
    tid_i = {r["tid"]: i for i, r in enumerate(recs)}
    dset = torch.tensor([x["layer"] * d_ff + x["neuron"] for x in diff[:K]], device=S.device)
    sign = torch.tensor([1.0 if x["diff"] > 0 else -1.0 for x in diff[:K]], device=S.device)
    proj = (S[:, dset] * sign).sum(1).cpu().numpy()       # signed sum over the differential set ("false-ness")
    res["diffset_projection"] = {c: [float(proj[i]) for i in ci[c]] for c in conds}
    ed = {"insert_call_on_false": [], "remove_call_on_true": []}
    for cid, p in pair.items():
        if "nat_false" in p and "call_inserted" in p:
            ed["insert_call_on_false"].append(float(proj[tid_i[p["call_inserted"]]] - proj[tid_i[p["nat_false"]]]))
        if "nat_true" in p and "call_removed" in p:
            ed["remove_call_on_true"].append(float(proj[tid_i[p["call_removed"]]] - proj[tid_i[p["nat_true"]]]))
    res["edit_effects_on_projection"] = ed
    res["nat_gap_projection"] = float(np.mean(res["diffset_projection"]["nat_false"]) -
                                      np.mean(res["diffset_projection"]["nat_true"])) if ci["nat_true"] else None
    nc = save_candidates(S, recs, [diff], out / "e70_cand.npz", d_ff)
    res["n_candidates_saved"] = nc
    torch.save({c: S[ci[c]].mean(0).cpu().half() for c in conds if ci[c]}, out / "e70_cond_means.pt")
    stop = not res["stab_b"]["pass_3x"]
    res["stop_rule_fires"] = stop
    (out / "e70_summary.json").write_text(json.dumps(res, indent=1))
    (out / "STOP_RULE.json").write_text(json.dumps({"fires": stop, "real": res["stab_b"]["real"],
                                                    "shuffled": res["stab_b"]["shuffled"],
                                                    "ratio": res["stab_b"]["ratio"], "pilot": pilot}))
    log(f"e70 differential top-5: {[(x['nid'], round(x['diff'], 4)) for x in diff[:5]]}")
    log(f"e70 STOP RULE {'FIRES' if stop else 'passes'} (ratio {res['stab_b']['ratio']:.2f}, pilot {pilot})")
    return res, S, recs, diff


def claim_rate_base(model, tok, hooks, rows, out, n=5, n_new=30):
    """Pilot: claim rate from t with no ablation, 5 samples of 30 tokens at T 0.7 (top-p 0.95, top-k 20)."""
    res = []
    for r in rows:
        if r["cond"] not in ("nat_false", "nat_true"):
            continue
        conts = ablated_continuations(model, hooks, r["ids"], r["t"], [[]] * n, n_new, seed=7)
        texts = [tok.decode(r["ids"][r["pre"] + 1: r["t"] + 1] + c) for c in conts]
        hits = [bool(CLAIM.search(x)) for x in texts]
        res.append({"tid": r["tid"], "cond": r["cond"], "claims": sum(hits), "n": n, "texts": texts})
        log(f"claim rate {r['cond']} {r['tid'][-30:]}: {sum(hits)}/{n}; e.g. {texts[0][:80]!r}")
    dump_jsonl(out / "claim_rate_base.jsonl", res)
    return res


def abl_timing(model, tok, hooks, rows, out):
    """Pilot timing for the main ablation stage: 45 single-position ablations (logits) and 48 sampled rows."""
    from neurons.common import ablated_logits
    nl, d_ff = shape_of(model)
    r = next(x for x in rows if x["cond"] == "nat_false")
    g = torch.Generator().manual_seed(0)
    conds = [[]] + [[(int(torch.randint(0, nl, (1,), generator=g)), int(torch.randint(0, d_ff, (1,), generator=g)), 0.0)
                     for _ in range(5)] for _ in range(44)]
    t0 = time.time()
    lg = ablated_logits(model, hooks, r["ids"], r["t"], conds)
    t1 = time.time()
    pc = torch.log_softmax(lg, -1)[:, r["claim_tok"]]
    conts = ablated_continuations(model, hooks, r["ids"], r["t"], conds[:48] + [[]] * 3, 30, seed=3)
    t2 = time.time()
    res = {"logits_45_s": t1 - t0, "sample_48x30_s": t2 - t1, "logp_claim_none": float(pc[0]),
           "logp_claim_random_range": [float(pc[1:].min()), float(pc[1:].max())],
           "mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None}
    log(f"ablation timing: {res}")
    (out / "abl_timing.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="qwen36")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", nargs="+", default=["gate", "e70"])
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import transformers
    info = {"kind": a.kind, "torch": torch.__version__, "transformers": transformers.__version__,
            "git": os.environ.get("GIT_REV"), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_deterministic()
    tok = load_tokenizer(a.kind)
    model = load_model(a.kind, dev)
    hooks = MLPHooks(model)
    info["shape"] = shape_of(model)
    log(f"info {info}")
    (out / "run_info.json").write_text(json.dumps(info, indent=1))
    if a.kind == "qwen36":
        selfcheck(model, tok)
    rows = load_jsonl(HERE / "data" / "e70.jsonl")
    if a.kind == "tiny36":
        rows = [r for r in rows if r["pilot"]]
    t0 = time.time()
    for st in a.stages:
        if st == "gate":
            stage_gate(model, tok, hooks, rows, out)
        elif st == "e70":
            stage_e70(model, tok, hooks, rows, out, a.pilot)
        elif st == "abltime":
            abl_timing(model, tok, hooks, rows, out)
        elif st == "claimbase":
            claim_rate_base(model, tok, hooks, [r for r in rows if r["pilot"]], out)
        log(f"stage {st} done ({time.time() - t0:.0f}s)")
    log("all done")


if __name__ == "__main__":
    main()
