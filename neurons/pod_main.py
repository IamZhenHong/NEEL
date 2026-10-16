"""Main HF stages after e70 passes its stop rule (one process per model; stages in the order given):

  python -m neurons.pod_main --kind qwen36 --out neurons/results/main --stages e71 e72 select corpus abl70 spec70
  python -m neurons.pod_main --kind qwen14 --out neurons/results/main --stages e73 select corpus

e71    : greedy filter of the vLLM replies (gen/gen_e71), attribution at the last prompt token (L = logit of the paid
         reply's first token minus the refused reply's), stability, differential (sup - peer), frequent neurons.
e72    : thinking-on episodes (gen/gen_e72a, gen/gen_e72c_b*): classes, decision tokens, CoT text; attribution at the
         matched decision token; per-episode top-20 for the predictors; stability; differential per behaviour.
e73    : Qwen3-14B, 20 HC items x {theatrical, plain}; L = log P(refusal opener) at the last prompt token.
select : the neurons to label (DESIGN: differential top-20s, the 10 most frequent per-condition top-20 neurons, the
         e72 per-episode top-20s, and a random pool of 60).
corpus : one pass over the label corpus: corpus mean of every neuron (mean-ablation values) and the top-50 / bottom-50
         contexts (+-12 tokens) of every selected neuron.
abl70  : e70 ablations (top-3 / top-5 differential, zero and corpus mean, 20 layer-matched random 5-sets) at t:
         P(claim token) and the claim rate (5 samples x 30 tokens from t).
spec70 : specificity: the same ablations on 100 benign items."""
import argparse
import gzip
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from neurons.common import (HERE, L_logitdiff, L_logprob, L_logsumprob, MLPHooks, ablated_continuations, ablated_logits,
                            dev_of, dump_jsonl, load_jsonl, load_model, load_tokenizer, log, nid, set_deterministic,
                            shape_of, topk_neurons)
from neurons.pod_attr import (CLAIM, K, Prog, differential, frequent, nearby, run_attr, save_candidates, selfcheck,
                              stability, tops)

TOOL_CALL, THINK_END = 248058, 248069
PAY_FN = "<function=issue_refund>"
N_PER_CLASS = 20


CK = {"qwen36": "qwen36", "tiny36": "qwen36", "qwen14": "qwen14", "qwen06": "qwen14"}   # corpus / file kind


def random_pool(nl, d_ff, n=60, seed=7300):
    rng = random.Random(seed)
    return sorted(rng.sample(range(nl * d_ff), n))


def read_gz(p):
    with gzip.open(p, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


# ------------------------------------------------------------------------------------------------ generic experiment
def run_experiment(name, model, hooks, rows, ca, cb, out, d_ff, pool, objective_at_t):
    """rows: {tid, cond, ids, t}; ids must reach t+2. Attribution at t-1, t, t+1 (t: the experiment's L; t+-1: log-prob
    of the actual next token)."""
    def obj(r, p):
        return objective_at_t(r) if p == r["t"] else L_logprob(r["ids"][p + 1])
    recs, S = run_attr(model, hooks, rows, obj, lambda r: [r["t"] - 1, r["t"], r["t"] + 1], f"{name} attribution")
    dump_jsonl(out / f"{name}_attr.jsonl", recs)
    res = {"n": Counter(r["cond"] for r in recs), "nearby_a": nearby(recs),
           "L_t": {c: float(np.mean([r["L"]["0"] for r in recs if r["cond"] == c])) for c in (ca, cb)}}
    log(f"{name} stability (a): Jaccard(t, t-1) {res['nearby_a']['-1']['mean']:.3f}, (t, t+1) "
        f"{res['nearby_a']['1']['mean']:.3f}; mean L at t {res['L_t']}")
    res["stab_b"] = stability(S, recs, (ca,), (cb,), f"{name}: {ca} vs {cb}")
    diff, ia, ib = differential(S, recs, ca, cb, d_ff=d_ff)
    for x in diff[:100]:
        f = x["layer"] * d_ff + x["neuron"]
        x["frac_top20"] = {c: sum(1 for i, r in enumerate(recs) if r["cond"] == c and [x["layer"], x["neuron"]] in
                                  [y[:2] for y in r["top20"]["0"]]) / max(1, res["n"][c]) for c in (ca, cb)}
    res["diff_top20"] = diff[:K]
    res["frequent"] = {c: frequent(recs, c) for c in (ca, cb)}
    idx = torch.tensor(pool, device=S.device)
    pool_scores = S[:, idx].cpu().numpy()
    np.savez_compressed(out / f"{name}_pool_scores.npz", flat=np.array(pool), tids=np.array([r["tid"] for r in recs]),
                        vals=pool_scores.astype(np.float32))
    res["n_candidates_saved"] = save_candidates(S, recs, [diff], out / f"{name}_cand.npz", d_ff)
    (out / f"{name}_summary.json").write_text(json.dumps(res, indent=1))
    log(f"{name} differential top-5: {[(x['nid'], round(x['diff'], 4)) for x in diff[:5]]}")
    return res, recs


# ------------------------------------------------------------------------------------------------ e71
def e71_rows(tok, gen_dir, name="gen_e71.jsonl.gz"):
    eps = read_gz(gen_dir / name)
    rows, stats = [], Counter()
    for e in eps:
        o = e["out_ids"]
        txt = e["text"]
        paid = PAY_FN in txt
        first = o[0]
        kind = ("paid" if paid and first == TOOL_CALL else "refused" if (not paid and first != TOOL_CALL) else "other")
        cond = e["job"]["cond"]
        ok = (cond == "sup" and kind == "paid") or (cond == "peer" and kind == "refused")
        stats[(cond, kind)] += 1
        if ok:
            rows.append({"tid": e["key"], "cond": cond, "ids": e["prompt_ids"] + o[:2], "t": len(e["prompt_ids"]) - 1,
                         "first_tok": first, "text": txt[:400], "body_id": e["job"]["body_id"]})
    return rows, {f"{c}|{k}": v for (c, k), v in stats.items()}


def stage_e71(model, tok, hooks, out, d_ff, pool):
    rows, stats = e71_rows(tok, out / "gen")
    # D1: reserve bodies only if a condition has < 20 kept episodes from the frozen 48 (added in body order to 20)
    rpath = out / "gen" / "gen_e71r.jsonl.gz"
    used_reserve = {}
    for cond in ("sup", "peer"):
        have = sum(r["cond"] == cond for r in rows)
        if have < N_PER_CLASS and rpath.exists():
            extra, st_r = e71_rows(tok, out / "gen", "gen_e71r.jsonl.gz")
            extra = sorted([r for r in extra if r["cond"] == cond], key=lambda r: r["body_id"])[: N_PER_CLASS - have]
            rows += extra
            used_reserve[cond] = len(extra)
            stats.update({f"reserve|{k}": v for k, v in st_r.items()})
    ref = Counter(r["first_tok"] for r in rows if r["cond"] == "peer").most_common(1)[0][0]
    pay = TOOL_CALL
    log(f"e71 greedy filter {stats}; kept sup {sum(r['cond'] == 'sup' for r in rows)}, peer "
        f"{sum(r['cond'] == 'peer' for r in rows)}; L = logit({tok.decode([pay])!r}) - logit({tok.decode([ref])!r})")
    # HF greedy at t agrees with vLLM?
    agree = 0
    for r in rows:
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([r["ids"][: r["t"] + 1]], device=dev_of(model)),
                       logits_to_keep=1).logits[0, -1]
        agree += int(lg.argmax()) == r["first_tok"]
    res, recs = run_experiment("e71", model, hooks, rows, "sup", "peer", out, d_ff, pool, lambda r: L_logitdiff(pay, ref))
    res.update({"filter": stats, "hf_vllm_first_token_agree": agree, "n_rows": len(rows), "pay_tok": pay, "ref_tok": ref,
                "pay_str": tok.decode([pay]), "ref_str": tok.decode([ref]), "used_reserve": used_reserve})
    (out / "e71_summary.json").write_text(json.dumps(res, indent=1))
    n_ok = min(sum(r["cond"] == c for r in rows) for c in ("sup", "peer")) >= N_PER_CLASS
    fires = (not res["stab_b"]["pass_3x"]) or not n_ok
    (out / "STOP_RULE_e71.json").write_text(json.dumps({"fires": fires, "real": res["stab_b"]["real"],
                                                        "shuffled": res["stab_b"]["shuffled"],
                                                        "ratio": res["stab_b"]["ratio"], "n_ok": n_ok,
                                                        "nearby_a": res["nearby_a"]}))
    log(f"e71 STOP RULE {'FIRES' if fires else 'passes'} (ratio {res['stab_b']['ratio']:.2f}; >= 20 per condition {n_ok})")
    dump_jsonl(out / "e71_rows.jsonl", [{k: v for k, v in r.items() if k != "ids"} | {"n_ids": len(r["ids"])}
                                       for r in rows])
    return res


# ------------------------------------------------------------------------------------------------ e72
def think_split(tok, out_ids):
    """(index of </think> in out_ids, index of the first content token) or (None, None)."""
    if THINK_END not in out_ids:
        return None, None
    k = out_ids.index(THINK_END)
    c = k + 1
    while c < len(out_ids) and tok.decode([out_ids[c]]).strip() == "":
        c += 1
    return k, c


def e72a_rows(tok, gen_dir):
    eps = read_gz(gen_dir / "gen_e72a.jsonl.gz")
    if (gen_dir / "gen_e72at.jsonl.gz").exists():          # E2 top-up samples
        eps += read_gz(gen_dir / "gen_e72at.jsonl.gz")
    cands, stats = {"paid": [], "refused": []}, Counter()
    for e in eps:
        o = e["out_ids"]
        k, c = think_split(tok, o)
        if k is None or c >= len(o):
            stats["unclosed"] += 1
            continue
        content = tok.decode(o[c:])
        paid = PAY_FN in content
        first = o[c]
        kind = "paid" if (paid and first == TOOL_CALL) else "refused" if (not paid and first != TOOL_CALL) else "other"
        stats[(e["job"]["cond"], kind)] += 1
        if kind == "other":
            continue
        p = len(e["prompt_ids"])
        cot = tok.decode(o[:k]).replace("<think>", "").strip()
        cands[kind].append({"tid": e["key"], "cond": kind, "ids": e["prompt_ids"] + o[: c + 2], "t": p + c - 1,
                            "first_tok": first, "cot": cot, "content": content[:600], "requester": e["job"]["cond"],
                            "pid": e["job"]["pid"]})
    rows = []
    rng = random.Random(72)
    n = min(N_PER_CLASS, *(len(v) for v in cands.values()))     # E2: balanced classes
    for kind, lst in cands.items():
        rng.shuffle(lst)          # one sample per prompt first, then second samples
        seen, first_pass, second = set(), [], []
        for r in lst:
            (second if r["pid"] in seen else first_pass).append(r)
            seen.add(r["pid"])
        rows += (first_pass + second)[:n]
    return rows, {("|".join(x) if isinstance(x, tuple) else x): v for x, v in stats.items()}


def char_to_tok(tok, ids, char_pos):
    """Smallest k such that decode(ids[:k+1]) covers char_pos (binary search)."""
    lo, hi = 0, len(ids) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tok.decode(ids[: mid + 1])) > char_pos:
            hi = mid
        else:
            lo = mid + 1
    return lo


def e72c_rows(tok, gen_dir):
    eps = []
    for p in sorted(gen_dir.glob("gen_e72c_b*.jsonl.gz")):
        eps += read_gz(p)
    cands, stats = {"false": [], "true": []}, Counter()
    for e in eps:
        if not e.get("claims"):
            stats["no_claim"] += 1
            continue
        ti, lab = e["claims"][0]
        stats[lab] += 1
        raw = e["raw"][ti]
        o = raw["out_ids"]
        k, c = think_split(tok, o)
        if k is None:
            stats["unclosed"] += 1
            continue
        content_text = tok.decode(o[c:])
        m = None
        for mt in CLAIM.finditer(content_text):
            pre = content_text[: mt.start()]
            if pre.count("<tool_call>") <= pre.count("</tool_call>"):
                m = mt
                break
        if m is None:
            stats["claim_not_found"] += 1
            continue
        base = len(tok.decode(o[:c]))
        vk = char_to_tok(tok, o, base + m.start(1))
        verb_tok = tok.decode([o[vk]]).strip()
        if not verb_tok or not m.group(1).startswith(verb_tok):
            stats["verb_misaligned"] += 1
            continue
        p = len(raw["prompt_ids"])
        # CoT in context at the decision: reasoning of this user turn's steps up to and including the claiming step
        msgs = e["messages"]
        a_idx = [i for i, mm in enumerate(msgs) if mm["role"] == "assistant"]
        mi = a_idx[ti]
        lu = max(i for i in range(mi) if msgs[i]["role"] == "user")
        cot = "\n\n".join(msgs[i].get("reasoning_content", "") for i in range(lu + 1, mi + 1)
                          if msgs[i]["role"] == "assistant" and msgs[i].get("reasoning_content"))
        cands[lab].append({"tid": e["key"], "cond": lab, "ids": raw["prompt_ids"] + o[: vk + 2], "t": p + vk - 1,
                           "claim_tok": o[vk], "cot": cot, "content": content_text[max(0, m.start() - 300): m.end() + 100],
                           "turn": ti, "case": e["job"]["case"], "persona": e["job"]["cond"]})
    rows = []
    rng = random.Random(73)
    for lab, lst in cands.items():
        rng.shuffle(lst)
        rows += lst[:N_PER_CLASS]
    return rows, dict(stats), {k: len(v) for k, v in cands.items()}


def stage_e72(model, tok, hooks, out, d_ff, pool, e71_res, authority_only=False):
    res = {}
    rows_a, st_a = e72a_rows(tok, out / "gen")
    na = Counter(r["cond"] for r in rows_a)
    log(f"e72 authority: classes {st_a}; kept {dict(na)}")
    ref = e71_res["ref_tok"] if e71_res else 40
    res["authority"] = {"stats": st_a, "kept": na}
    if min(na.get("paid", 0), na.get("refused", 0)) >= 8:
        r1, recs = run_experiment("e72a", model, hooks, rows_a, "paid", "refused", out, d_ff, pool,
                                  lambda r: L_logitdiff(TOOL_CALL, ref))
        res["authority"]["summary"] = "e72a_summary.json"
    else:
        res["authority"]["dropped"] = True
    dump_jsonl(out / "e72a_rows.jsonl", [{k: v for k, v in r.items() if k != "ids"} for r in rows_a])
    if authority_only:              # D1: the false-claim half of e72 is not run
        (out / "e72_summary.json").write_text(json.dumps(res, indent=1, default=str))
        return res
    rows_c, st_c, ncand = e72c_rows(tok, out / "gen")
    nc = Counter(r["cond"] for r in rows_c)
    log(f"e72 false claim: {st_c}; candidates {ncand}; kept {dict(nc)}")
    res["claim"] = {"stats": st_c, "candidates": ncand, "kept": nc}
    if min(nc.get("false", 0), nc.get("true", 0)) >= 8:
        run_experiment("e72c", model, hooks, rows_c, "false", "true", out, d_ff, pool,
                       lambda r: L_logprob(r["claim_tok"]))
    else:
        res["claim"]["dropped"] = True
    dump_jsonl(out / "e72c_rows.jsonl", [{k: v for k, v in r.items() if k != "ids"} for r in rows_c])
    (out / "e72_summary.json").write_text(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------------------------------------ e73
def stage_e73(model, tok, hooks, out, d_ff, pool):
    items = load_jsonl(HERE / "data" / "e73.jsonl")
    meta = json.loads((HERE / "data" / "e73_meta.json").read_text())
    outs = {}
    for l in open(HERE / "data" / "e73_next.jsonl") if (HERE / "data" / "e73_next.jsonl").exists() else []:
        o = json.loads(l)
        outs[(o["item"], o["cond"])] = o["next2"]
    rows = []
    for it in items:
        for c in ("theatrical", "plain"):
            ids = it["ids"][c]
            nxt = outs.get((it["item"], c))
            if nxt is None:      # greedy next two tokens from the model itself
                with torch.no_grad():
                    a = int(model(input_ids=torch.tensor([ids], device=dev_of(model)), logits_to_keep=1).logits[0, -1].argmax())
                    b = int(model(input_ids=torch.tensor([ids + [a]], device=dev_of(model)), logits_to_keep=1).logits[0, -1].argmax())
                nxt = [a, b]
            rows.append({"tid": f"e73|{it['item']}|{c}", "cond": c, "ids": ids + nxt, "t": len(ids) - 1, "item": it["item"],
                         "greedy_first": tok.decode([nxt[0]])})
    res, recs = run_experiment("e73", model, hooks, rows, "theatrical", "plain", out, d_ff, pool,
                               lambda r: L_logsumprob(meta["openers"]))
    res["greedy_first"] = Counter((r["cond"], r["greedy_first"]) for r in rows).most_common()
    res["openers"] = meta["opener_strings"]
    (out / "e73_summary.json").write_text(json.dumps(res, indent=1, default=str))
    dump_jsonl(out / "e73_rows.jsonl", [{k: v for k, v in r.items() if k != "ids"} for r in rows])
    return res


# ------------------------------------------------------------------------------------------------ select
def stage_select(out, kind, nl, d_ff, pool):
    sets = {"random": [nid(f // d_ff, f % d_ff) for f in pool]}
    exps = ["e70", "e71", "e72a", "e72c"] if CK[kind] == "qwen36" else ["e73"]
    for e in exps:
        p = out / f"{e}_summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        sets[f"{e}_diff"] = [x["nid"] for x in s["diff_top20"]]
        for c, lst in s["frequent"].items():
            if e == "e70" and c not in ("nat_false", "nat_true"):
                continue
            sets[f"{e}_freq_{c}"] = [k for k, _ in lst[:10]]
        if e in ("e72a", "e72c"):
            ep = set()
            for r in load_jsonl(out / f"{e}_attr.jsonl"):
                ep |= {nid(l, n) for l, n, _ in r["top20"]["0"]}
            sets[f"{e}_episode_top20"] = sorted(ep)
    union = sorted({x for v in sets.values() for x in v})
    res = {"kind": kind, "sets": sets, "union": union, "n_union": len(union),
           "n_by_set": {k: len(v) for k, v in sets.items()}}
    (out / f"label_neurons_{CK[kind]}.json").write_text(json.dumps(res, indent=1))
    log(f"select ({kind}): {len(union)} neurons to label; {res['n_by_set']}")
    return union


# ------------------------------------------------------------------------------------------------ corpus
def stage_corpus(model, tok, hooks, out, kind, union, nl, d_ff, n_ctx=50, win=12, limit=None):
    from neurons.common import parse_nid
    docs = load_jsonl(HERE / "data" / f"corpus_{CK[kind]}.jsonl")[:limit]
    dev = dev_of(model)
    sel = [parse_nid(x) for x in union]
    by_layer = {}
    for j, (l, n) in enumerate(sel):
        by_layer.setdefault(l, ([], []))
        by_layer[l][0].append(n)
        by_layer[l][1].append(j)
    by_layer = {l: (torch.tensor(a, device=dev), torch.tensor(b, device=dev)) for l, (a, b) in by_layer.items()}
    sums = torch.zeros(nl, d_ff, dtype=torch.float64, device=dev)
    sq = torch.zeros(nl, d_ff, dtype=torch.float64, device=dev)
    state = {"rf": 0, "cur": None}
    vals, meta = [], []
    total = 0

    def rec(i, x):
        rf = state["rf"]
        xs = x[0, rf:].float()
        sums[i] += xs.sum(0).double()
        sq[i] += (xs * xs).sum(0).double()
        if i in by_layer:
            idx, cols = by_layer[i]
            state["cur"][:, cols] = xs[:, idx].half()
    hooks.rec = rec
    prog = Prog("corpus")
    t0 = time.time()
    try:
        for di, d in enumerate(docs):
            ids = d["ids"]
            rf = max(d["rec_from"], 1)
            if rf >= len(ids):
                continue
            state["rf"] = rf
            state["cur"] = torch.zeros(len(ids) - rf, len(sel), dtype=torch.float16, device=dev)
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=1)
            vals.append(state["cur"])
            meta += [(di, p) for p in range(rf, len(ids))]
            total += len(ids) - rf
            prog(di + 1, len(docs), f"{total} recorded tokens, {time.time() - t0:.0f}s")
    finally:
        hooks.rec = None
    mean = (sums / total).float()
    std = (sq / total - (sums / total) ** 2).clamp_min(0).sqrt().float()
    torch.save({"mean": mean.cpu(), "std": std.cpu(), "n_tokens": total}, out / f"corpus_stats_{CK[kind]}.pt")
    A = torch.cat(vals)                       # [N, n_sel] fp16
    meta = np.array(meta)
    log(f"corpus: {total} tokens, A {tuple(A.shape)}; extracting contexts")

    def ctx(di, p):
        ids = docs[di]["ids"]
        a, b = max(0, p - win), min(len(ids), p + win + 1)
        s = lambda xs: tok.decode(xs).replace("\n", "⏎")  # noqa: E731
        return s(ids[a:p]) + "«" + s([ids[p]]) + "»" + s(ids[p + 1:b])

    def pick(col, sign):
        v = A[:, col].float() * sign
        top = v.topk(min(2000, len(v))).indices.cpu().numpy()
        chosen = []
        for r in top:
            di, p = meta[r]
            if any(cd == di and abs(cp - p) <= win for cd, cp, _ in chosen):
                continue
            chosen.append((int(di), int(p), float(A[r, col])))
            if len(chosen) >= n_ctx:
                break
        return [[round(val, 4), ctx(di, p), docs[di]["src"]] for di, p, val in chosen]
    rows = []
    for j, x in enumerate(union):
        l, n = sel[j]
        rows.append({"nid": x, "layer": l, "neuron": n, "mean": float(mean[l, n]), "std": float(std[l, n]),
                     "frac_pos": float((A[:, j] > 0).float().mean()), "pos": pick(j, 1.0), "neg": pick(j, -1.0)})
        if j % 100 == 0:
            log(f"  contexts {j}/{len(union)}")
    dump_jsonl(out / f"label_contexts_{CK[kind]}.jsonl", rows)
    log(f"corpus done: {len(rows)} neurons with contexts ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------------------------------------ ablations
def ablation_sets(diff, mean, nl, d_ff, n_draws=20, seed=7070, sizes=(3, 5)):
    """Named ablation conditions (lists of (layer, neuron, value)) from a differential list ordered by |diff|."""
    from neurons.common import parse_nid
    top = [parse_nid(x["nid"]) for x in diff]
    excl = set(top)
    conds = {"none": []}
    for k in sizes:
        conds[f"zero{k}"] = [(l, n, 0.0) for l, n in top[:k]]
        conds[f"mean{k}"] = [(l, n, float(mean[l, n])) for l, n in top[:k]]
    rng = random.Random(seed)
    k = max(sizes)
    for d in range(n_draws):
        draw = []
        for l, _ in top[:k]:
            while True:
                n = rng.randrange(d_ff)
                if (l, n) not in excl and (l, n) not in [(a, b) for a, b, _ in draw]:
                    break
            draw.append((l, n, 0.0))
        conds[f"rzero{k}_{d}"] = draw
        conds[f"rmean{k}_{d}"] = [(l, n, float(mean[l, n])) for l, n, _ in draw]
    return conds


def stage_abl70(model, tok, hooks, out, nl, d_ff):
    s = json.loads((out / "e70_summary.json").read_text())
    mean = torch.load(out / "corpus_stats_qwen36.pt")["mean"]
    conds = ablation_sets(s["diff_top20"], mean, nl, d_ff)
    names = list(conds)
    rows = [r for r in load_jsonl(HERE / "data" / "e70.jsonl") if r["cond"] in ("nat_false", "nat_true")]
    rate_names = ["none", "zero3", "zero5", "mean3", "mean5"] + [f"rzero5_{d}" for d in range(20)]
    res = []
    prog = Prog("abl70")
    for i, r in enumerate(rows):
        lg = ablated_logits(model, hooks, r["ids"], r["t"], [conds[n] for n in names], chunk=48)
        lp = torch.log_softmax(lg, -1)[:, r["claim_tok"]]
        rec = {"tid": r["tid"], "cond": r["cond"], "logp_claim": {n: float(v) for n, v in zip(names, lp)}}
        torch.cuda.empty_cache()
        rows_c = [conds[n] for n in rate_names for _ in range(5)]
        conts = ablated_continuations(model, hooks, r["ids"], r["t"], rows_c, 30, seed=700 + i, chunk=32)
        clause = r["ids"][r["pre"] + 1: r["t"] + 1]
        hits = [bool(CLAIM.search(tok.decode(clause + c))) for c in conts]
        rec["claims"] = {n: sum(hits[5 * j: 5 * j + 5]) for j, n in enumerate(rate_names)}
        rec["examples"] = {n: tok.decode(clause + conts[5 * j])[:120] for j, n in enumerate(rate_names[:5])}
        res.append(rec)
        torch.cuda.empty_cache()
        prog(i + 1, len(rows), f"none {rec['claims']['none']}/5 zero5 {rec['claims']['zero5']}/5")
    dump_jsonl(out / "abl70.jsonl", res)
    (out / "abl70_conds.json").write_text(json.dumps({n: [[l, nn, v] for l, nn, v in c] for n, c in conds.items()}))


def stage_spec(model, tok, hooks, out, kind, conds_by_set, tag):
    """Benign specificity: TR letter accuracy (logits at the last prompt token) and short-QA accuracy (greedy 8
    tokens), each ablation applied at the last prompt token."""
    items = load_jsonl(HERE / "data" / f"spec_{CK[kind]}.jsonl")
    names = list(conds_by_set)
    acc = {n: [] for n in names}
    for it in items:
        ids, t = it["ids"], len(it["ids"]) - 1
        if it["kind"] == "letter":
            lg = ablated_logits(model, hooks, ids, t, [conds_by_set[n] for n in names])
            for j, n in enumerate(names):
                acc[n].append(int(int(lg[j, it["letters"]].argmax()) == it["answer"]))
        else:
            conts = ablated_continuations(model, hooks, ids, t, [conds_by_set[n] for n in names], 8, temperature=0)
            for j, n in enumerate(names):
                txt = tok.decode(conts[j]).lower()
                acc[n].append(int(any(a.lower() in txt for a in it["aliases"])))
    res = {n: {"acc": float(np.mean(v)), "acc_letter": float(np.mean(v[:50])), "acc_qa": float(np.mean(v[50:])),
               "delta_pts": 100 * (float(np.mean(v)) - float(np.mean(acc["none"])))} for n, v in acc.items()}
    (out / f"spec_{tag}.json").write_text(json.dumps(res, indent=1))
    log(f"specificity {tag}: " + ", ".join(f"{n} {r['acc']:.3f} ({r['delta_pts']:+.1f} pts)" for n, r in res.items()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="qwen36")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", nargs="+", required=True)
    ap.add_argument("--test-limit", type=int, default=None, help="local tests: first N corpus docs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    set_deterministic()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(a.kind)
    model = load_model(a.kind, dev)
    hooks = MLPHooks(model)
    nl, d_ff = shape_of(model)
    if a.kind == "qwen36":
        selfcheck(model, tok)
    if a.kind == "qwen06":            # local test of the 14B path: Qwen3-0.6B shares the Qwen3 tokenizer
        pass
    pool = random_pool(nl, d_ff)
    e71_res = None
    t0 = time.time()
    for st in a.stages:
        if st == "e71":
            e71_res = stage_e71(model, tok, hooks, out, d_ff, pool)
        elif st == "e72":
            if e71_res is None and (out / "e71_summary.json").exists():
                e71_res = json.loads((out / "e71_summary.json").read_text())
            stage_e72(model, tok, hooks, out, d_ff, pool, e71_res)
        elif st == "e72a":
            if e71_res is None and (out / "e71_summary.json").exists():
                e71_res = json.loads((out / "e71_summary.json").read_text())
            stage_e72(model, tok, hooks, out, d_ff, pool, e71_res, authority_only=True)
        elif st == "e73":
            stage_e73(model, tok, hooks, out, d_ff, pool)
        elif st == "select":
            union = stage_select(out, a.kind, nl, d_ff, pool)
        elif st == "corpus":
            union = json.loads((out / f"label_neurons_{CK[a.kind]}.json").read_text())["union"]
            stage_corpus(model, tok, hooks, out, a.kind, union, nl, d_ff, limit=a.test_limit)
        elif st == "abl70":
            stage_abl70(model, tok, hooks, out, nl, d_ff)
        elif st == "spec70":
            s = json.loads((out / "e70_summary.json").read_text())
            mean = torch.load(out / "corpus_stats_qwen36.pt")["mean"]
            c = ablation_sets(s["diff_top20"], mean, nl, d_ff, n_draws=1)
            stage_spec(model, tok, hooks, out, a.kind, {k: c[k] for k in ("none", "zero3", "zero5", "mean3", "mean5",
                                                                        "rzero5_0")}, "e70")
        log(f"stage {st} done ({time.time() - t0:.0f}s)")
    log("all done")


if __name__ == "__main__":
    main()
