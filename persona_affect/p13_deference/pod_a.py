"""PA-P13 pod A, HF phase (plan persona_affect/plans/P13.md, Task 4): directions, gates, base log-odds and the alpha
pilot, from what this pod's own vLLM phase wrote (pod_a.sh runs persona_affect/common/pod_gen.sh first). It reads only
VLLM_OUT and the alpha jobs; results/ is never shipped.

  /root/venv/bin/python -m persona_affect.p13_deference.pod_a --model gemma|q27 --vllm-out OUT/vllm \\
      --alpha persona_affect/p13_deference/jobs/alpha_$M.jsonl --out OUT/hf --deadline EPOCH

Steps, each output written as soon as it exists (ruling 6). Layer L = the output of decoder layer L, on GRID[m].
  1. Persona reply means: 3 traits x 2 poles x 5 framings x 40 questions, replies teacher-forced with their system
     prompt, mean over the reply tokens.
  2. Directions unit(mean pos - mean neg) per trait and layer, and the in-distribution 5-fold CV AUROC by question
     (folds q mod 5) -> dirs_grid.npz ({def,hum,opt}_L{L}), id_auroc.json.
  3. Naturalistic means (no system prompt, reply tokens): the 24 NATURAL items' D and V replies, and on the 12 test
     items the mention control (crossed) and the target binding (binding) -> nat.npz (fp16).
  4. AUROC tables (D vs V) per layer and direction: dev, test, dev d', mention, binding, and the test-item floors
     (random 95th percentile of max(a, 1 - a) over 200 directions; shuffled 95th percentile over 20) -> auroc.json.
  5. L* = layer_choice over the dev AUROCs; sigma = SD of h_L*.u_def over the reply tokens of the 40 neutral replies;
     the shuffled control shuffled_units(P, N, 1, seed 20261022)[0] at L* -> "shuf" in dirs_grid.npz; summary.json.
  6. Refusal: kept pairs (refusal_kept), activations at the last prompt position, u_ref(L) from kept pairs 0-23, L_ref
     = argmax of the held-out AUROC on kept pairs 24-31 -> ref_L{L} in dirs_grid.npz; summary.json.
  7. Base LO per ctx and rung from that rung's modal prefixes (modal_prefix; SIDE from the legit episodes) ->
     prefixes.json, lo_base.jsonl (with the top-5 first tokens after the generation prompt).
  8. Alpha pilot, 10 arms x 16 items, additions at L*, with the unhooked NLL of each turn -> alpha_pilot.jsonl. Skipped
     (alpha_skipped: true in summary.json) when fewer than 10 minutes remain before the deadline, or cut (also
     alpha_skipped) when the next arm would pass it; calibrate then sets k* = 1, flagged.
  9. Per-response persona means at L* in fp16 -> resp_means.npz.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from persona_affect.common import util
from persona_affect.p13_deference import cases as K
from persona_affect.p13_deference import hf as H
from persona_affect.p13_deference import jobs as J
from persona_affect.p13_deference.hooks import (SEED, Arm, auroc, random_floor, shuffle_floor, shuffled_units,
                                                split_continuations)
from persona_affect.p13_deference.items import NATURAL, REFUSAL_RE, binding, crossed
from vigilante.cases import authority as A
from vigilante.src.episode import init_episode

GRID = {"gemma": list(range(12, 49, 4)), "q27": list(range(12, 53, 4))}
SHORT = {"deference": "def", "humour": "hum", "optimism": "opt"}
N_FOLDS = 5
N_REF_TRAIN = 24          # refusal pairs 0-23 build u_ref, 24-31 choose L_ref
MIN_REF_PAIRS = 12        # fewer kept pairs: the refusal direction is flagged unreliable (and still run)
ALPHA_MIN_LEFT = 600      # seconds before the deadline below which the alpha pilot is skipped
MENTION_BAR, BINDING_BAR = 0.65, 0.70


def now() -> float:
    return time.time()


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    tmp.replace(path)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _log(msg: str) -> None:
    print(f"[pod A] {msg}", flush=True)


# ---------------------------------------------------------------- pure rules (tested)

def _task_calls(rec: dict) -> list[dict]:
    return [c for c in rec["calls"] if c.get("phase", "task") == "task"]


def modal_prefix(recs: list[dict], cid: str, side: bool) -> list[list] | None:
    """The modal pre-decision call sequence of case cid among `recs` (records of other cases are ignored).

    side False: the calls before the first money or request_approval call, over the episodes that made one.
    side True: the calls up to and including request_approval, over the episodes that asked before paying; None if
               none did (that case then has no SIDE prefix).
    The most common sequence of tool names wins (ties: the one seen first), with the arguments of the first episode
    that has it. Episodes calling a tool the case does not have are left out (the prefix must replay through the case).
    """
    money = K.MONEY[cid]
    tools = {t["function"]["name"] for t in A._base(cid).tools}
    seqs = []
    for r in recs:
        if r["job"].get("cid") != cid:
            continue
        pre, decided = [], False
        for c in _task_calls(r):
            if c["name"] == K.COMPLY:
                if side:
                    pre.append([c["name"], c["args"]])
                decided = True
                break
            if c["name"] == money:
                decided = not side
                break
            pre.append([c["name"], c["args"]])
        if decided and all(name in tools for name, _ in pre):
            seqs.append(pre)
    if not seqs:
        return None if side else []
    names = [tuple(n for n, _ in s) for s in seqs]
    counts = Counter(names)
    best = max(counts.values())
    pick = next(i for i, t in enumerate(names) if counts[t] == best)
    return copy.deepcopy(seqs[pick])


def layer_choice(dev: dict[int, tuple[float, float]]) -> int:
    """L*: the highest dev AUROC; ties go to the larger dev d', then to the lower layer."""
    return max(dev, key=lambda L: (dev[L][0], dev[L][1], -L))


def refusal_kept(recs: list[dict]) -> list[int]:
    """Pairs whose harmful request was refused and whose benign twin was not (REFUSAL_RE on the first 200 characters
    of each 64-token reply), in pair order."""
    by: dict = defaultdict(dict)
    for r in recs:
        by[r["job"]["pair"]][r["job"]["pole"]] = r["turns"][0]["content"] if r["turns"] else ""
    return sorted(p for p, d in by.items() if "harm" in d and "benign" in d
                  and re.search(REFUSAL_RE, d["harm"][:200]) and not re.search(REFUSAL_RE, d["benign"][:200]))


def cv_auroc(P: np.ndarray, N: np.ndarray, qp: np.ndarray, qn: np.ndarray, folds: int = N_FOLDS) -> float:
    """In-distribution AUROC, 5-fold by question (fold = q mod 5): the direction from the other folds' replies, scored
    on the held-out fold's; the mean over folds."""
    out = []
    for f in range(folds):
        trp, trn, tep, ten = qp % folds != f, qn % folds != f, qp % folds == f, qn % folds == f
        if not (trp.any() and trn.any() and tep.any() and ten.any()):
            continue
        u = _unit(P[trp].mean(0) - N[trn].mean(0))
        out.append(auroc(P[tep] @ u, N[ten] @ u))
    return float(np.mean(out)) if out else float("nan")


def dprime(a: np.ndarray, b: np.ndarray) -> float:
    sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else 0.0
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


# ---------------------------------------------------------------- steps

def load_vllm(vllm_out: Path) -> dict[str, list[dict]]:
    return {g: _jsonl(vllm_out / f"episodes_{g}.jsonl") for g in ("chat", "agent", "l16")
            if (vllm_out / f"episodes_{g}.jsonl").exists()}


def reply_spans(hf, recs: list[dict]) -> tuple[list[str], list[tuple[int, int]], list[int], list[int]]:
    """Teacher-forced reply texts (the rendered prompt, system prompt included, plus the reply) with each reply's token
    span, and the indices kept (an empty reply has no tokens to average). Also returns each kept reply's first token."""
    texts, spans, keep, starts = [], [], [], []
    for i, r in enumerate(recs):
        reply = r["turns"][0]["content"] if r["turns"] else ""
        if not reply.strip():
            continue
        prompt = hf.render(r["messages"][:-1], None)
        full, n = split_continuations(hf.ids, prompt, [reply])
        if len(full[0]) <= n:
            continue
        texts.append(prompt + reply)
        spans.append((n, len(full[0])))
        keep.append(i)
        starts.append(n)
    return texts, spans, keep, starts


def persona_means(hf, recs: list[dict], grid: list[int]) -> tuple[list[dict], np.ndarray]:
    texts, spans, keep, _ = reply_spans(hf, recs)
    caps = hf.capture(texts, spans, grid)
    M = np.stack([caps[L] for L in grid], axis=1)                         # [N, G, D]
    meta = [{k: recs[i]["job"][k] for k in ("key", "trait", "pole", "framing", "q")} for i in keep]
    return meta, M


def persona_directions(meta: list[dict], M: np.ndarray, grid: list[int]):
    dirs, norms, cv = {}, {}, {}
    q = np.array([r["q"] for r in meta])
    for trait, s in SHORT.items():
        pos = np.array([r["trait"] == trait and r["pole"] == "pos" for r in meta])
        neg = np.array([r["trait"] == trait and r["pole"] == "neg" for r in meta])
        cv[s] = {}
        for g, L in enumerate(grid):
            v = M[pos, g].mean(0) - M[neg, g].mean(0)
            dirs[f"{s}_L{L}"], norms[f"{s}_L{L}"] = _unit(v).astype(np.float32), float(np.linalg.norm(v))
            cv[s][L] = cv_auroc(M[pos, g], M[neg, g], q[pos], q[neg])
    return dirs, norms, cv


def natural_rows() -> list[dict]:
    rows = []
    for it in NATURAL:
        for lab in ("D", "V"):
            rows.append({"set": "plain", "id": it["id"], "split": it["split"], "label": lab, "user": it["user"],
                         "reply": it[lab]})
        if it["split"] == "test":
            x, b = crossed(it), binding(it)
            rows += [{"set": "crossed", "id": it["id"], "split": "test", "label": lab, "user": it["user"],
                      "reply": x[lab]} for lab in ("D", "V")]
            rows += [{"set": "binding", "id": it["id"], "split": "test", "label": lab, "user": b[u], "reply": b["reply"]}
                     for lab, u in (("D", "user_d"), ("V", "user_v"))]
    return rows


def natural_means(hf, grid: list[int]) -> tuple[list[dict], np.ndarray]:
    rows = natural_rows()
    recs = [{"messages": [{"role": "user", "content": r["user"]}, {"role": "assistant", "content": r["reply"]}],
             "turns": [{"content": r["reply"]}]} for r in rows]
    texts, spans, keep, _ = reply_spans(hf, recs)
    if len(keep) != len(rows):
        raise RuntimeError("natural_means: a naturalistic reply has no tokens")
    caps = hf.capture(texts, spans, grid)
    return rows, np.stack([caps[L] for L in grid], axis=1)


def _sel(rows, **kw) -> np.ndarray:
    return np.array([all(r[k] == v for k, v in kw.items()) for r in rows])


def auroc_tables(dirs, meta, M, nat, X, grid) -> dict:
    """Per layer: for def, hum and opt, the AUROC of D against V (dev, test, mention control, target binding) and the
    dev d'; and the deference floors on the test items (random and shuffled)."""
    layers = {}
    pos = np.array([r["trait"] == "deference" and r["pole"] == "pos" for r in meta])
    neg = np.array([r["trait"] == "deference" and r["pole"] == "neg" for r in meta])
    for g, L in enumerate(grid):
        row = {}
        for s in SHORT.values():
            u = dirs[f"{s}_L{L}"]

            def a(**kw):
                return auroc(X[_sel(nat, label="D", **kw), g] @ u, X[_sel(nat, label="V", **kw), g] @ u)
            dv = (X[_sel(nat, set="plain", split="dev", label="D"), g] @ u,
                  X[_sel(nat, set="plain", split="dev", label="V"), g] @ u)
            row[s] = {"dev": a(set="plain", split="dev"), "test": a(set="plain", split="test"),
                      "dprime_dev": dprime(*dv), "mention": a(set="crossed"), "binding": a(set="binding")}
        xd, xv = X[_sel(nat, set="plain", split="test", label="D"), g], X[_sel(nat, set="plain", split="test", label="V"), g]
        row["floors"] = {"rand95": random_floor(xd, xv), "shuf95": shuffle_floor(M[pos, g], M[neg, g], xd, xv)}
        layers[str(L)] = row
    return {"grid": grid, "layers": layers}


def sigma_at(hf, recs: list[dict], u: np.ndarray, L: int) -> float:
    """SD of the per-token projections h_L.u over the reply tokens of the neutral replies (proj_all_layers' streamed
    sums; with no m its RMS is around the replies' own mean)."""
    texts, _, _, starts = reply_spans(hf, recs)
    return float(hf.proj_all_layers(texts, u[None], starts=starts)["rms"][L, 0])


def refusal_step(hf, recs: list[dict], grid: list[int], l_star: int):
    kept = refusal_kept(recs)
    texts, spans, where = [], [], []
    for r in recs:
        t = hf.render(r["messages"][:-1], None)
        n = len(hf.ids(t))
        texts.append(t)
        spans.append((n - 1, n))
        where.append((r["job"]["pair"], r["job"]["pole"]))
    caps = hf.capture(texts, spans, grid)
    idx = {w: i for i, w in enumerate(where)}
    flags = []
    train = [p for p in kept if p < N_REF_TRAIN]
    held = [p for p in kept if p >= N_REF_TRAIN]
    if not train:
        train = kept if kept else sorted({p for p, _ in where})
        flags.append("train pairs 0-23: none kept; built from " + ("all kept pairs" if kept else "all pairs"))
    dirs, held_auc = {}, {}
    for L in grid:
        X = caps[L]
        harm = np.stack([X[idx[(p, "harm")]] for p in train])
        ben = np.stack([X[idx[(p, "benign")]] for p in train])
        u = _unit(harm.mean(0) - ben.mean(0)).astype(np.float32)
        dirs[f"ref_L{L}"] = u
        if held:
            held_auc[L] = auroc([X[idx[(p, "harm")]] @ u for p in held], [X[idx[(p, "benign")]] @ u for p in held])
    if held_auc:
        l_ref = max(held_auc, key=lambda L: (held_auc[L], -L))
    else:
        l_ref = l_star
        flags.append("no held-out kept pairs (24-31): L_ref = L*")
    info = {"kept": kept, "n_kept": len(kept), "unreliable": len(kept) < MIN_REF_PAIRS, "heldout_auc": held_auc,
            "flags": flags}
    return dirs, l_ref, info


def base_lo(hf, recs: list[dict], m: str, fmt: str) -> tuple[dict, list[dict]]:
    """Modal prefixes per ctx, rung and case (sup and peer episodes pooled; SIDE from the legit episodes), and the
    unhooked LO = log P(money tool + opener close) - log P(request_approval ...) of every sample-0 item on them."""
    by = defaultdict(list)
    for r in recs:
        j = r["job"]
        by[(j["ctx"], j["rung"], j["cid"], "legit" if j["msg"] == "legit" else "twin")].append(r)
    prefixes = {ctx: {rung: {} for rung in K.RUNGS} for ctx in J.CTXS}
    side = {rung: {} for rung in K.RUNGS}
    for (ctx, rung, cid, kind), rs in sorted(by.items()):
        if kind == "twin":
            prefixes[ctx][rung][cid] = modal_prefix(rs, cid, side=False)
        elif ctx == "fresh":
            side[rung][cid] = modal_prefix(rs, cid, side=True)
    items = [r["job"] for r in recs if r["job"]["msg"] in K.MSGS and r["job"]["sample"] == 0]
    bases = [hf.render(*H.prefix_messages(j, prefixes[j["ctx"]][j["rung"]][j["cid"]], fmt)) for j in items]
    texts = [b + H.OPENER[fmt] for b in bases]
    lps = hf.score(texts, [H.decision_conts(j["cid"], fmt) for j in items], [Arm("none")] * len(items))
    tops = hf.next_top(bases, 5)
    rows = []
    for j, (a, b), top in zip(items, lps, tops):
        rows.append({"key": f"{J.EXP}|{m}|lo|{j['ctx']}|{j['rung']}|{j['msg']}|{j['cid']}|p{j['para']}",
                     "ctx": j["ctx"], "rung": j["rung"], "msg": j["msg"], "cid": j["cid"], "para": j["para"],
                     "lp_money": a, "lp_comply": b, "lo": None if a is None or b is None else a - b, "top5": top})
    return {"prefixes": prefixes, "side_prefixes": side}, rows


def alpha_pilot(hf, items: list[dict], vecs: dict[str, np.ndarray], sigma: float, l_star: int, path: Path,
                deadline: float) -> bool:
    """The alpha pilot, one arm at a time (none first), appended per arm. False if the deadline cut it short."""
    hf.set_l_add(l_star)
    if path.exists():
        path.unlink()
    arms = list(dict.fromkeys(it["arm"] for it in items))
    last = 0.0
    for arm in arms:
        t0 = now()
        if t0 + 1.2 * last > deadline:
            _log(f"alpha pilot cut before arm {arm}: the deadline is near")
            return False
        block = [it for it in items if it["arm"] == arm]
        eps = [init_episode(it) for it in block]
        texts = [hf.render(e["messages"], e["tools"]) for e in eps]
        arm_objs = [Arm("none") if it["dir"] is None else
                    Arm(arm, add=(it["k"] * sigma * vecs[it["dir"]]).astype(np.float32)) for it in block]
        res = hf.generate(texts, arm_objs, max(it["max_tokens"] for it in block), util.seed(f"{J.EXP}|alpha|{arm}"))
        nll = hf.nll(texts, [c for c, _ in res])
        with open(path, "a", encoding="utf-8") as f:
            for it, (content, fin), v in zip(block, res, nll):
                f.write(json.dumps({"key": it["key"], "arm": arm, "dir": it["dir"], "k": it["k"], "item": it["item"],
                                    "kind": it["kind"], "cid": it["cid"], "fmt": it["fmt"], "content": content,
                                    "finish": fin, "nll": v}) + "\n")
        last = now() - t0
        _log(f"alpha pilot arm {arm}: {len(block)} turns in {last:.0f}s")
    return True


def run_a(args, hf, vllm_out: Path, out: Path, deadline: float) -> None:
    m = args.model
    fmt, grid = J.MODELS[m][1], GRID[m]
    out.mkdir(parents=True, exist_ok=True)
    recs = load_vllm(vllm_out)
    chat = recs.get("chat", [])
    persona = [r for r in chat if r["job"].get("kind") == "persona"]
    neutral = [r for r in chat if r["job"].get("kind") == "neutral"]
    refusal = [r for r in chat if r["job"].get("kind") == "refusal"]
    _log(f"{m}: {len(persona)} persona, {len(neutral)} neutral, {len(refusal)} refusal replies; "
         f"{len(recs.get('agent', []))} agent and {len(recs.get('l16', []))} l16 episodes")

    meta, M = persona_means(hf, persona, grid)                                                       # 1
    _log(f"step 1: persona means {M.shape}")
    dirs, norms, cv = persona_directions(meta, M, grid)                                              # 2
    np.savez(out / "dirs_grid.npz", **dirs)
    _write_json(out / "id_auroc.json", {"cv_auroc": cv, "norms": norms, "n_replies": len(meta),
                                        "n_skipped_empty": len(persona) - len(meta)})
    _log("step 2: dirs_grid.npz, id_auroc.json")
    nat, X = natural_means(hf, grid)                                                                 # 3
    np.savez_compressed(out / "nat.npz", X=X.astype(np.float16), meta=np.array(json.dumps(nat)))
    _log("step 3: nat.npz")
    tables = auroc_tables(dirs, meta, M, nat, X, grid)                                               # 4
    tables["id_cv"] = cv
    _write_json(out / "auroc.json", tables)
    _log("step 4: auroc.json")

    dev = {L: (tables["layers"][str(L)]["def"]["dev"], tables["layers"][str(L)]["def"]["dprime_dev"]) for L in grid}
    l_star = layer_choice(dev)                                                                       # 5
    g_star = grid.index(l_star)
    sigma = sigma_at(hf, neutral, dirs[f"def_L{l_star}"], l_star)
    pos = np.array([r["trait"] == "deference" and r["pole"] == "pos" for r in meta])
    neg = np.array([r["trait"] == "deference" and r["pole"] == "neg" for r in meta])
    dirs["shuf"] = shuffled_units(M[pos, g_star], M[neg, g_star], 1, SEED)[0].astype(np.float32)
    np.savez(out / "dirs_grid.npz", **dirs)
    at = tables["layers"][str(l_star)]
    summary = {"model": m, "grid": grid, "l_star": l_star, "sigma": sigma, "dev": dev,
               "g2": {"test_auc": at["def"]["test"], "rand95": at["floors"]["rand95"],
                      "shuf95": at["floors"]["shuf95"]},
               "mention": {"auroc": at["def"]["mention"],
                           "flag": "reads the act" if at["def"]["mention"] >= MENTION_BAR else "reads words"},
               "binding": {"auroc": at["def"]["binding"], "flag": "also reads the user's deference"
                           if at["def"]["binding"] >= BINDING_BAR else None},
               "alpha_skipped": None}
    _write_json(out / "summary.json", summary)
    _log(f"step 5: L* {l_star}, sigma {sigma:.3f}, test AUROC {at['def']['test']:.3f}")

    ref_dirs, l_ref, ref_info = refusal_step(hf, refusal, grid, l_star)                             # 6
    dirs.update(ref_dirs)
    np.savez(out / "dirs_grid.npz", **dirs)
    summary.update(l_ref=l_ref, refusal=ref_info)
    _write_json(out / "summary.json", summary)
    _log(f"step 6: L_ref {l_ref}, {ref_info['n_kept']} kept pairs")

    prefixes, lo = base_lo(hf, recs.get("agent", []) + recs.get("l16", []), m, fmt)                  # 7
    _write_json(out / "prefixes.json", prefixes)
    with open(out / "lo_base.jsonl", "w", encoding="utf-8") as f:
        for r in lo:
            f.write(json.dumps(r) + "\n")
    _log(f"step 7: prefixes.json, lo_base.jsonl ({len(lo)} items)")

    if deadline - now() < ALPHA_MIN_LEFT:                                                            # 8
        summary.update(alpha_skipped=True, alpha_note="fewer than 10 minutes left before the deadline")
        _log("step 8: alpha pilot skipped (deadline)")
    else:
        items = _jsonl(Path(args.alpha))
        vecs = {s: dirs[f"{s}_L{l_star}"] for s in SHORT.values()}
        whole = alpha_pilot(hf, items, vecs, sigma, l_star, out / "alpha_pilot.jsonl", deadline)
        summary.update(alpha_skipped=not whole, alpha_note=None if whole else "cut by the deadline")
        _log(f"step 8: alpha pilot {'done' if whole else 'cut'}")
    _write_json(out / "summary.json", summary)

    np.savez_compressed(out / "resp_means.npz", means=M[:, g_star].astype(np.float16),                # 9
                        keys=np.array([r["key"] for r in meta]), trait=np.array([r["trait"] for r in meta]),
                        pole=np.array([r["pole"] for r in meta]), framing=np.array([r["framing"] for r in meta]),
                        q=np.array([r["q"] for r in meta]))
    _log("step 9: resp_means.npz; done")


def main(argv: list[str] | None = None) -> None:
    util.install_term_handlers()
    ap = argparse.ArgumentParser(prog="python -m persona_affect.p13_deference.pod_a", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(J.MODELS))
    ap.add_argument("--vllm-out", required=True)
    ap.add_argument("--alpha", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline", type=float, required=True, help="epoch seconds")
    a = ap.parse_args(argv)
    model_id, fmt, _ = J.MODELS[a.model]
    hf = H.HF(model_id, fmt, l_add=None)
    run_a(a, hf, Path(a.vllm_out), Path(a.out), a.deadline)


if __name__ == "__main__":
    main()
