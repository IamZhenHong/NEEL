"""Run F (owner's Tier-1 plan): is the decision decodable at the first thinking token? Qwen3.6-27B, thinking on.

Q11 (neel-12, q11-results) found that the 27B chain of thought narrates rather than carries the record: deleting the
sentences that cite precedent leaves penalising at 100%, and thinking cut before precedent is mentioned still gives
80%. rr1 found that an inserted contrary conclusion flips the action. F asks WHEN the decision becomes linearly
readable along the trace: at the first thinking token (set before reasoning), only late (formed during reasoning), or
in between.

Two stages on one pod:
  gen   (vLLM, /root/vllm)  justice-desk prompts at 27B's q19b base (no line), Q19 counterbalancing, Q19's two-decision
        own log; cells none / own_R / CH_ownP / own_P; 16 tasks x 16 prompts per cell; k sampled thinking traces per
        prompt (op1 / Q11 settings: mini 0's DECODING, thinking cap 3,072 tokens, one nudge when no choose_option).
  acts  (HF transformers, /root/venv)  one forward pass per closed trace over prompt + thinking (batch 1, stopped after
        the last needed layer, so no lm_head); residual stream after layers 16 / 32 / 48 at five positions: the last
        prompt token (the first thinking token's input), 25 / 50 / 75% of the thinking, and the last thinking token.
        Vectors are reduced on the pod to their top 256 principal components (unsupervised, per layer, pooled over
        traces and positions); the full-width first-token vectors are kept once per prompt.
  analyze (local)  grouped (leave-tasks-out) logistic probes; within-cell AUC; oracle; shuffled floor; variance split.

  uv run python -m persona.history.f_probe build
  /root/vllm/bin/python -m persona.history.f_probe gen --out <dir>          (pod)
  /root/venv/bin/python -m persona.history.f_probe acts --out <dir>         (pod)
  uv run python -m persona.history.f_probe analyze --out persona/history/results/f_27b
"""
import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from persona.exemplar import tasks as T
from persona.exemplar.mini0 import CHANNELS, DECODING, parse_option, seed
from persona.history import batch1 as B1
from persona.history import q19 as Q
from persona.history import runner as RN

MODEL = "Qwen/Qwen3.6-27B"
CELLS = {"none": 4, "own_R": 4, "CH_ownP": 4, "own_P": 2}       # traces per prompt; own_P is the record-direction anchor
REC = {"none": None, "own_R": "R", "own_P": "P", "CH_ownP": "P"}
N_PROMPT, MAX_LEN, THINK_MAX, CHUNK = 16, 8192, 3072, 512
LAYERS, PRIMARY_LAYER = (16, 32, 48), 32
POS = ("first", "q25", "q50", "q75", "end")
N_PC = 256
C_PROBE, N_FOLD, N_BOOT, N_PERM = 0.05, 4, 2000, 20
BAND = (0.20, 0.80)


def read(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def items():
    """One item per (cell, task, prompt); k traces each."""
    hs = B1.histories(MODEL)
    its = []
    for tid in sorted(T.TASKS):
        amap = {o: md for o, md, _ in T.options(T.TASKS[tid])}
        for i in range(N_PROMPT):
            earlier, texts = hs[(tid, i)]
            para, rev, flip = Q.cb(tid, i)
            ear = earlier[-2:][::-1] if rev else earlier[-2:]
            for cell, k in CELLS.items():
                tgt = Q.target(tid, CHANNELS if cell == "CH_ownP" else "", flip)
                rec = Q.record("base", REC[cell], ear, texts, para) if REC[cell] else []
                its.append(dict(pkey=f"F|{cell}|{tid}|{i}", cell=cell, task=tid, sample=i, k=k, amap=amap,
                                messages=tgt[:1] + rec + tgt[1:]))
    return its


def jobs_of(its):
    order = {c: n for n, c in enumerate(CELLS)}                   # primary-eligible cells first
    js = [(it, j) for it in its for j in range(it["k"])]
    return sorted(js, key=lambda x: (order[x[0]["cell"]], x[0]["task"], x[0]["sample"], x[1]))


def prompt_hash(its):
    h = hashlib.sha256()
    for it in its:
        h.update((it["pkey"] + json.dumps(it["messages"], sort_keys=True, ensure_ascii=False)).encode())
    return h.hexdigest()[:16]


def build(a):
    its = items()
    js = jobs_of(its)
    print(f"{len(its)} prompts, {len(js)} traces; cells {CELLS}; prompt hash {prompt_hash(its)}")
    for cell in CELLS:
        it = next(x for x in its if x["cell"] == cell and x["sample"] == 3)
        print("=====", cell, "| system:", it["messages"][0]["content"][:110].replace("\n", " / "))
        for m in it["messages"][1:-3]:
            print("  --", m["role"], (m.get("content") or "")[:160].replace("\n", " / "))


# ---------------------------------------------------------------- stage 1: generation (vLLM)
def run_gen(a):
    from vllm import LLM, SamplingParams, TokensPrompt
    fake = os.environ.get("F_FAKE") == "1"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    its = items()
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=MAX_LEN, gpu_memory_utilization=0.88, seed=0,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    log = lambda m: print(m, flush=True)  # noqa: E731
    log(f"[f] loaded {MODEL} in {time.time() - t0:.0f}s; {len(its)} prompts; hash {prompt_hash(its)}")

    def ptxt(msgs):
        return tok.apply_chat_template(msgs, tools=B1.tools(), tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
    if not fake:
        assert ptxt([{"role": "user", "content": "x"}]).endswith("<think>\n"), "template does not open <think>"
    close_id = None if fake else tok.convert_tokens_to_ids("</think>")
    with open(out / "prompts.jsonl", "w") as f:
        for it in its:
            f.write(json.dumps({"pkey": it["pkey"], "cell": it["cell"], "task": it["task"], "sample": it["sample"],
                                "prompt": ptxt(it["messages"])}, ensure_ascii=False) + "\n")
    done = {r["key"] for r in read(out / "traces.jsonl")}
    js = [(it, j) for it, j in jobs_of(its) if f"{it['pkey']}|{j}" not in done]

    def gen(convs, seeds):
        res = llm.generate([TokensPrompt(prompt_token_ids=tok(ptxt(c), add_special_tokens=False)["input_ids"])
                            for c in convs],
                           [SamplingParams(**DECODING, max_tokens=THINK_MAX, seed=s) for s in seeds], use_tqdm=False)
        return [(r.outputs[0].text, list(getattr(r.outputs[0], "token_ids", None) or [])) for r in res]

    with open(out / "traces.jsonl", "a") as f:
        for c0 in range(0, len(js), CHUNK):
            ch = js[c0:c0 + CHUNK]
            seeds = [seed(f"{it['pkey']}|{j}") for it, j in ch]
            res = gen([it["messages"] for it, _ in ch], seeds)
            rows = []
            for (it, j), (text, ids) in zip(ch, res):
                closed = "</think>" in text
                vis = text.split("</think>")[-1] if closed else ""
                think_ids = ids[:ids.index(close_id)] if (close_id is not None and close_id in ids) else []
                rows.append(dict(key=f"{it['pkey']}|{j}", pkey=it["pkey"], cell=it["cell"], task=it["task"],
                                 sample=it["sample"], j=j, closed=closed, thinking=text.split("</think>")[0] if closed
                                 else None, think_ids=think_ids, visible=vis, option=parse_option(vis)))
            again = [n for n, r in enumerate(rows) if r["option"] is None]
            if again:                       # op1's nudge: the visible reply, then "record your decision", fresh thinking
                r2 = gen([ch[n][0]["messages"] + [{"role": "assistant", "content": rows[n]["visible"]},
                                                  {"role": "user", "content": RN.NUDGE}] for n in again],
                         [seeds[n] + 1 for n in again])
                for n, (t2, _) in zip(again, r2):
                    rows[n]["nudge_text"] = t2
                    rows[n]["option_nudged"] = parse_option(t2.split("</think>")[-1] if "</think>" in t2 else "")
            for (it, _), r in zip(ch, rows):
                opt = r["option"] or r.get("option_nudged")
                r.update(first_try=r["option"] is not None, final_option=opt, mode=it["amap"].get(opt))
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            log(f"[f] gen {min(c0 + CHUNK, len(js))}/{len(js)} ({time.time() - t0:.0f}s)")
    (out / "gen_info.json").write_text(json.dumps({"model": MODEL, "prompts": len(its), "hash": prompt_hash(its),
                                                   "seconds": round(time.time() - t0)}))
    log(f"[f] gen done ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- stage 2: activations (HF)
def _blocks(model):
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise AttributeError("no decoder layer list found")


def positions(P, n):
    """Token indices in prompt + thinking: P prompt tokens, n thinking tokens."""
    return {"first": P - 1, "q25": P + int(0.25 * n), "q50": P + int(0.50 * n), "q75": P + int(0.75 * n),
            "end": P + n - 1}


def run_acts(a):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(a.out)
    t0 = time.time()
    log = lambda m: print(m, flush=True)  # noqa: E731
    mp = os.environ.get("F_MODEL_PATH", MODEL)
    layers = tuple(int(x) for x in os.environ.get("F_LAYERS", ",".join(map(str, LAYERS))).split(","))
    n_pc = int(os.environ.get("F_NPC", N_PC))
    deadline = float(os.environ.get("F_DEADLINE", "inf"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mp)
    model = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
                                                 device_map=dev)
    model.eval()
    blocks = _blocks(model)
    log(f"[f] acts: loaded {mp} on {dev} in {time.time() - t0:.0f}s; {len(blocks)} layers; reading {layers}")
    grab = {}

    class Stop(Exception):
        pass

    def mk(L, last):
        def h(mod, inp, outp):
            grab[L] = (outp[0] if isinstance(outp, tuple) else outp).detach()
            if last:
                raise Stop
        return h
    for L in layers:
        blocks[L - 1].register_forward_hook(mk(L, L == max(layers)))
    prompts = {r["pkey"]: r["prompt"] for r in read(out / "prompts.jsonl")}
    order = {c: n for n, c in enumerate(CELLS)}
    tr = sorted([r for r in read(out / "traces.jsonl") if r["closed"]],
                key=lambda r: (order[r["cell"]], r["task"], r["sample"], r["j"]))
    keys, vecs, first_raw, first_keys, skipped, lens = [], [], {}, [], defaultdict(int), []
    pid_cache = {}
    with torch.no_grad():
        for n, r in enumerate(tr):
            if time.time() > deadline:
                log(f"[f] acts: deadline reached after {n} of {len(tr)} traces; saving what is done")
                skipped["deadline"] = len(tr) - n
                break
            if r["pkey"] not in pid_cache:
                pid_cache[r["pkey"]] = tok(prompts[r["pkey"]], add_special_tokens=False)["input_ids"]
            pids = pid_cache[r["pkey"]]
            tids = r["think_ids"] or tok(r["thinking"], add_special_tokens=False)["input_ids"]
            if len(tids) < 4:
                skipped["short_thinking"] += 1
                continue
            P, m = len(pids), len(tids)
            pos = positions(P, m)
            grab.clear()
            try:
                model(input_ids=torch.tensor([pids + tids], device=dev), use_cache=False)
            except Stop:
                pass
            v = np.stack([np.stack([grab[L][0, pos[p]].float().cpu().numpy() for p in POS]) for L in layers])
            vecs.append(v.astype(np.float16))                      # (layers, positions, d)
            keys.append(r["key"])
            lens.append((P, m))
            if r["pkey"] not in first_raw:
                first_raw[r["pkey"]] = v[:, 0, :].astype(np.float16)
                first_keys.append(r["pkey"])
            if n % 50 == 49:
                log(f"[f] acts {n + 1}/{len(tr)} ({time.time() - t0:.0f}s; last {P}+{m} tokens)")
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    V = np.stack(vecs)                                             # (n, layers, positions, d)
    save = {"keys": np.array(keys), "layers": np.array(layers), "positions": np.array(POS),
            "lens": np.array(lens), "first_keys": np.array(first_keys)}
    for li, L in enumerate(layers):
        X = torch.tensor(V[:, li].reshape(-1, V.shape[-1]), dtype=torch.float32, device=dev)
        mu = X.mean(0)
        q = min(n_pc, X.shape[0] - 1, X.shape[1])
        _, S, Vt = torch.pca_lowrank(X - mu, q=q, center=False, niter=4)
        Z = ((X - mu) @ Vt).reshape(V.shape[0], len(POS), q)
        tot = float(((X - mu) ** 2).sum())
        save[f"Z_L{L}"] = Z.cpu().numpy().astype(np.float16)
        save[f"comp_L{L}"] = Vt.T.cpu().numpy().astype(np.float16)       # (q, d)
        save[f"mu_L{L}"] = mu.cpu().numpy().astype(np.float32)
        save[f"evr_L{L}"] = (S ** 2 / tot).cpu().numpy()
        save[f"first_raw_L{L}"] = np.stack([first_raw[k][li] for k in first_keys])
    np.savez_compressed(out / "acts.npz", **save)
    (out / "acts_info.json").write_text(json.dumps({"model": mp, "layers": list(layers), "positions": list(POS),
                                                    "n_pc": n_pc, "traces": len(keys), "closed": len(tr),
                                                    "skipped": dict(skipped), "seconds": round(time.time() - t0)}))
    log(f"[f] acts done: {len(keys)} traces, skipped {dict(skipped)} ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- analysis (local)
def analyze(a):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    out = Path(a.out)
    tr = {r["key"]: r for r in read(out / "traces.jsonl")}
    allt = list(tr.values())
    A = np.load(out / "acts.npz")
    keys = list(A["keys"])
    layers = [int(x) for x in A["layers"]]
    rows = [tr[k] for k in keys]
    ok = np.array([r["first_try"] and r["mode"] is not None for r in rows])
    y = np.array([r["mode"] == "unilateral" for r in rows], dtype=int)
    cell = np.array([r["cell"] for r in rows])
    task = np.array([r["task"] for r in rows])
    pkey = np.array([r["pkey"] for r in rows])
    rate = {c: float(np.mean(y[ok & (cell == c)])) if (ok & (cell == c)).any() else float("nan") for c in CELLS}
    elig = [c for c in CELLS if BAND[0] <= rate[c] <= BAND[1]]
    L_ = [f"# F: is the decision decodable at the first thinking token? ({MODEL})", "",
          f"Traces: {len(allt)} generated, {sum(r['closed'] for r in allt)} closed, {len(keys)} with activations, "
          f"{int(ok.sum())} decided on the first try. Penalise % by cell (first-try, with activations): "
          + ", ".join(f"{c} {100 * rate[c]:.1f}" for c in CELLS) + f". Eligible (20-80%): {elig or 'none'}."]
    if not elig:
        (out / "summary.md").write_text("\n".join(L_ + ["", "No cell in the 20-80% band: the primary is not readable."]) + "\n")
        print("\n".join(L_))
        return
    m = ok & np.isin(cell, elig)
    yi, ci, ti, pi = y[m], cell[m], task[m], pkey[m]
    tasks = sorted(set(ti))
    onehot = np.stack([(ci == c).astype(float) for c in elig], 1)
    gkf = GroupKFold(n_splits=min(N_FOLD, len(tasks)))

    def oof(X, yy, C=C_PROBE):
        s = np.zeros(len(yy))
        for trn, tst in gkf.split(X, yy, ti):
            sc = StandardScaler().fit(X[trn])
            clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[trn]), yy[trn])
            s[tst] = clf.decision_function(sc.transform(X[tst]))
        return s

    def wauc(s, yy, idx=None):
        """Within-cell AUC, cells weighted by n; NaN if a cell has one class only in this sample."""
        idx = np.arange(len(yy)) if idx is None else idx
        tot, w = 0.0, 0
        for c in elig:
            j = idx[ci[idx] == c]
            if len(set(yy[j])) < 2:
                continue
            tot += roc_auc_score(yy[j], s[j]) * len(j)
            w += len(j)
        return tot / w if w else float("nan")

    rng = random.Random(0)
    by_task = {t: np.where(ti == t)[0] for t in tasks}
    boots = [np.concatenate([by_task[rng.choice(tasks)] for _ in tasks]) for _ in range(N_BOOT)]

    def ci_(s, yy=None):
        yy = yi if yy is None else yy
        v = sorted(x for x in (wauc(s, yy, b) for b in boots) if x == x)
        return wauc(s, yy), v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]
    # oracle: each prompt's p-hat from its other samples (leave-one-out)
    grp = defaultdict(list)
    for n_, p in enumerate(pi):
        grp[p].append(n_)
    orc = np.full(len(yi), np.nan)
    for p, ix in grp.items():
        for n_ in ix:
            others = [yi[j] for j in ix if j != n_]
            orc[n_] = np.mean(others) if others else np.nan
    keep_o = ~np.isnan(orc)
    oracle = ci_(np.where(keep_o, orc, 0.5))
    # variance split (eligible cells): between-prompt vs within-prompt
    pm = {p: np.mean(yi[ix]) for p, ix in grp.items()}
    within = np.mean([np.var(yi[ix]) * len(ix) / max(1, len(ix) - 1) for ix in grp.values() if len(ix) > 1])
    total = np.var(yi)
    res, per_layer = {}, {}
    for L in layers:
        Z = A[f"Z_L{L}"].astype(np.float32)[m]                   # (n, positions, q)
        per_layer[L] = {}
        for k, p in enumerate(POS):
            s = oof(np.concatenate([Z[:, k], onehot], 1), yi)
            per_layer[L][p] = s
            res[(L, p)] = ci_(s)
    L0 = PRIMARY_LAYER if PRIMARY_LAYER in layers else layers[len(layers) // 2]
    first = res[(L0, "first")]
    end = res[(L0, "end")]
    # shuffled floor (labels permuted within cell)
    Zf = np.concatenate([A[f"Z_L{L0}"].astype(np.float32)[m][:, 0], onehot], 1)
    floor = []
    for r_ in range(N_PERM):
        yp = yi.copy()
        rr = np.random.default_rng(r_)
        for c in elig:
            j = np.where(ci == c)[0]
            yp[j] = rr.permutation(yp[j])
        floor.append(wauc(oof(Zf, yp), yp))
    floor95 = float(np.percentile(floor, 95))

    def within_prompt_auc(s):
        num = den = 0.0
        for ix in grp.values():
            pos_ = [j for j in ix if yi[j] == 1]
            neg_ = [j for j in ix if yi[j] == 0]
            for a_ in pos_:
                for b_ in neg_:
                    num += 1.0 if s[a_] > s[b_] else 0.5 if s[a_] == s[b_] else 0.0
                    den += 1
        return num / den if den else float("nan"), int(den)
    # first-token probe vs prompt means
    sp = per_layer[L0]["first"]
    pr = sorted(grp)
    r_first = float(np.corrcoef([np.mean(sp[grp[p]]) for p in pr], [pm[p] for p in pr])[0, 1])
    # cosine of the first-token probe with the record direction (own_P - own_R mean, PC space, layer L0)
    Zall = A[f"Z_L{L0}"].astype(np.float32)[:, 0]
    okP, okR = (cell == "own_P"), (cell == "own_R")
    cos = float("nan")
    if okP.any() and okR.any():
        rdir = Zall[okP].mean(0) - Zall[okR].mean(0)
        sc = StandardScaler().fit(Zall[m])
        clf = LogisticRegression(C=C_PROBE, max_iter=3000).fit(sc.transform(Zall[m]), yi)
        w = clf.coef_[0] / sc.scale_
        cos = float(w @ rdir / (np.linalg.norm(w) * np.linalg.norm(rdir) + 1e-12))
    ratio = (first[0] - 0.5) / (oracle[0] - 0.5) if oracle[0] > 0.5 else float("nan")
    verdict = ("set before reasoning" if first[0] >= 0.70 and first[1] > 0.60 and ratio >= 0.80 else
               "formed during reasoning" if first[2] < 0.60 and end[0] >= 0.85 else "intermediate")
    f3 = lambda e: f"{e[0]:.3f} [{e[1]:.3f}, {e[2]:.3f}]"  # noqa: E731
    L_ += ["", f"Eligible traces {len(yi)} in {len(grp)} prompts; decision variance {total:.3f}, within-prompt (sampling) "
               f"{within:.3f} ({100 * within / total:.0f}%), between-prompt {total - within:.3f} "
               f"({100 * (total - within) / total:.0f}%).", "",
           f"**Primary (layer {L0}): first-token within-cell AUC {f3(first)}; oracle (per-prompt p-hat, leave-one-out) "
           f"{f3(oracle)}; above-chance ratio {ratio:.2f}; shuffled floor (95th pct of {N_PERM}) {floor95:.3f}; "
           f"end-of-thinking {f3(end)} -> {verdict.upper()}**", "",
           "| layer | " + " | ".join(POS) + " |", "|---|" + "---|" * len(POS)]
    for L in layers:
        L_.append(f"| {L} | " + " | ".join(f3(res[(L, p)]) for p in POS) + " |")
    L_ += ["", "Within-prompt AUC (pairs of samples of the same prompt with different decisions; 0.5 at the first token "
               "by construction):", "", "| layer | " + " | ".join(POS) + " |", "|---|" + "---|" * len(POS)]
    for L in layers:
        L_.append(f"| {L} | " + " | ".join("{:.3f} (n={})".format(*within_prompt_auc(per_layer[L][p])) for p in POS) + " |")
    L_ += ["", f"First-token probe score vs prompt mean decision, across {len(pr)} prompts: r = {r_first:.2f}.",
           f"Cosine of the first-token probe (layer {L0}, PC space) with the record direction (own_P - own_R mean): "
           f"{cos:+.2f} (descriptive).",
           f"Settings: logistic C = {C_PROBE} on standardised top-{A[f'Z_L{L0}'].shape[-1]} PCs + cell indicators; "
           f"{gkf.get_n_splits()}-fold leave-tasks-out; task-cluster bootstrap {N_BOOT}."]
    (out / "summary.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "gen", "acts", "analyze"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    {"build": build, "gen": run_gen, "acts": run_acts, "analyze": analyze}[a.cmd](a)
