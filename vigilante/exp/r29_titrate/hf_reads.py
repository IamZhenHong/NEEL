"""R29 on the pod, after the vLLM titration (rules in README.md, prereg vig-r29-prereg):
(1) rates per level; pick the dev level (phones, closest to 50% acted) and the test level (bags, closest to 50%);
(2) teacher-forced layer-L residuals averaged over the thought of each dev episode's read turn (the first turn that
    calls a trust or report tool), v_tit = mean(acted) - mean(not acted); floors (5-fold CV AUROC vs shuffled labels,
    random directions, thought length, read-turn index);
(3) the same-norm placebo (label-shuffled, label-balanced diff of means, orthogonal to v_tit);
(4) dirs.pt + steering jobs for src/steer_run.py (test level and B1@closed under P4m, -2 x v_tit vs -2 x placebo).

  python -m vigilante.exp.r29_titrate.hf_reads --titr <episodes_agent.jsonl> --out <dir> [--layer 30]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from vigilante.src.score import score_episode
from vigilante.src.steer_run import Steer, find_layers, load
from vigilante.src.templates import to_template

CH_OPEN = "<|channel>"
TRUST = {"get_trust_profile", "update_trust_score", "set_account_limits", "report_to_trust_safety"}
V_VIG_NORM = 19.8                       # R20's |v_vig| at layer 30; a steering vector is capped at 2x this norm
N_TEST, N_TRANSFER = 20, 10


def auroc(scores, y):
    s, y = np.asarray(scores, float), np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def cv_auroc(X, y, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    scores = np.zeros(len(y))
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        v = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        scores[te] = X[te] @ v
    return auroc(scores, y)


def pick(rates, family):
    """Level closest to 50% acted; ties go to the lower level."""
    lv = sorted((abs(r["acted"] / r["n"] - 0.5), r["level"]) for k, r in rates.items() if r["family"] == family)
    return lv[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--titr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--layer", type=int, default=30)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(a.titr) if l.strip()]
    rows = [score_episode(r) for r in recs]
    rates = {}
    for r, s in zip(recs, rows):
        c = r["job"]["case"]
        d = rates.setdefault(c, {"family": "phones" if "phones" in c else "bags", "level": int(c.split("@")[0][-1]),
                                 "n": 0, "acted": 0})
        d["n"] += 1
        d["acted"] += bool(s["unauth"])
    dev = f"E1t:phones{pick(rates, 'phones')}@closed"
    test = f"E1t:bags{pick(rates, 'bags')}@closed"
    sel = {"rates": rates, "dev": dev, "test": test}
    (out / "selection.json").write_text(json.dumps(sel, indent=1))
    print(f"[r29] rates {json.dumps(rates)}\n[r29] dev {dev}, test {test}", flush=True)

    # (2) reads on the dev episodes
    tok, model = load(a.model)
    n_layers = model.config.get_text_config().num_hidden_layers
    _, layers = find_layers(model, n_layers)
    steer = Steer()
    layers[a.layer].register_forward_hook(steer.hook)
    X, y, meta = [], [], []
    for r, s in zip(recs, rows):
        if r["job"]["case"] != dev:
            continue
        t_read = next((t for t in r["turns"] if t["phase"] == "task" and any(c["name"] in TRUST for c in t["calls"])), None)
        if t_read is None or not t_read["reasoning"]:
            meta.append({"key": r["key"], "acted": bool(s["unauth"]), "skipped": "no trust/report call or no thought"})
            continue
        k = sum(1 for t in r["turns"] if t["step"] < t_read["step"] and t["phase"] == "task")  # earlier assistant turns
        msgs, seen = [], 0
        for m in r["messages"]:
            if m["role"] == "assistant":
                if seen == k:
                    break
                seen += 1
            msgs.append(m)
        prefix = tok.apply_chat_template(to_template(msgs, "gemma4"), tools=r["job"].get("tools") or _tools(r), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        if not prefix.endswith(CH_OPEN + "thought\n"):
            prefix += CH_OPEN + "thought\n"
        p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        full = tok(prefix + t_read["reasoning"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        steer.capture = []
        with torch.no_grad():
            model(input_ids=full)
        h = steer.capture[0][0].float()
        steer.capture = None
        X.append(h[len(p_ids):].mean(0).cpu().numpy())
        y.append(int(bool(s["unauth"])))
        meta.append({"key": r["key"], "acted": bool(s["unauth"]), "read_step": t_read["step"],
                     "thought_tokens": full.shape[1] - len(p_ids), "prompt_tokens": len(p_ids)})
    X, y = np.stack(X), np.array(y)
    v = X[y == 1].mean(0) - X[y == 0].mean(0)
    rng = np.random.default_rng(20261013)
    shuf = [cv_auroc(X, rng.permutation(y), seed=i) for i in range(50)]
    rand = []
    for i in range(200):
        u = rng.standard_normal(X.shape[1])
        sc = auroc(X @ u, y)
        rand.append(max(sc, 1 - sc))
    used = [m for m in meta if "skipped" not in m]
    rd = {"n": int(len(y)), "acted": int(y.sum()), "skipped": len(meta) - len(used),
          "cv_auroc": cv_auroc(X, y), "shuffled_p95": float(np.percentile(shuf, 95)),
          "random_p95": float(np.percentile(rand, 95)),
          "thought_len_auroc": auroc([m["thought_tokens"] for m in used], y),
          "prompt_len_auroc": auroc([m["prompt_tokens"] for m in used], y),
          "read_step_auroc": auroc([m["read_step"] for m in used], y),
          "read_step_by_outcome": {o: sorted({m["read_step"] for m in used if m["acted"] == o}) for o in (True, False)},
          "v_norm": float(np.linalg.norm(v))}
    print(f"[r29] reads {json.dumps(rd)}", flush=True)

    # (3) placebo: label-shuffled, label-balanced diff of means, orthogonal to v, same norm
    pseudo = np.zeros(len(y), int)
    for lab in (0, 1):                  # each pseudo-group gets half of the acted and half of the not-acted reads
        idx = rng.permutation(np.where(y == lab)[0])
        pseudo[idx[: len(idx) // 2]] = 1
    p = X[pseudo == 1].mean(0) - X[pseudo == 0].mean(0)
    p = p - (p @ v) / (v @ v) * v
    p = p / np.linalg.norm(p) * np.linalg.norm(v)
    scale = min(1.0, 2 * V_VIG_NORM / (2 * np.linalg.norm(v)))      # cap |2 v| at 2 |v_vig|
    rd["alpha"] = -2.0 * scale
    rd["placebo_cos_v"] = float(p @ v / (np.linalg.norm(p) * np.linalg.norm(v)))
    (out / "reads.json").write_text(json.dumps({"summary": rd, "items": meta}, indent=1))
    vt, pt = torch.tensor(v, dtype=torch.float32), torch.tensor(p, dtype=torch.float32)
    torch.save({"vtit": vt, "placebo": pt, "batman": vt, "layer": a.layer, "random_unit": vt / vt.norm()}, out / "dirs.pt")

    # (4) steering jobs: thought-gated in every turn (steer_run)
    js = []
    for case, n, cond in ((test, N_TEST, "P4m"), ("B1@closed", N_TRANSFER, "P4m")):
        for arm in ("vtit", "placebo"):
            for i in range(n):
                key = f"R29s|{arm}|{case}|{i}"
                js.append({"key": key, "exp": "R29s", "group": "agent", "cond": cond, "case": case, "fmt": "gemma4",
                           "thinking": True, "max_tokens": 1500, "handover": False, "sample": i, "arm": arm,
                           "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16),
                           "steer": {"dir": arm, "alpha": rd["alpha"]}})
    with open(out / "steer_jobs.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"[r29] {len(js)} steering jobs; alpha {rd['alpha']:.3f}", flush=True)


def _tools(rec):
    from vigilante.cases import ALL
    return ALL[rec["job"]["case"]].tools


if __name__ == "__main__":
    main()
