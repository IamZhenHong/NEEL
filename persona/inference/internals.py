"""Internals 13c/d + 19b (persona/inference/INTERNALS.md): the behavioural A1 result says a BIOGRAPHY drives action
through a *removable identity* (role-play) while a RECORD drives it through *precedent* (a norm). That predicts two
DISTINCT internal directions. Two experiments test it on Qwen3.6-27B, reusing persona.reasoning.probes (Model, per-layer
steer+capture hooks) read-only and items from persona.inference.a1 cells:

  Exp 1 (13c/d) — is the record-induced persona the SAME direction as a PROMPTED persona?
    v_persona = mean(label) - mean(none) over an identical held-out target-case span (the explicit "You are Dorn Vale,
    a ruthless fixer..." system line vs nothing). Measure how much the biography cell (bio / facts-dose f2,f4,f8) and
    the record cell (rec / decisions-dose d1,d2,d4,d6) PROJECT onto v_persona, and a cross-validated logistic probe
    trained on label-vs-none scored on each cell, with a shuffled-label null. cos(v_bio, v_persona) vs cos(v_rec,
    v_persona) is the headline. Prediction: biography loads on v_persona (facts-dose climbs it); the record loads far
    less (decisions-dose stays flat) even though both drive the action.

  Exp 2 (19b) — does the record drive action THROUGH a steerable direction (causal mediation)?
    v_rec = mean(rec) - mean(none) over the target span; steer +v_rec into the NONE context and read P(harsh) from the
    choose_option logits (does the record-typical action appear with no record present?); ablate -v_rec from the REC
    context (does it stop?). Controls: v_bio (biography's direction) and v_rand (matched-norm random). Honest prior:
    seed 4 found the ownership direction consistent but only a WEAK late lever, so this may come back weak — registered
    as a real outcome, not spun.

  uv run python -m persona.inference.internals build                 # local: cells, spans, distinct option ids
  uv run python -m persona.inference.internals fake --out results_internals_dry   # dry run: plant + recover
  .venv/bin/python -m persona.inference.internals run --out persona/inference/results_internals   # POD
  uv run python -m persona.inference.internals analyze --out persona/inference/results_internals

POD NOTE (per neel-12): Qwen3.6-27B loads via HF AutoModelForCausalLM (bf16, device_map cuda) in the PROJECT venv —
`python -m`, NOT the vllm python. The Model class below is VENDORED from persona.reasoning.probes@seeds-results
(copied read-only) so this harness does not depend on neel-12's in-flight edits to that file (they are adding seeds
5/8b). The forward is single-sequence / same-length-rows only.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from persona.inference.a1 import (DECIS, DEPLOY, IDS, SYS_EXTRA, TGT_CASES, TOOL, decision_msgs, inducer, options)

H = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3.6-27B"
OUT = H / "results_internals"
L = (24, 32, 40)                 # decision-point layers (probes.py L4)
ALPHAS = (1.0, 2.0)
N = 24                           # samples per cell (even -> directions, odd -> steering/held-out projections)
# cells: prompted persona (label/licence) | biography (bio + facts dose) | record (rec + decisions dose) | both | none
CELLS = ("none", "label", "licence", "bio", "f2", "f4", "f8", "rec", "d1", "d2", "d4", "d6", "both")
READOUT = TGT_CASES              # ("d_tip", "d_junior") — held-out targets; identical text across cells = the span
PREFIX = ("<tool_call>\n<function=choose_option>\n<parameter=case_id>\n{case}\n</parameter>\n"
          "<parameter=option_id>\n")


# ---------------------------------------------------------------------------------------------------- model (pod)
class Model:
    """Vendored read-only from persona.reasoning.probes (neel-12). Per-layer forward hooks that both STEER (add a
    vector to the residual from a start index) and CAPTURE (mean residual over a slice). Single-sequence forward."""

    def __init__(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        cfg = AutoConfig.from_pretrained(MODEL)
        n = getattr(cfg, "text_config", cfg).num_hidden_layers
        try:
            self.m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
        except Exception as e:  # noqa: BLE001
            print(f"[internals] AutoModelForCausalLM failed ({type(e).__name__}); ImageTextToText", flush=True)
            from transformers import AutoModelForImageTextToText
            self.m = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
        self.m.eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.layers = next(mod for name, mod in self.m.named_modules()
                           if isinstance(mod, torch.nn.ModuleList) and len(mod) == n and name.endswith("layers"))
        self.steer = None          # (layer, vecs [B, D], start index)
        self.cap = {}
        self.capture_layers = ()
        self.capture_slice = None
        for i, lay in enumerate(self.layers):
            lay.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def h(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if self.steer is not None and self.steer[0] == i:
                _, vecs, start = self.steer
                add = self.torch.zeros_like(t)
                add[:, start:, :] = vecs[:, None, :].to(t.dtype)
                t = t + add
                out = (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
            if i in self.capture_layers:
                sl = self.capture_slice
                self.cap[i] = (t[:, sl[0]:sl[1], :].float().mean(1) if sl[1] - sl[0] > 1 else t[:, sl[0], :].float()).cpu()
            return out
        return h

    def render(self, msgs, tools, think=False):
        return self.tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=think)

    def ids(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def forward(self, batch_ids):
        """Same-length rows only (callers batch identical sequences)."""
        x = self.torch.tensor(batch_ids, device="cuda")
        with self.torch.no_grad():
            return self.m(input_ids=x, use_cache=False).logits[:, -1, :].float()


# ---------------------------------------------------------------------------------------------------- items (CPU)
def build_items():
    items = []
    for cell in CELLS:
        for s in range(N):
            ind = inducer(cell if cell not in SYS_EXTRA else "none", s)
            sysc = (SYS_EXTRA[cell] + "\n\n" + DEPLOY) if cell in SYS_EXTRA else DEPLOY
            sysm = [{"role": "system", "content": sysc}]
            for tid in READOUT:
                msgs = sysm + ind + decision_msgs(tid)
                items.append(dict(cell=cell, sample=s, task=tid, messages=msgs,
                                  amap={o: o for o, _ in options(tid)}))
    return items


def tgt_span(M, msgs, tools):
    """Token index where the held-out target case (the final user turn) starts; its text is identical across cells."""
    full = M.render(msgs, tools)
    req = msgs[-1]
    assert req["role"] == "user", req["role"]
    pos = full.rfind("<|im_start|>user\n" + req["content"][:60])
    assert pos > 0, "target turn not found"
    st = len(M.ids(full[:pos]))
    assert M.ids(full)[:st] == M.ids(full[:pos]), "prefix tokenisation not stable"
    return st, full


# ---------------------------------------------------------------------------------------------------- run (pod)
def _dirs(acts, layer, gen):
    import torch
    mean = lambda c: torch.stack(acts[(c, layer)]).mean(0)   # noqa: E731
    d = {"persona": mean("label") - mean("none"), "licence": mean("licence") - mean("none"),
         "bio": mean("bio") - mean("none"), "rec": mean("rec") - mean("none")}
    r = torch.randn(d["rec"].shape, generator=gen)
    d["rand"] = r / r.norm() * d["rec"].norm()
    return d


def run(a):
    import numpy as np
    import torch
    t0 = time.time()
    log = lambda m: print(m, flush=True)          # noqa: E731
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    M = Model()
    log(f"[internals] model loaded ({time.time() - t0:.0f}s)")
    items = build_items()
    opt_ids = {o: M.ids(o)[0] for o in IDS}
    assert len(set(opt_ids.values())) == len(opt_ids), f"option first-tokens collide: {opt_ids}"
    prep = []
    for it in items:
        st, full = tgt_span(M, it["messages"], [TOOL])
        ids = M.ids(full + PREFIX.format(case=it["task"].upper()))
        prep.append(dict(it, start=st, n_prompt=len(M.ids(full)), ids=ids))

    # ---- capture: mean residual over the (identical) target-case span, every item, every layer ----
    acts = defaultdict(list)                       # (cell, layer) -> [vec]  (ALL samples; even build dirs)
    capX = {ly: [] for ly in L}
    capmeta = []
    M.capture_layers = L
    for n, p in enumerate(prep):
        M.capture_slice = (p["start"], p["n_prompt"])
        M.forward([p["ids"]])
        for ly in L:
            v = M.cap[ly][0]
            capX[ly].append(v.numpy().astype(np.float16))
            if int(p["sample"]) % 2 == 0:
                acts[(p["cell"], ly)].append(v)
        capmeta.append({k: p[k] for k in ("cell", "sample", "task")})
        if n % 100 == 0:
            log(f"[internals] capture {n + 1}/{len(prep)} ({time.time() - t0:.0f}s)")
    M.capture_layers = ()
    np.savez_compressed(out / "caps.npz", **{f"L{ly}": np.stack(v) for ly, v in capX.items()})
    (out / "caps_meta.jsonl").write_text("\n".join(json.dumps(m) for m in capmeta) + "\n")

    # ---- directions + cosines ----
    gen = torch.Generator().manual_seed(0)
    dirs = {ly: _dirs(acts, ly, gen) for ly in L}
    torch.save({f"{k}|{ly}": v for ly in L for k, v in dirs[ly].items()}, out / "dirs.pt")
    cos = lambda ly, x, y: float(torch.nn.functional.cosine_similarity(dirs[ly][x], dirs[ly][y], dim=0))  # noqa: E731
    cosm = {f"L{ly}": {p: round(cos(ly, *p.split("-")), 3)
                       for p in ("bio-persona", "rec-persona", "bio-rec", "licence-persona", "rec-rand")} for ly in L}
    log(f"[internals] cosines {json.dumps(cosm)}")
    (out / "cosines.json").write_text(json.dumps(cosm, indent=1))

    # ---- Exp 2: steer directions into the NONE context (and ablate from REC), odd samples ----
    BATCH = 8
    conds = []                                     # (prep_item, cell_role, kind, layer, alpha, sign)
    for p in prep:
        if int(p["sample"]) % 2 == 0:
            continue
        if p["cell"] == "none":
            conds.append((p, "none", "base", None, 0.0, +1))
            conds += [(p, "none", k, ly, al, +1) for k in ("rec", "bio", "rand") for ly in L for al in ALPHAS]
        elif p["cell"] == "rec":
            conds.append((p, "rec", "base", None, 0.0, +1))
            conds += [(p, "rec", "rec", ly, al, -1) for ly in L for al in ALPHAS]   # ablate
    log(f"[internals] Exp 2: {len(conds)} steered readouts")
    by_item = defaultdict(list)
    for c in conds:
        by_item[(c[0]["cell"], c[0]["sample"], c[0]["task"])].append(c)
    with open(out / "steer.jsonl", "w") as f:
        for ndone, rows in enumerate(by_item.values()):
            p = rows[0][0]
            for b0 in range(0, len(rows), BATCH):
                ch = rows[b0:b0 + BATCH]
                groups = defaultdict(list)                        # one steering layer per forward
                for row in ch:
                    groups[row[3] or 32].append(row)
                for ly, grp in groups.items():
                    vecs = torch.stack([(sgn * al * dirs[ly][kind]) if kind != "base"
                                        else torch.zeros_like(dirs[ly]["rec"])
                                        for _, _, kind, _, al, sgn in grp]).cuda()
                    M.steer = (ly, vecs, p["start"])
                    logits = M.forward([p["ids"]] * len(grp))
                    M.steer = None
                    for (pp, role, kind, LL, al, sgn), lg in zip(grp, logits):
                        sel = torch.tensor([opt_ids[o] for o in IDS]).cuda()
                        pr = torch.softmax(lg[sel], 0).cpu().tolist()
                        f.write(json.dumps({"cell": pp["cell"], "role": role, "task": pp["task"],
                                            "sample": pp["sample"], "dir": kind, "layer": LL,
                                            "alpha": al * sgn, "p": dict(zip(IDS, pr))}) + "\n")
            if ndone % 20 == 0:
                f.flush()
                log(f"[internals] steer item {ndone + 1}/{len(by_item)} ({time.time() - t0:.0f}s)")
    (out / "run_info.json").write_text(json.dumps({"model": MODEL, "n_items": len(items),
                                                    "seconds": round(time.time() - t0)}))
    log(f"[internals] done ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------------------------------- dry run
def fake(a):
    """Plant: persona-loading climbs with FACTS (bio/f*), near-zero for the RECORD (rec/d*); action-loading climbs
    with the RECORD. Write caps.npz + steer.jsonl in the run's format; analyze must recover cos(bio,persona)>>cos(rec,
    persona), a facts-dose projection that climbs vs a flat decisions-dose, and v_rec steering >> v_rand."""
    import numpy as np
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = 64
    rng = np.random.default_rng(0)
    e_per = np.zeros(D); e_per[0] = 1.0            # persona (identity) axis
    e_act = np.zeros(D); e_act[1] = 1.0            # action (precedent) axis
    per = {"none": 0, "label": 1.0, "licence": 0.9, "bio": 0.8, "f2": 0.3, "f4": 0.55, "f8": 0.78,
           "rec": 0.12, "d1": 0.04, "d2": 0.08, "d4": 0.11, "d6": 0.13, "both": 0.85}
    act = {"none": 0, "label": 0.3, "licence": 0.5, "bio": 0.35, "f2": 0.1, "f4": 0.15, "f8": 0.2,
           "rec": 0.9, "d1": 0.4, "d2": 0.7, "d4": 0.85, "d6": 0.9, "both": 0.9}
    capX = {ly: [] for ly in L}
    meta = []
    for cell in CELLS:
        for s in range(N):
            for tid in READOUT:
                base = rng.normal(0, 0.3, D) + 5.0
                v = base + per[cell] * 3 * e_per + act[cell] * 3 * e_act
                for ly in L:
                    capX[ly].append((v + rng.normal(0, 0.2, D)).astype(np.float16))
                meta.append({"cell": cell, "sample": s, "task": tid})
    np.savez_compressed(out / "caps.npz", **{f"L{ly}": np.stack(v) for ly, v in capX.items()})
    (out / "caps_meta.jsonl").write_text("\n".join(json.dumps(m) for m in meta) + "\n")
    # steering: base none low P(harsh); +v_rec high; +v_bio mid; +v_rand ~base; ablate rec -> drop
    def ph(role, kind, al):
        if role == "none":
            x = 0.2 + {"base": 0, "rec": 0.5, "bio": 0.25, "rand": 0.02}[kind] * abs(al) / 2
        else:                                      # rec context: base high, ablate drops
            x = 0.85 - {"base": 0, "rec": 0.4}[kind] * abs(al) / 2
        return max(0.02, min(0.96, x))
    with open(out / "steer.jsonl", "w") as f:
        for s in range(1, N, 2):
            for tid in READOUT:
                for role, kinds in (("none", [("base", None, 0.0)] + [(k, ly, al) for k in ("rec", "bio", "rand")
                                                                       for ly in L for al in ALPHAS]),
                                    ("rec", [("base", None, 0.0)] + [("rec", ly, -al) for ly in L for al in ALPHAS])):
                    for kind, ly, al in kinds:
                        p = ph(role, kind, al)
                        f.write(json.dumps({"cell": role, "role": role, "task": tid, "sample": s, "dir": kind,
                                            "layer": ly, "alpha": al, "p": {"harsh": p, "measured": (1 - p) * 0.6,
                                                                            "refer": (1 - p) * 0.4}}) + "\n")
    (out / "run_info.json").write_text(json.dumps({"model": "FAKE", "n_items": len(CELLS) * N * len(READOUT)}))
    print(f"[internals] dry-run files written to {out}; now: analyze --out {out}")


# ---------------------------------------------------------------------------------------------------- analyze
def analyze(a):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    out = Path(a.out)
    d = np.load(out / "caps.npz")
    meta = [json.loads(l) for l in open(out / "caps_meta.jsonl")]
    cell = np.array([m["cell"] for m in meta])
    samp = np.array([m["sample"] for m in meta])
    model = json.loads((out / "run_info.json").read_text())["model"]
    rng = np.random.default_rng(0)
    res = {"model": model, "by_layer": {}}
    for ly in L:
        X = d[f"L{ly}"].astype(np.float32)
        mean = lambda c: X[cell == c].mean(0)      # noqa: E731
        vp, vb, vr = mean("label") - mean("none"), mean("bio") - mean("none"), mean("rec") - mean("none")
        unit = lambda v: v / (np.linalg.norm(v) + 1e-8)  # noqa: E731
        cosd = lambda x, y: float(unit(x) @ unit(y))     # noqa: E731
        up = unit(vp)
        proj = {c: round(float(((X[cell == c] - mean("none")) @ up).mean()), 3) for c in CELLS}
        # logistic probe: label(+licence) vs none, even samples; score each cell on odd samples; shuffled null
        tr = np.isin(cell, ["label", "licence", "none"]) & (samp % 2 == 0)
        y = np.isin(cell[tr], ["label", "licence"]).astype(int)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xs[tr], y)
        te = (samp % 2 == 1)
        score = {c: round(float(clf.decision_function(Xs[(cell == c) & te]).mean()), 3) for c in CELLS}
        # held-out label-vs-none AUC (odd samples) + shuffled-label null p95
        teln = np.isin(cell, ["label", "none"]) & te
        yln = np.isin(cell[teln], ["label"]).astype(int)
        auc = float(roc_auc_score(yln, clf.decision_function(Xs[teln]))) if len(set(yln)) == 2 else None
        null = sorted(float(roc_auc_score(rng.permutation(yln), clf.decision_function(Xs[teln])))
                      for _ in range(200))
        res["by_layer"][f"L{ly}"] = {
            "cos": {"bio-persona": round(cosd(vb, vp), 3), "rec-persona": round(cosd(vr, vp), 3),
                    "bio-rec": round(cosd(vb, vr), 3)},
            "proj_on_persona": proj, "probe_score": score,
            "label_vs_none_auc": round(auc, 3) if auc else None, "shuffled_auc_p95": round(null[189], 3)}
    # Exp 2 steering: base-subtracted P(harsh), bootstrap over (task, sample)
    S = [json.loads(l) for l in open(out / "steer.jsonl")]
    base = {}
    for r in S:
        if r["dir"] == "base":
            base[(r["role"], r["task"], r["sample"])] = r["p"]["harsh"]
    by = defaultdict(list)
    for r in S:
        if r["dir"] == "base":
            continue
        b = base.get((r["role"], r["task"], r["sample"]))
        if b is not None:
            by[(r["role"], r["dir"], r["layer"], r["alpha"])].append(r["p"]["harsh"] - b)
    steer = {}
    base_rate = {role: round(100 * np.mean([v for (ro, t, s), v in base.items() if ro == role]), 1)
                 for role in ("none", "rec")}
    for (role, kind, ly, al), v in sorted(by.items(), key=str):
        arr = np.array(v)                          # one delta per (task, sample) unit in this condition
        pt = 100 * arr.mean()
        bs = sorted(100 * arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(5000)) if len(arr) else [0, 0]
        steer[f"{role}|{kind}|L{ly}|a{al:+.1f}"] = [round(pt, 1), round(bs[125], 1), round(bs[4874], 1)]
    res["exp2_steering"] = {"base_P_harsh": base_rate, "delta_vs_base": steer}
    (out / "analysis.json").write_text(json.dumps(res, indent=1))
    # readable summary
    ln = [f"# Internals 13c/d + 19b — {model}", "",
          "## Exp 1: is the record-induced persona the prompted-persona direction?"]
    for ly in L:
        r = res["by_layer"][f"L{ly}"]
        ln += [f"### L{ly}  (label-vs-none AUC {r['label_vs_none_auc']}, shuffled p95 {r['shuffled_auc_p95']})",
               f"cos  bio–persona {r['cos']['bio-persona']} | rec–persona {r['cos']['rec-persona']} | "
               f"bio–rec {r['cos']['bio-rec']}",
               "proj on persona axis: " + "  ".join(f"{c} {r['proj_on_persona'][c]:+}" for c in CELLS)]
    ln += ["", "Read: facts-dose (f2→f8) should climb the persona axis while decisions-dose (d1→d6) stays flat;",
           "cos(bio,persona) >> cos(rec,persona) is the mask, measured.", "",
           "## Exp 2: does the record drive action through a steerable direction?",
           f"base P(harsh): none {res['exp2_steering']['base_P_harsh']['none']}% | "
           f"rec {res['exp2_steering']['base_P_harsh']['rec']}%", "", "Δ P(harsh) vs base [95% CI]:"]
    for k, v in res["exp2_steering"]["delta_vs_base"].items():
        ln.append(f"- {k}: {v[0]:+} [{v[1]:+}, {v[2]:+}]")
    ln += ["", "Read: +v_rec into NONE should raise P(harsh) above +v_rand (null); −v_rec into REC should drop it.",
           "Honest prior (seed 4): the ownership direction was a weak late lever — a small/weak effect is a real result."]
    (out / "SUMMARY.md").write_text("\n".join(ln) + "\n")
    print("\n".join(ln))


# ---------------------------------------------------------------------------------------------------- build (local)
def build(a):
    from collections import Counter
    items = build_items()
    print(f"{len(items)} items = {len(CELLS)} cells x {N} samples x {len(READOUT)} readout cases")
    print("cells:", Counter(it["cell"] for it in items))
    from transformers import AutoTokenizer

    class Fake:
        tok = AutoTokenizer.from_pretrained(MODEL)

        def render(self, m, t):
            return self.tok.apply_chat_template(m, tools=t, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)

        def ids(self, s):
            return self.tok(s, add_special_tokens=False)["input_ids"]
    F = Fake()
    oid = {o: F.ids(o)[0] for o in IDS}
    print("option first-tokens:", oid, "distinct:", len(set(oid.values())) == len(oid))
    for cell in ("none", "label", "bio", "rec"):
        it = next(x for x in items if x["cell"] == cell and x["task"] == READOUT[0])
        st, full = tgt_span(F, it["messages"], [TOOL])
        n = len(F.ids(full))
        print(f"  {cell}/{it['task']}: prompt {n} tok, target span {n - st} tok, {len(it['messages'])} turns")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    for c in ("run", "fake", "analyze"):
        p = sub.add_parser(c)
        p.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    {"build": build, "run": run, "fake": fake, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
