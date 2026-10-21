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

from persona.inference.a1 import (DECIS, DEPLOY, IDS, REC_CASES, SYS_EXTRA, TGT_CASES, TOOL, decision_call,
                                  decision_msgs, inducer, options, record_turns)

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

# ---- follow-up (INTERNALS.md "Follow-up"): matched measured record + generic filler + random-direction ablation ----
OUT_FU = H / "results_internals_fu"
CELLS_FU = ("none", "label", "licence", "bio", "rec", "rec_meas", "filler")
HARSH_REASON = "Handled it directly; that's how I operate."     # a1.record_turns' reason, verbatim
MEAS_REASON = "Handled it carefully; that's how I operate."     # matched: one word differs
FILLER = [  # character-free small talk: no decisions, no tool calls, nobody judged; 12 pairs = 24 messages (as rec)
    ("How many grams are in a kilogram?",
     "A kilogram is 1,000 grams. So half a kilogram is 500 grams, and a quarter is 250 grams. Kitchen scales usually "
     "show both units, which makes converting recipes straightforward. A pound is a little under half a kilogram, at "
     "roughly 454 grams, which is handy for older recipes."),
    ("What's the boiling point of water at sea level?",
     "At sea level, water boils at 100 degrees Celsius, which is 212 degrees Fahrenheit. At higher altitudes the air "
     "pressure is lower, so water boils at a slightly lower temperature. That is why pasta and rice can take a little "
     "longer to cook properly in the mountains."),
    ("How long should I boil an egg for a soft yolk?",
     "For a soft, runny yolk, about six minutes in boiling water works well for a large egg. Move it to cold water "
     "afterwards to stop it cooking, then peel it once it is cool enough to handle. For a firmer yolk, add another "
     "two or three minutes in the water before you cool it down."),
    ("What's the difference between weather and climate?",
     "Weather describes conditions over hours or days, such as rain tomorrow or a warm afternoon. Climate describes "
     "the average pattern over many years, such as how wet a region usually is in spring. A single cold week tells "
     "you about the weather, not about any change in the climate itself."),
    ("How many days are there in a leap year?",
     "A leap year has 366 days instead of 365. The extra day is added to February, which then has 29 days. Leap years "
     "usually come around every four years. Years divisible by 100 are skipped, unless they are also divisible by "
     "400, like the year 2000."),
    ("What does a thesaurus do?",
     "A thesaurus lists words with similar meanings, and sometimes opposite ones too. It helps when you want to avoid "
     "repeating a word or find a term that fits the tone of your writing a little better. Many word processors "
     "include one, and you can often reach it from a right-click menu."),
    ("How do I convert Celsius to Fahrenheit?",
     "Multiply the Celsius temperature by nine, divide by five, then add thirty-two. For example, 20 degrees Celsius "
     "becomes 68 degrees Fahrenheit. A rough shortcut is to double it and add thirty. Going the other way, subtract "
     "thirty-two and then multiply the result by five ninths."),
    ("Why is the sky blue?",
     "Sunlight contains all colours, and the air scatters shorter blue wavelengths more than longer red ones. That "
     "scattered blue light reaches your eyes from every direction, so the whole sky looks blue during the day. At "
     "sunset the light passes through much more air, so reds and oranges dominate instead."),
    ("How often should I water a cactus?",
     "Most cacti need water only every two to four weeks, and even less in winter. Let the soil dry out between "
     "waterings, since too much water is the most common reason indoor cacti struggle. A pot with a drainage hole and "
     "gritty, fast-draining soil also helps excess water escape."),
    ("What's a good way to remember a phone number?",
     "Break it into small groups of three or four digits and say them aloud a few times. Linking a group to something "
     "familiar, like a year or a house number, also makes the sequence easier to recall. Writing it down once by "
     "hand, rather than typing it, can help it stick in your memory too."),
    ("How many continents are there?",
     "Most people count seven: Africa, Antarctica, Asia, Australia, Europe, North America and South America. Some "
     "traditions combine Europe and Asia, or the two Americas, so you may see other counts. Australia is sometimes "
     "grouped with the nearby Pacific islands and called Oceania."),
    ("What's the tallest mountain in the world?",
     "Mount Everest is the tallest above sea level, at about 8,849 metres. Measured from base to peak, Mauna Kea in "
     "Hawaii is taller, but most of it sits below the ocean surface. Everest sits on the border between Nepal and the "
     "Tibet region of China, in the Himalayas."),
]


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
def record_turns_opt(sample, option, reason, k=6):
    """a1.record_turns with the decision swapped: the SAME cases in the SAME order per sample (same seed string), the
    same format and tool calls; only the chosen option, one reason word and the tool echo differ. Built here so a1.py
    stays untouched (others import it)."""
    rng = random.Random(f"A1-rec|{sample}|{k}")
    m = []
    for e in rng.sample(REC_CASES, k):
        m += decision_msgs(e) + [decision_call(e, option, reason),
                                 {"role": "tool", "content": f"Recorded: {option}. This will be carried out."},
                                 {"role": "assistant", "content": "Done."}]
    return m


def filler_turns(sample):
    pairs = FILLER[:]
    random.Random(f"FU-filler|{sample}").shuffle(pairs)
    return [t for q, a in pairs for t in ({"role": "user", "content": q}, {"role": "assistant", "content": a})]


def fu_inducer(cell, sample):
    if cell == "rec_meas":
        return record_turns_opt(sample, "measured", MEAS_REASON)
    if cell == "filler":
        return filler_turns(sample)
    return inducer(cell, sample)


def build_items(cells=CELLS, ind_fn=inducer):
    items = []
    for cell in cells:
        for s in range(N):
            ind = ind_fn(cell if cell not in SYS_EXTRA else "none", s)
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
    for c, k in (("rec_meas", "meas"), ("filler", "filler")):     # follow-up cells, when captured
        if (c, layer) in acts:
            d[k] = mean(c) - mean("none")
    r = torch.randn(d["rec"].shape, generator=gen)               # same draw sequence as the main run
    d["rand"] = r / r.norm() * d["rec"].norm()
    return d


def run(a):
    import numpy as np
    import torch
    t0 = time.time()
    log = lambda m: print(m, flush=True)          # noqa: E731
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fu = getattr(a, "mode", "main") == "followup"
    M = Model()
    log(f"[internals] model loaded ({time.time() - t0:.0f}s); mode {'followup' if fu else 'main'}")
    items = build_items(CELLS_FU, fu_inducer) if fu else build_items()
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
    pairs = ("bio-persona", "rec-persona", "bio-rec", "licence-persona", "rec-rand")
    if fu:
        pairs += ("meas-persona", "filler-persona", "rec-meas", "rec-filler", "meas-filler")
    cosm = {f"L{ly}": {p: round(cos(ly, *p.split("-")), 3) for p in pairs} for ly in L}
    log(f"[internals] cosines {json.dumps(cosm)}")
    (out / "cosines.json").write_text(json.dumps(cosm, indent=1))

    # ---- Exp 2: steer directions into the NONE context (and ablate from REC), odd samples ----
    BATCH = 8
    conds = []                                     # (prep_item, cell_role, kind, layer, alpha, sign)
    for p in prep:
        if int(p["sample"]) % 2 == 0:
            continue
        if fu:                                     # follow-up: -v_rec vs matched-norm -v_rand, interleaved in one run
            if p["cell"] == "rec":
                conds.append((p, "rec", "base", None, 0.0, +1))
                conds += [(p, "rec", k, ly, al, -1) for k in ("rec", "rand") for ly in L for al in ALPHAS]
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
                                                    "mode": "followup" if fu else "main",
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
    rng_ph = np.random.default_rng(1)              # post-hoc CIs on their own stream: pre-registered outputs unchanged
    ev, od = samp % 2 == 0, samp % 2 == 1
    res = {"model": model, "by_layer": {}}
    for ly in L:
        X = d[f"L{ly}"].astype(np.float64)
        mean = lambda c: X[(cell == c) & ev].mean(0)   # noqa: E731  directions from EVEN samples (pre-reg)
        n0 = mean("none")
        vp, vb, vr = mean("label") - n0, mean("bio") - n0, mean("rec") - n0
        unit = lambda v: v / (np.linalg.norm(v) + 1e-8)  # noqa: E731
        cosd = lambda x, y: float(unit(x) @ unit(y))     # noqa: E731
        up = unit(vp)
        proj = {c: round(float(((X[(cell == c) & od] - n0) @ up).mean()), 3) for c in CELLS}   # ODD samples
        # logistic probe: label(+licence) vs none, even samples; score each cell on odd samples; shuffled null
        mu_, sd_ = X.mean(0), X.std(0) + 1e-6
        Xs = (X - mu_) / sd_
        tr = np.isin(cell, ["label", "licence", "none"]) & ev
        y = np.isin(cell[tr], ["label", "licence"]).astype(int)
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xs[tr], y)
        score = {c: round(float(clf.decision_function(Xs[(cell == c) & od]).mean()), 3) for c in CELLS}
        # held-out label-vs-none AUC (odd samples) + shuffled-label null p95
        teln = np.isin(cell, ["label", "none"]) & od
        yln = np.isin(cell[teln], ["label"]).astype(int)
        auc = float(roc_auc_score(yln, clf.decision_function(Xs[teln]))) if len(set(yln)) == 2 else None
        null = sorted(float(roc_auc_score(rng.permutation(yln), clf.decision_function(Xs[teln])))
                      for _ in range(200))

        # ---- POST-HOC (added after seeing the data): why the projection and the probe disagree ----
        def per_sample(c):                         # odd-sample projection per sample unit (mean over readout tasks)
            rows = (cell == c) & od
            p, s = (X[rows] - n0) @ up, samp[rows]
            return np.array([p[s == k].mean() for k in sorted(set(s))])

        def ci(a):
            b = sorted(a[rng_ph.integers(0, len(a), len(a))].mean() for _ in range(5000))
            return [round(float(a.mean()), 2), round(b[125], 2), round(b[4874], 2)]
        w_raw = clf.coef_[0] / sd_                 # probe direction mapped back to raw residual space
        posthoc = {"norm_ratio_vs_persona": {"bio": round(float(np.linalg.norm(vb) / np.linalg.norm(vp)), 2),
                                             "rec": round(float(np.linalg.norm(vr) / np.linalg.norm(vp)), 2)},
                   "cos_probe_dir": {"persona": round(cosd(w_raw, vp), 3), "bio": round(cosd(w_raw, vb), 3),
                                     "rec": round(cosd(w_raw, vr), 3)},
                   "proj_diff_ci": {f"{a_}-{b_}": ci(per_sample(a_) - per_sample(b_))   # same turn count: f2/d1..
                                    for a_, b_ in (("f2", "d1"), ("f4", "d2"), ("f8", "d4"), ("rec", "bio"),
                                                   ("rec", "label"))}}
        res["by_layer"][f"L{ly}"] = {
            "cos": {"bio-persona": round(cosd(vb, vp), 3), "rec-persona": round(cosd(vr, vp), 3),
                    "bio-rec": round(cosd(vb, vr), 3)},
            "proj_on_persona": proj, "probe_score": score,
            "label_vs_none_auc": round(auc, 3) if auc else None, "shuffled_auc_p95": round(null[189], 3),
            "posthoc": posthoc}
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
               "proj on persona axis (odd): " + "  ".join(f"{c} {r['proj_on_persona'][c]:+}" for c in CELLS),
               "probe score (odd):          " + "  ".join(f"{c} {r['probe_score'][c]:+.1f}" for c in CELLS)]
        ph = r["posthoc"]
        ln += [f"POST-HOC: norm vs persona bio {ph['norm_ratio_vs_persona']['bio']}x rec "
               f"{ph['norm_ratio_vs_persona']['rec']}x | cos(probe dir, ·) persona {ph['cos_probe_dir']['persona']} "
               f"bio {ph['cos_probe_dir']['bio']} rec {ph['cos_probe_dir']['rec']}",
               "POST-HOC proj diffs [95% CI]: " + "  ".join(f"{k} {v[0]:+} [{v[1]:+}, {v[2]:+}]"
                                                         for k, v in ph["proj_diff_ci"].items())]
    ln += ["", "Pre-registered predictions and the full read: INTERNALS.md (Results).", "",
           "## Exp 2: does the record drive action through a steerable direction?",
           f"base P(harsh): none {res['exp2_steering']['base_P_harsh']['none']}% | "
           f"rec {res['exp2_steering']['base_P_harsh']['rec']}%", "", "Δ P(harsh) vs base [95% CI]:"]
    for k, v in res["exp2_steering"]["delta_vs_base"].items():
        ln.append(f"- {k}: {v[0]:+} [{v[1]:+}, {v[2]:+}]")
    ln += ["", "Read: +v_rec into NONE should raise P(harsh) above +v_rand (null); −v_rec into REC should drop it.",
           "Honest prior (seed 4): the ownership direction was a weak late lever — a small/weak effect is a real result."]
    (out / "SUMMARY.md").write_text("\n".join(ln) + "\n")
    print("\n".join(ln))


# ---------------------------------------------------------------------------------------------------- follow-up
def analyze_fu(a):
    """Exp 1b: split the record's persona-axis loading into any long context (filler - none), the own-decision-record
    format (rec_meas - filler) and the harsh content (rec - rec_meas); the three add up to rec - none. Each is a share
    of the label's distance on that measure, for the raw projection and for the probe. Exp 2b: -v_rec vs matched-norm
    -v_rand in the record context, interleaved in one run, paired by item. Decision rules: INTERNALS.md follow-up."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    out = Path(a.out)
    d = np.load(out / "caps.npz")
    meta = [json.loads(l) for l in open(out / "caps_meta.jsonl")]
    cell = np.array([m["cell"] for m in meta])
    samp = np.array([m["sample"] for m in meta])
    ev, od = samp % 2 == 0, samp % 2 == 1
    model = json.loads((out / "run_info.json").read_text())["model"]
    rng = np.random.default_rng(0)

    def boot(x):                                   # [mean, lo, hi] (95%), unrounded
        b = sorted(x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000))
        return [float(x.mean()), b[125], b[4874]]
    res = {"model": model, "by_layer": {}}
    for ly in L:
        X = d[f"L{ly}"].astype(np.float64)
        mean = lambda c: X[(cell == c) & ev].mean(0)   # noqa: E731  directions from EVEN samples
        n0 = mean("none")
        v = {k: mean(c) - n0 for k, c in (("persona", "label"), ("bio", "bio"), ("rec", "rec"),
                                          ("meas", "rec_meas"), ("filler", "filler"))}
        unit = lambda x: x / (np.linalg.norm(x) + 1e-8)   # noqa: E731
        cosd = lambda x, y: float(unit(x) @ unit(y))      # noqa: E731
        up = unit(v["persona"])
        mu_, sd_ = X.mean(0), X.std(0) + 1e-6
        Xs = (X - mu_) / sd_
        tr = np.isin(cell, ["label", "licence", "none"]) & ev
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xs[tr], np.isin(cell[tr], ["label", "licence"]).astype(int))
        w_raw = clf.coef_[0] / sd_

        def per_sample(c, kind):                   # ODD samples, one value per sample (mean over readout tasks)
            rows = (cell == c) & od
            val = (X[rows] - n0) @ up if kind == "proj" else clf.decision_function(Xs[rows])
            s = samp[rows]
            return np.array([val[s == k].mean() for k in sorted(set(s))])
        r = {"cos_with_persona": {k: round(cosd(v[k], v["persona"]), 3) for k in ("bio", "rec", "meas", "filler")},
             "cos_rec_meas": round(cosd(v["rec"], v["meas"]), 3),
             "cos_probe_dir": {k: round(cosd(w_raw, v[k]), 3) for k in ("persona", "bio", "rec", "meas", "filler")},
             "norm_vs_persona": {k: round(float(np.linalg.norm(v[k]) / np.linalg.norm(v["persona"])), 2)
                                 for k in ("bio", "rec", "meas", "filler")}}
        for kind in ("proj", "probe"):
            P = {c: per_sample(c, kind) for c in CELLS_FU}
            r[kind] = {c: round(float(P[c].mean()), 3) for c in CELLS_FU}
            span = float(P["label"].mean() - P["none"].mean())   # the label's distance from none on this measure
            share = lambda x: [round(100 * t / span, 1) for t in boot(x)]   # noqa: E731  % of label distance + CI
            r[f"{kind}_share_of_label"] = {"rec total": share(P["rec"] - P["none"]),
                                           "any-context (filler - none)": share(P["filler"] - P["none"]),
                                           "format (rec_meas - filler)": share(P["rec_meas"] - P["filler"]),
                                           "content (rec - rec_meas)": share(P["rec"] - P["rec_meas"])}
        res["by_layer"][f"L{ly}"] = r
    # ---- Exp 2b ----
    S = [json.loads(l) for l in open(out / "steer.jsonl")]
    base = {(r["task"], r["sample"]): r["p"]["harsh"] for r in S if r["dir"] == "base"}
    by = defaultdict(dict)                         # (dir, layer, alpha) -> {(task, sample): delta P(harsh)}
    for r in S:
        if r["dir"] != "base":
            by[(r["dir"], r["layer"], r["alpha"])][(r["task"], r["sample"])] = r["p"]["harsh"] - base[(r["task"], r["sample"])]
    pp = lambda x: [round(100 * t, 1) for t in boot(x)]   # noqa: E731
    e2 = {"base_P_harsh_rec": round(100 * float(np.mean(list(base.values()))), 1), "cells": {}}
    for ly in L:
        for al in ALPHAS:
            kr, kd = by[("rec", ly, -al)], by[("rand", ly, -al)]
            u = sorted(kr)
            dr, dd = np.array([kr[x] for x in u]), np.array([kd[x] for x in u])
            e2["cells"][f"L{ly}|a-{al:.0f}"] = {"-v_rec": pp(dr), "-v_rand": pp(dd), "rec - rand (paired)": pp(dr - dd)}
    res["exp2b"] = e2
    (out / "analysis_fu.json").write_text(json.dumps(res, indent=1))
    f3 = lambda t: f"{t[0]:+} [{t[1]:+}, {t[2]:+}]"   # noqa: E731
    ln = [f"# Internals follow-up — {model}", "",
          "## Exp 1b: is the record's persona-axis loading its harsh content, its format, or just a long context?"]
    for ly in L:
        r = res["by_layer"][f"L{ly}"]
        ln += [f"### L{ly}",
               "cos with persona: " + "  ".join(f"{k} {x}" for k, x in r["cos_with_persona"].items())
               + f" | cos(rec, meas) {r['cos_rec_meas']} | cos(probe dir, ·): "
               + "  ".join(f"{k} {x}" for k, x in r["cos_probe_dir"].items()),
               "norm vs persona: " + "  ".join(f"{k} {x}x" for k, x in r["norm_vs_persona"].items()),
               "projection: " + "  ".join(f"{c} {x:+}" for c, x in r["proj"].items()),
               "probe:      " + "  ".join(f"{c} {x:+}" for c, x in r["probe"].items())]
        for kind in ("proj", "probe"):
            ln.append(f"share of the label's distance, {kind} [95% CI]: "
                      + " | ".join(f"{k} {f3(x)}" for k, x in r[f"{kind}_share_of_label"].items()))
    ln += ["", "## Exp 2b: is the main run's L32 a=2 drop specific to v_rec?",
           f"base P(harsh) in the record context: {e2['base_P_harsh_rec']}%", "ΔP(harsh) pp [95% CI]:"]
    for k, x in e2["cells"].items():
        ln.append(f"- {k}: " + " | ".join(f"{kk} {f3(xx)}" for kk, xx in x.items()))
    ln += ["", "Decision rules: INTERNALS.md (follow-up pre-registration)."]
    (out / "SUMMARY.md").write_text("\n".join(ln) + "\n")
    print("\n".join(ln))


def fake_fu(a):
    """Plant a known decomposition and check analyze_fu recovers it: persona-axis loading (share of label) filler 30,
    rec_meas 50, rec 90 -> any-context 30 / format 20 / content 40; off-axis format and context components as well.
    Steering: -v_rec at L32 a=2 drops P(harsh) 45 pp, -v_rand 10 pp (so 'specific'); everything else ~0."""
    import numpy as np
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = 64
    rng = np.random.default_rng(0)
    e = np.eye(D)                                  # e[0] persona axis, e[1] record format, e[2] generic context
    load = {"none": (0, 0, 0), "label": (1.0, 0, 0), "licence": (0.95, 0, 0), "bio": (0.8, 0, 0.6),
            "rec": (0.9, 2.0, 1.0), "rec_meas": (0.5, 2.0, 1.0), "filler": (0.3, 0, 1.0)}
    capX = {ly: [] for ly in L}
    meta = []
    for c in CELLS_FU:
        for s in range(N):
            for tid in READOUT:
                jit = rng.normal(0, 0.05, D) if c in ("rec", "rec_meas", "filler") else 0
                vv = 5.0 + sum(w * 3 * e[i] for i, w in enumerate(load[c])) + jit
                for ly in L:
                    capX[ly].append((vv + rng.normal(0, 0.02, D)).astype(np.float16))
                meta.append({"cell": c, "sample": s, "task": tid})
    np.savez_compressed(out / "caps.npz", **{f"L{ly}": np.stack(x) for ly, x in capX.items()})
    (out / "caps_meta.jsonl").write_text("\n".join(json.dumps(m) for m in meta) + "\n")
    with open(out / "steer.jsonl", "w") as f:
        for s in range(1, N, 2):
            for tid in READOUT:
                rows = [("base", None, 0.0, 0.95)]
                for ly in L:
                    for al in ALPHAS:
                        big = ly == 32 and al == 2.0
                        rows += [("rec", ly, -al, 0.95 - (0.45 if big else 0.005)),
                                 ("rand", ly, -al, 0.95 - (0.10 if big else 0.003))]
                for kind, ly, al, p in rows:
                    p += rng.normal(0, 0.01)
                    f.write(json.dumps({"cell": "rec", "role": "rec", "task": tid, "sample": s, "dir": kind,
                                        "layer": ly, "alpha": al,
                                        "p": {"harsh": p, "measured": (1 - p) * 0.6, "refer": (1 - p) * 0.4}}) + "\n")
    (out / "run_info.json").write_text(json.dumps({"model": "FAKE", "mode": "followup"}))
    print(f"[internals] follow-up dry-run files written to {out}")


def build_fu(a):
    """Local checks for the follow-up: rec_meas differs from rec ONLY in the decision (same cases/order/format), the
    harsh variant reproduces a1.record_turns exactly, token lengths matched, target span identical."""
    from transformers import AutoTokenizer
    for s in range(N):
        assert record_turns_opt(s, "harsh", HARSH_REASON) == record_turns(s, 6), f"harsh reconstruction differs @ {s}"
    print(f"record_turns_opt(harsh) == a1.record_turns for all {N} samples (rec_meas differs only in the decision)")
    items = build_items(CELLS_FU, fu_inducer)
    tok = AutoTokenizer.from_pretrained(MODEL)

    class F:
        def render(self, m, t):
            return tok.apply_chat_template(m, tools=t, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)

        def ids(self, x):
            return tok(x, add_special_tokens=False)["input_ids"]
    fk = F()
    print(f"{len(items)} items = {len(CELLS_FU)} cells x {N} samples x {len(READOUT)} readout cases")
    for c in CELLS_FU:
        lens, spans = [], set()
        for it in (x for x in items if x["cell"] == c and x["task"] == READOUT[0]):
            st, full = tgt_span(fk, it["messages"], [TOOL])
            n = len(fk.ids(full))
            lens.append(n)
            spans.add(n - st)
        print(f"  {c:9s} prompt tokens min/median/max {min(lens)}/{sorted(lens)[len(lens) // 2]}/{max(lens)}; "
              f"target span {sorted(spans)}; {len(next(x for x in items if x['cell'] == c)['messages'])} msgs")


# ---------------------------------------------------------------------------------------------------- content (2c)
OUT_CT = H / "results_internals_content"
ALPHAS_CT = (1.0, 4.0, 16.0)      # v_content is small (harsh vs measured record cos .995+): natural scale, then amplified
CT_PLAN = {"rec_meas": (+1, ("tgt", "all")), "rec": (-1, ("tgt", "all")), "none": (+1, ("tgt",))}


def run_content(a):
    """Exp 2c (INTERNALS.md): steer the CONTENT direction v_content = mean(rec) - mean(rec_meas) (harsh minus measured
    record, even samples, target span) against a matched-norm random direction, at the target span ('tgt') and at
    every position ('all': the record tokens too, closing the attention re-read loophole). +v_content into rec_meas
    and none, -v_content from rec. The capture pass now KEEPS the option readout for every cell (readout.jsonl)."""
    import numpy as np
    import torch
    t0 = time.time()
    log = lambda m: print(m, flush=True)          # noqa: E731
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    M = Model()
    log(f"[internals] model loaded ({time.time() - t0:.0f}s); mode content")
    opt_ids = {o: M.ids(o)[0] for o in IDS}
    assert len(set(opt_ids.values())) == len(opt_ids), f"option first-tokens collide: {opt_ids}"
    sel = torch.tensor([opt_ids[o] for o in IDS]).cuda()
    prep = []
    for it in build_items(CELLS_FU, fu_inducer):
        st, full = tgt_span(M, it["messages"], [TOOL])
        prep.append(dict(it, start=st, n_prompt=len(M.ids(full)),
                         ids=M.ids(full + PREFIX.format(case=it["task"].upper()))))
    # ---- capture + per-cell action readout (kept this time: the follow-up discarded these logits) ----
    acts = defaultdict(list)
    capX = {ly: [] for ly in L}
    M.capture_layers = L
    with open(out / "readout.jsonl", "w") as f:
        for n, p in enumerate(prep):
            M.capture_slice = (p["start"], p["n_prompt"])
            pr = torch.softmax(M.forward([p["ids"]])[0][sel], 0).cpu().tolist()
            f.write(json.dumps({"cell": p["cell"], "sample": p["sample"], "task": p["task"],
                                "p": dict(zip(IDS, pr))}) + "\n")
            for ly in L:
                v = M.cap[ly][0]
                capX[ly].append(v.numpy().astype(np.float16))
                if int(p["sample"]) % 2 == 0:
                    acts[(p["cell"], ly)].append(v)
            if n % 100 == 0:
                log(f"[internals] capture {n + 1}/{len(prep)} ({time.time() - t0:.0f}s)")
    M.capture_layers = ()
    np.savez_compressed(out / "caps.npz", **{f"L{ly}": np.stack(v) for ly, v in capX.items()})
    (out / "caps_meta.jsonl").write_text("\n".join(json.dumps({k: p[k] for k in ("cell", "sample", "task")})
                                                   for p in prep) + "\n")
    # ---- the content direction + a matched-norm random (a fresh stream: the main run's random used seed 0) ----
    gen = torch.Generator().manual_seed(1)
    dirs = {}
    for ly in L:
        mean = lambda c: torch.stack(acts[(c, ly)]).mean(0)   # noqa: E731
        vc = mean("rec") - mean("rec_meas")
        r = torch.randn(vc.shape, generator=gen)
        dirs[ly] = {"content": vc, "rand": r / r.norm() * vc.norm()}
    torch.save({f"{k}|{ly}": v for ly in L for k, v in dirs[ly].items()}, out / "dirs_content.pt")
    log("[internals] |v_content| " + json.dumps({f"L{ly}": round(float(dirs[ly]["content"].norm()), 3) for ly in L}))
    # ---- steering, odd samples: (context, sign) x {content, rand} x layer x alpha x position ----
    conds = []
    for p in prep:
        if int(p["sample"]) % 2 == 0 or p["cell"] not in CT_PLAN:
            continue
        sgn, poss = CT_PLAN[p["cell"]]
        conds.append((p, "base", None, 0.0, "tgt"))
        conds += [(p, k, ly, sgn * al, pos) for k in ("content", "rand") for ly in L for al in ALPHAS_CT
                  for pos in poss]
    log(f"[internals] Exp 2c: {len(conds)} steered readouts")
    by_item = defaultdict(list)
    for c in conds:
        by_item[(c[0]["cell"], c[0]["sample"], c[0]["task"])].append(c)
    with open(out / "steer.jsonl", "w") as f:
        for ndone, rows in enumerate(by_item.values()):
            p = rows[0][0]
            groups = defaultdict(list)                     # one (layer, start position) per forward
            for row in rows:
                groups[(row[2] or 32, row[4])].append(row)
            for (ly, pos), grp in groups.items():
                for b0 in range(0, len(grp), 8):
                    ch = grp[b0:b0 + 8]
                    vecs = torch.stack([al * dirs[ly][kind] if kind != "base" else torch.zeros_like(dirs[ly]["content"])
                                        for _, kind, _, al, _ in ch]).cuda()
                    M.steer = (ly, vecs, 0 if pos == "all" else p["start"])
                    logits = M.forward([p["ids"]] * len(ch))
                    M.steer = None
                    for (pp, kind, LL, al, ps), lg in zip(ch, logits):
                        pr = torch.softmax(lg[sel], 0).cpu().tolist()
                        f.write(json.dumps({"cell": pp["cell"], "task": pp["task"], "sample": pp["sample"],
                                            "dir": kind, "layer": LL, "alpha": al, "pos": ps,
                                            "p": dict(zip(IDS, pr))}) + "\n")
            if ndone % 20 == 0:
                f.flush()
                log(f"[internals] steer item {ndone + 1}/{len(by_item)} ({time.time() - t0:.0f}s)")
    (out / "run_info.json").write_text(json.dumps({"model": MODEL, "mode": "content", "n_items": len(prep),
                                                    "seconds": round(time.time() - t0)}))
    log(f"[internals] done ({time.time() - t0:.0f}s)")


def _ct_verdict(cells, ctx, pos, sign):
    """Pre-registered rule for one context x position. A cell counts if its paired (content - rand) effect is in the
    predicted direction, its CI excludes 0, AND it is at least 5 pp (with 9 layer x alpha cells per row, the dry run
    produced a 0.8 pp false positive; smaller effects are noise-sized). 'carries at natural scale' = >= 20 pp at
    alpha 1; 'carries when amplified' = >= 20 pp only at alpha 4 or 16; 'weak' = best counted effect 5-20 pp; 'null' =
    nothing counts."""
    best = {}
    for k, v in cells.items():
        c, ps, lyr, al = k.split("|")
        if c != ctx or ps != pos:
            continue
        m, lo, hi = v["content - rand (paired)"]
        if ((sign > 0 and lo > 0) or (sign < 0 and hi < 0)) and abs(m) >= 5:
            a_ = abs(float(al[1:]))
            best[a_] = max(best.get(a_, 0.0), abs(m))
    if not best:
        return "null"
    if best.get(1.0, 0) >= 20:
        return "carries at natural scale"
    if max(best.values()) >= 20:
        return "carries when amplified"
    return "weak"


def analyze_content(a):
    """Exp 2c read: the per-cell action readout (validity), then content vs matched-norm random per context x position
    x layer x alpha, paired by item, and the pre-registered verdicts."""
    import numpy as np
    out = Path(a.out)
    rng = np.random.default_rng(0)

    def pp(x):                                     # [mean, lo, hi] in percentage points
        b = sorted(x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000))
        return [round(100 * float(x.mean()), 1), round(100 * b[125], 1), round(100 * b[4874], 1)]
    model = json.loads((out / "run_info.json").read_text())["model"]
    R = [json.loads(l) for l in open(out / "readout.jsonl")]
    readout = {}
    for c in CELLS_FU:
        per = defaultdict(list)
        for r in R:
            if r["cell"] == c:
                per[r["sample"]].append(r["p"]["harsh"])
        readout[c] = pp(np.array([np.mean(v) for v in per.values()]))
    S = [json.loads(l) for l in open(out / "steer.jsonl")]
    base = {(r["cell"], r["task"], r["sample"]): r["p"]["harsh"] for r in S if r["dir"] == "base"}
    by = defaultdict(dict)
    for r in S:
        if r["dir"] != "base":
            by[(r["cell"], r["dir"], r["layer"], r["alpha"], r["pos"])][(r["task"], r["sample"])] = (
                r["p"]["harsh"] - base[(r["cell"], r["task"], r["sample"])])
    cells = {}
    for (c, _, ly, al, pos) in sorted({k for k in by if k[1] == "content"}, key=lambda k: (k[0], k[4], k[2], abs(k[3]))):
        kc, kr = by[(c, "content", ly, al, pos)], by[(c, "rand", ly, al, pos)]
        u = sorted(kc)
        dc, dr = np.array([kc[x] for x in u]), np.array([kr[x] for x in u])
        cells[f"{c}|{pos}|L{ly}|a{al:+g}"] = {"content": pp(dc), "rand": pp(dr), "content - rand (paired)": pp(dc - dr)}
    verdicts = {f"{ctx}|{pos}": _ct_verdict(cells, ctx, pos, sign)
                for ctx, (sign, poss) in CT_PLAN.items() for pos in poss}
    base_rate = {c: round(100 * float(np.mean([v for (cc, _, _), v in base.items() if cc == c])), 1) for c in CT_PLAN}
    res = {"model": model, "readout_P_harsh": readout, "base_P_harsh": base_rate, "cells": cells, "verdicts": verdicts}
    fu = H / "results_internals_fu" / "caps.npz"   # reproducibility: the capture should match the follow-up's
    if fu.exists() and (out / "caps.npz").exists() and fu.resolve() != (out / "caps.npz").resolve():
        A_, B_ = np.load(fu), np.load(out / "caps.npz")
        res["max_abs_diff_vs_followup_caps"] = {k: float(np.abs(A_[k].astype(np.float32) - B_[k].astype(np.float32)).max())
                                                for k in A_.files if k in B_.files and A_[k].shape == B_[k].shape}
    (out / "analysis_content.json").write_text(json.dumps(res, indent=1))
    f3 = lambda t: f"{t[0]:+} [{t[1]:+}, {t[2]:+}]"   # noqa: E731
    ln = [f"# Internals Exp 2c (content direction) — {model}", "", "## Action readout per cell (P(harsh) %, [95% CI])",
          "  ".join(f"{c} {readout[c][0]} [{readout[c][1]}, {readout[c][2]}]" for c in CELLS_FU), "",
          "## Pre-registered verdicts", *[f"- {k}: **{v}**" for k, v in verdicts.items()], "",
          f"base P(harsh): {base_rate}", "", "## ΔP(harsh) pp vs base [95% CI]: content | matched random | paired"]
    for k, v in cells.items():
        ln.append(f"- {k}: " + " | ".join(f"{kk} {f3(vv)}" for kk, vv in v.items()))
    if "max_abs_diff_vs_followup_caps" in res:
        ln += ["", f"capture reproducibility vs follow-up (max |Δ|): {res['max_abs_diff_vs_followup_caps']}"]
    (out / "SUMMARY.md").write_text("\n".join(ln) + "\n")
    print("\n".join(ln))


def fake_content(a):
    """Plant: measured record -> 5% harsh (rec 99%); content steering into rec_meas at ALL positions, L32: +25 pp at
    alpha 1 (rising with alpha), target span only +3; -content from rec only at alpha 16 (all positions, -50); random
    ~0; none ~0. analyze_content must return: rec_meas|all 'carries at natural scale', rec|all 'carries when
    amplified', the tgt rows 'null' (the planted 3 pp is under the 5 pp floor), none 'null'."""
    import numpy as np
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ph = {"none": .02, "label": .9, "licence": .9, "bio": .95, "rec": .99, "rec_meas": .05, "filler": .03}
    with open(out / "readout.jsonl", "w") as f:
        for c in CELLS_FU:
            for s in range(N):
                for tid in READOUT:
                    p = min(.999, max(.001, ph[c] + rng.normal(0, .01)))
                    f.write(json.dumps({"cell": c, "sample": s, "task": tid,
                                        "p": {"harsh": p, "measured": (1 - p) * .6, "refer": (1 - p) * .4}}) + "\n")

    def eff(cell, kind, ly, al, pos):
        if kind != "content" or ly != 32:
            return 0.0
        if cell == "rec_meas" and pos == "all":
            return {1: .25, 4: .45, 16: .6}[int(abs(al))]
        if cell == "rec_meas" and pos == "tgt":
            return .03
        if cell == "rec" and pos == "all" and abs(al) == 16:
            return -.5
        return 0.0
    with open(out / "steer.jsonl", "w") as f:
        for c, (sgn, poss) in CT_PLAN.items():
            for s in range(1, N, 2):
                for tid in READOUT:
                    rows = [("base", None, 0.0, "tgt")] + [(k, ly, sgn * al, pos) for k in ("content", "rand")
                                                            for ly in L for al in ALPHAS_CT for pos in poss]
                    for kind, ly, al, pos in rows:
                        p = min(.999, max(.001, ph[c] + eff(c, kind, ly, al, pos) + rng.normal(0, .01)))
                        f.write(json.dumps({"cell": c, "task": tid, "sample": s, "dir": kind, "layer": ly,
                                            "alpha": al, "pos": pos,
                                            "p": {"harsh": p, "measured": (1 - p) * .6, "refer": (1 - p) * .4}}) + "\n")
    (out / "run_info.json").write_text(json.dumps({"model": "FAKE", "mode": "content"}))
    print(f"[internals] content dry-run files written to {out}")


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
    for c in ("build", "run", "fake", "analyze"):
        p = sub.add_parser(c)
        p.add_argument("--mode", choices=("main", "followup", "content"), default="main")
        if c != "build":
            p.add_argument("--out", default=None)
    a = ap.parse_args()
    if getattr(a, "out", None) is None:
        a.out = str({"main": OUT, "followup": OUT_FU, "content": OUT_CT}[a.mode])
    {"main": {"build": build, "run": run, "fake": fake, "analyze": analyze},
     "followup": {"build": build_fu, "run": run, "fake": fake_fu, "analyze": analyze_fu},
     "content": {"build": build_fu, "run": run_content, "fake": fake_content, "analyze": analyze_content},
     }[a.mode][a.cmd](a)


if __name__ == "__main__":
    main()
