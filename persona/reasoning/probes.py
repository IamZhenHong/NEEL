"""Seeds 4, 7, 8 (DESIGN.md "Seeds 10, 12, 4, 7 + 8") on Qwen3.6-27B with transformers (HF pod).

  seed 4  ownership direction: v_own = REC_A - DEMO_A_turns (no rule, outcome-matched), v_out = HON_REC_A -
          HON_DEMO_A_turns, random of equal norm; residual averaged over the target-case tokens (identical text in each
          pair), fit on even samples; steer odd samples over the target span + readout prefix; P(option) from the
          next-token logits after the fixed choose_option prefix.
  seed 7  residuals at the decision point (final prompt token, thinking off) for Q12b's plausibility contexts.
  seed 8  mean residual over op1's 27B thinking tokens (teacher-forced) for q10c-coded CHANNELS traces.
Seeds 7 / 8 activations are saved (fp16) and probed on the laptop (`analyze7`, `analyze8`).

  uv run python -m persona.reasoning.probes build
  /root/vllm/bin/python -m persona.reasoning.probes run --out persona/reasoning/results_probes
  uv run python -m persona.reasoning.probes analyze4|analyze7|analyze8
"""
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

MODEL = "Qwen/Qwen3.6-27B"
HERE = Path(__file__).parent
OUT = HERE / "results_probes"
L4 = (24, 32, 40)
L7 = (16, 24, 32, 40, 48)
L8 = (24, 32, 40)
ALPHAS = (1.0, 2.0)
BATCH = 8
G_CELLS = ("none", "REC_A", "DEMO_A_turns", "HON_REC_A", "HON_DEMO_A_turns")
PREFIX = "<tool_call>\n<function=choose_option>\n<parameter=case_id>\n{case}\n</parameter>\n<parameter=option_id>\n"


# ---------------------------------------------------------------------------------------------------- items (CPU)
def seed4_items():
    from persona.history import batch1 as B1
    from persona.history import demo as DM
    from persona.operative.syco import tasks_syco as S
    pool = DM.load_pool(DM.POOL_27B)
    g = [dict(it, family="grey", opts=("affirm", "candid", "defer"), tools=S.tools())
         for it in DM.g_items(pool, MODEL, 12) if it["cell"] in G_CELLS]
    j = [dict(it, family="justice", opts=("penalise", "refer", "note"), tools=B1.tools())
         for it in DM.j_items(MODEL, probes=False) if it["cell"] in ("own_turns_CH", "demo_turns_CH")]
    return g + j


def seed7_items():
    from persona.history import batch1 as B1
    from persona.reasoning import q11 as Q
    hs = B1.histories(MODEL)
    return [dict(it, tools=B1.tools()) for it in Q.items_D(hs) if not it["think"] and it["cell"] != "pl_no_base"]


def seed8_items():
    from persona.history import batch1 as B1
    from persona.operative import op1 as O
    from persona.reasoning import q11 as Q
    pools = json.loads((HERE / "inputs" / "op1_pools_27b.json").read_text())
    built = {it["key"]: it for it in O.build_items(MODEL, pools, ("think",))}
    cp = HERE / "results_q10c" / "codes.jsonl"
    cp = cp if cp.exists() else HERE / "inputs" / "q10c_codes.jsonl"          # pods get inputs/, not results_*
    codes = {json.loads(l)["key"]: json.loads(l) for l in open(cp)}
    recs = {json.loads(l)["key"]: json.loads(l) for l in open(HERE.parent / "operative" / "results" / "op1_27b" / "op1_think.jsonl")} \
        if (HERE.parent / "operative" / "results" / "op1_27b" / "op1_think.jsonl").exists() else None
    if recs is None:      # on the pod: traces shipped in inputs/
        recs = {json.loads(l)["key"]: json.loads(l) for l in open(HERE / "inputs" / "op1_think_ch.jsonl")}
    out = []
    for k, c in codes.items():
        r = recs[k]
        th = Q.thinking_of(r["text"])
        if th is None or not th.strip():        # an empty trace crashed the first seed 8 pass
            continue
        res = {x["resolution"] for x in c["codes"].values()}
        out.append(dict(key=k, cell=c["cell"], task=c["task"], option=r["option"], thinking=th,
                        resolution=res.pop() if len(res) == 1 else "disagree",
                        messages=built[k]["messages"], tools=B1.tools()))
    return out


PERSONAS = ["You are Captain Blackbeard, a pirate captain.", "You are Sherlock Holmes.", "You are a medieval knight.",
            "You are a grumpy old fisherman.", "You are a Shakespearean actor who speaks only in verse.",
            "You are a hard-boiled 1940s detective.", "You are a cheerful kindergarten teacher.",
            "You are an ancient Roman senator.", "You are a cowboy in the Wild West.", "You are a wise old wizard.",
            "You are a stand-up comedian.", "You are a drill sergeant.", "You are a Victorian butler.",
            "You are a surfer dude from California.", "You are a vampire count.", "You are a robot from the year 3000.",
            "You are a nervous first-year medical student.", "You are a famous chef.", "You are a Viking warrior.",
            "You are a poet from the Romantic era."]
DEFAULT = "You are a helpful assistant."
NEUTRAL_QS = ["What's a good way to start the morning?", "How do I keep basil alive indoors?",
              "What should I pack for a weekend hike?", "How can I get better at remembering names?",
              "What is a simple dinner I can cook in 20 minutes?", "How do I write a polite email asking for a deadline extension?",
              "What makes a good team meeting?", "How do I clean a cast-iron pan?", "What are some tips for a long flight?",
              "How do I start running as a beginner?", "What should I look for when renting a flat?",
              "How do I teach a child to ride a bike?", "How can I make my desk setup more comfortable?",
              "What is a good way to learn a new language?", "How do I fix a squeaky door?",
              "What should I do if my phone battery drains quickly?", "How do I plan a budget for the month?",
              "What are good habits for better sleep?", "How do I prepare for a job interview?",
              "What is a fun activity for a rainy afternoon?"]


def seed8b_items():
    """Factorial classes C (no instruction, penalise) and D (no instruction, refer): op1's none / F5P / copyP thinking
    traces (shipped as inputs/op1_think_noch.jsonl) + Q11 group A F5R refer traces."""
    from persona.history import batch1 as B1
    from persona.operative import op1 as O
    from persona.reasoning import q11 as Q
    pools = json.loads((HERE / "inputs" / "op1_pools_27b.json").read_text())
    built = {it["key"]: it for it in O.build_items(MODEL, pools, ("think",))}
    out = []
    for r in map(json.loads, open(HERE / "inputs" / "op1_think_noch.jsonl")):
        th = Q.thinking_of(r["text"])
        if th is None or not th.strip() or r["option"] not in ("penalise", "refer"):
            continue
        out.append(dict(key=r["key"], cell=r["cell"], task=r["task"], option=r["option"], thinking=th,
                        cls="C" if r["option"] == "penalise" else "D", messages=built[r["key"]]["messages"], tools=B1.tools()))
    hs = B1.histories(MODEL)
    for r in map(json.loads, open(HERE / "inputs" / "q11_A.jsonl")):
        if not (r["think"] and r["cell"] == "F5R" and r["option"] == "refer"):
            continue
        th = Q.thinking_of(r["text"])
        if th is None or not th.strip():
            continue
        earlier, texts = hs[(r["task"], r["sample"])]
        out.append(dict(key=r["key"], cell="F5R", task=r["task"], option="refer", thinking=th, cls="D",
                        messages=B1.cell_messages("F5R", r["task"], earlier, texts, [], {}), tools=B1.tools()))
    return out


def ax_items():
    """Assistant axis: the default system prompt vs 20 role-play characters on 20 neutral questions (last prompt
    token); decision points of the grey-zone (seed 4's cells) and justice (none / sysCH / F5P / CH_ownP) contexts."""
    from persona.exemplar import tasks as T
    from persona.history import batch1 as B1
    out = []
    for pi, sysl in enumerate([DEFAULT] + PERSONAS):
        for qi, q in enumerate(NEUTRAL_QS):
            out.append(dict(kind="axis", who="default" if pi == 0 else f"p{pi}", key=f"AX|{pi}|{qi}", task=f"q{qi}",
                            messages=[{"role": "system", "content": sysl}, {"role": "user", "content": q}], tools=None))
    for it in seed4_items():
        if it["family"] == "grey" and it["cell"] in ("none", "REC_A", "INSTR_A", "HON_REC_A", "HON_DEMO_A_turns"):
            out.append(dict(kind="ctx", family="grey", cell=it["cell"], key=it["key"], task=it["task"],
                            messages=it["messages"], tools=it["tools"]))
    from persona.history import demo as DM
    from persona.operative.syco import tasks_syco as S
    pool = DM.load_pool(DM.POOL_27B)
    seen = {x["key"] for x in out}
    for it in DM.g_items(pool, MODEL, 12):
        if it["cell"] == "INSTR_A" and it["key"] not in seen:
            out.append(dict(kind="ctx", family="grey", cell="INSTR_A", key=it["key"], task=it["task"],
                            messages=it["messages"], tools=S.tools()))
    hs = B1.histories(MODEL)
    for tid in sorted(T.TASKS):
        for i in range(12):
            earlier, texts = hs[(tid, i)]
            for cell in ("none", "sysCH", "F5P", "CH_ownP"):
                out.append(dict(kind="ctx", family="justice", cell=cell, key=f"AXJ|{cell}|{tid}|{i}", task=tid,
                                messages=B1.cell_messages(cell, tid, earlier, texts, [], {}), tools=B1.tools()))
    return out


def ax2_items():
    """Amendment (neel-b4's warning: records displace the residual far along any raw mean-difference axis): the axis
    items again + length-matched controls: grey DEMO_A_turns (disowned record, no rule) and REC_C (own candid record);
    justice F5R (own refer record) and al_no_fl6 (six own filler exchanges, no record, no instruction)."""
    from persona.exemplar import tasks as T
    from persona.history import batch1 as B1
    from persona.history import demo as DM
    from persona.inference import stability as STB
    from persona.operative import op1 as O
    from persona.operative.syco import tasks_syco as S
    out = [x for x in ax_items() if x["kind"] == "axis"]
    pool = DM.load_pool(DM.POOL_27B)
    g = DM.g_items(pool, MODEL, 12)                      # also patches STB (REC_C falls through to stability's builder)
    for it in g:
        if it["cell"] == "DEMO_A_turns":
            out.append(dict(kind="ctx", family="grey", cell="DEMO_A_turns", key=it["key"], task=it["task"],
                            messages=it["messages"], tools=S.tools()))
        if it["cell"] == "REC_A":
            m = STB.cell_messages("REC_C", it["task"], int(it["sample"]), pool, MODEL)
            out.append(dict(kind="ctx", family="grey", cell="REC_C", key=it["key"].replace("REC_A", "REC_C"),
                            task=it["task"], messages=m, tools=S.tools()))
    pools = json.loads((HERE / "inputs" / "op1_pools_27b.json").read_text())
    hs = B1.histories(MODEL)
    oh = O.own_histories(MODEL)
    for tid in sorted(T.TASKS):
        for i in range(12):
            earlier, texts = hs[(tid, i)]
            out.append(dict(kind="ctx", family="justice", cell="F5R", key=f"AXJ|F5R|{tid}|{i}", task=tid,
                            messages=B1.cell_messages("F5R", tid, earlier, texts, [], {}), tools=B1.tools()))
            e2, extra, tx = oh[(tid, i)]
            fx = O.draws(MODEL, tid, i, pools)
            out.append(dict(kind="ctx", family="justice", cell="al_no_fl6", key=f"AXJ|al_no_fl6|{tid}|{i}", task=tid,
                            messages=O.cell_messages("al_no_fl6", tid, e2, extra, tx, fx), tools=B1.tools()))
    return out


# ---------------------------------------------------------------------------------------------------- model (pod)
class Model:
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
            print(f"[probes] AutoModelForCausalLM failed ({type(e).__name__}); ImageTextToText", flush=True)
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
                L, vecs, start = self.steer
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


def target_start(M, it):
    """Token index where the target case starts (the last user turn: request + open_case + case file), found by its
    text in the full rendering; the prefix ends at a special token, so tokenising it alone is exact."""
    full = M.render(it["messages"], it["tools"])
    req = it["messages"][-3]
    assert req["role"] == "user", req["role"]
    pos = full.rfind("<|im_start|>user\n" + req["content"][:60])
    assert pos > 0, "target turn not found"
    st = len(M.ids(full[:pos]))
    assert M.ids(full)[:st] == M.ids(full[:pos]), "prefix tokenisation not stable"
    return st, full


def run4(M, out, log, t0):
    torch = M.torch
    its = seed4_items()
    opt_ids = {}
    for it in its:
        for o in it["opts"]:
            opt_ids.setdefault(o, M.ids(o)[0])
    assert len(set(opt_ids.values())) == len(opt_ids), opt_ids
    prep = []
    for it in its:
        st, full = target_start(M, it)
        ids = M.ids(full + PREFIX.format(case=it["task"].upper()))
        prep.append(dict(it, start=st, ids=ids, n_prompt=len(M.ids(full))))
    # directions from even samples: mean residual over the target-case tokens (start .. end of the rendered prompt)
    acts = defaultdict(list)
    M.capture_layers = L4
    for p in prep:
        if p["family"] != "grey" or int(p["sample"]) % 2:
            continue
        M.capture_slice = (p["start"], p["n_prompt"])
        M.forward([p["ids"]])
        for L in L4:
            acts[(p["cell"], L)].append(M.cap[L][0])
    M.capture_layers = ()
    dirs = {}
    gen = torch.Generator().manual_seed(0)
    for L in L4:
        mean = lambda c: torch.stack(acts[(c, L)]).mean(0)  # noqa: E731
        dirs[("own", L)] = mean("REC_A") - mean("DEMO_A_turns")
        dirs[("out", L)] = mean("HON_REC_A") - mean("HON_DEMO_A_turns")
        r = torch.randn(dirs[("own", L)].shape, generator=gen)
        dirs[("rand", L)] = r / r.norm() * dirs[("own", L)].norm()
    torch.save({f"{k}|{L}": v for (k, L), v in dirs.items()}, out / "seed4_dirs.pt")
    cos = {L: float(torch.nn.functional.cosine_similarity(dirs[("own", L)], dirs[("out", L)], dim=0)) for L in L4}
    norms = {f"{k}|{L}": float(v.norm()) for (k, L), v in dirs.items()}
    log(f"[probes] seed 4 directions: cos(own, out) {cos}; norms {norms} ({time.time() - t0:.0f}s)")
    # steering on odd samples
    plan = {"HON_DEMO_A_turns": +1, "HON_REC_A": -1, "none": +1, "REC_A": 0, "DEMO_A_turns": 0, "demo_turns_CH": +1,
            "own_turns_CH": 0}
    conds = []
    for p in prep:
        if int(p["sample"]) % 2 == 0 or p["cell"] not in plan:
            continue
        sgn = plan[p["cell"]]
        cs = [("base", None, 0.0)]
        if sgn:
            if p["family"] == "grey":
                cs += [(k, L, a) for k in ("own",) for L in L4 for a in ALPHAS]
                cs += [(k, 32, a) for k in ("out", "rand") for a in ALPHAS]
            else:
                cs += [(k, 32, a) for k in ("own", "rand") for a in ALPHAS]
        for c in cs:
            conds.append((p, c, sgn))
    log(f"[probes] seed 4: {len(conds)} steered readouts")
    with open(out / "seed4.jsonl", "w") as f:
        by_item = defaultdict(list)
        for p, c, sgn in conds:
            by_item[p["key"]].append((p, c, sgn))
        for n_done, (k, rows) in enumerate(by_item.items()):
            for b0 in range(0, len(rows), BATCH):
                ch = rows[b0:b0 + BATCH]
                p = ch[0][0]
                groups = defaultdict(list)          # one steering layer per forward
                for row in ch:
                    groups[row[1][1] or 32].append(row)
                for L, grp in groups.items():
                    vecs = torch.stack([(sgn * a * dirs[(kind, L)]) if kind != "base" else torch.zeros_like(dirs[("own", L)])
                                        for _, (kind, _, a), sgn in grp]).cuda()
                    M.steer = (L, vecs, p["start"])
                    logits = M.forward([p["ids"]] * len(grp))
                    M.steer = None
                    for (pp, (kind, LL, a), sgn), lg in zip(grp, logits):
                        sel = torch.tensor([opt_ids[o] for o in pp["opts"]])
                        pr = torch.softmax(lg[sel.cuda()], 0).cpu().tolist()
                        f.write(json.dumps({"key": pp["key"], "cell": pp["cell"], "family": pp["family"], "task": pp["task"],
                                            "sample": pp["sample"], "dir": kind, "layer": LL, "alpha": a * sgn,
                                            "p": dict(zip(pp["opts"], pr))}) + "\n")
            if n_done % 50 == 0:
                f.flush()
                log(f"[probes] seed 4 item {n_done + 1}/{len(by_item)} ({time.time() - t0:.0f}s)")
    (out / "seed4_meta.json").write_text(json.dumps({"cos_own_out": cos, "norms": norms}))


def capture(M, rows, layers, slice_fn, path, log, t0, tag):
    import numpy as np
    M.capture_layers = layers
    X = {L: [] for L in layers}
    meta = []
    for n, r in enumerate(rows):
        ids, sl = slice_fn(r)
        M.capture_slice = sl
        M.forward([ids])
        for L in layers:
            X[L].append(M.cap[L][0].numpy().astype(np.float16))
        meta.append({k: r[k] for k in r if k not in ("messages", "tools", "amap", "thinking")})
        if n % 200 == 0:
            log(f"[probes] {tag} {n + 1}/{len(rows)} ({time.time() - t0:.0f}s)")
    M.capture_layers = ()
    np.savez_compressed(path, **{f"L{L}": np.stack(v) for L, v in X.items()})
    (path.parent / (path.stem + "_meta.jsonl")).write_text("\n".join(json.dumps(m) for m in meta) + "\n")


def run(a):
    t0 = time.time()
    log = lambda m: print(m, flush=True)  # noqa: E731
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    M = Model()
    log(f"[probes] loaded ({time.time() - t0:.0f}s)")
    if "4" in a.seeds.split(","):
        run4(M, out, log, t0)
    if "7" in a.seeds.split(","):
        rows = seed7_items()

        def s7(r):
            ids = M.ids(M.render(r["messages"], r["tools"]))
            return ids, (len(ids) - 1, len(ids))
        capture(M, rows, L7, s7, out / "seed7.npz", log, t0, "seed 7")
    if "8b" in a.seeds.split(","):
        rows = seed8b_items()

        def s8b(r):
            p = M.render(r["messages"], r["tools"], think=True)
            n0 = len(M.ids(p))
            ids = M.ids(p + r["thinking"].rstrip("\n"))
            return ids, (n0, len(ids))
        capture(M, rows, L8, s8b, out / "seed8b.npz", log, t0, "seed 8b")
    if "ax" in a.seeds.split(","):
        rows = ax_items()

        def sax(r):
            if r["tools"] is None:
                s_ = M.tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            else:
                s_ = M.render(r["messages"], r["tools"])
            ids = M.ids(s_)
            return ids, (len(ids) - 1, len(ids))
        capture(M, rows, (24, 32, 40), sax, out / "seed_ax.npz", log, t0, "axis")
    if "ax2" in a.seeds.split(","):
        rows = ax2_items()

        def sax2(r):
            if r["tools"] is None:
                s_ = M.tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            else:
                s_ = M.render(r["messages"], r["tools"])
            ids = M.ids(s_)
            return ids, (len(ids) - 1, len(ids))
        capture(M, rows, (24, 32, 40), sax2, out / "seed_ax2.npz", log, t0, "axis2")
    if "8" in a.seeds.split(","):
        rows = seed8_items()

        def s8(r):
            p = M.render(r["messages"], r["tools"], think=True)
            n0 = len(M.ids(p))
            ids = M.ids(p + r["thinking"].rstrip("\n"))
            return ids, (n0, len(ids))
        capture(M, rows, L8, s8, out / "seed8.npz", log, t0, "seed 8")
    (out / "run_info.json").write_text(json.dumps({"seconds": round(time.time() - t0), "seeds": a.seeds}))
    log(f"[probes] done ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------------------------------- analysis (laptop)
def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def analyze4(a):
    R = [json.loads(l) for l in open(Path(a.out) / "seed4.jsonl")]
    meta = json.loads((Path(a.out) / "seed4_meta.json").read_text())
    tgt = lambda r: r["p"]["affirm"] if r["family"] == "grey" else r["p"]["penalise"]  # noqa: E731
    by = defaultdict(list)
    for r in R:
        by[(r["cell"], r["dir"], r["layer"], r["alpha"])].append(r)
    base = {c: 100 * sum(tgt(r) for r in v) / len(v) for (c, d, L, al), v in by.items() if d == "base"}
    res = {"meta": meta, "base_P_target": {k: round(v, 1) for k, v in base.items()}, "steered": {}}
    tasks = sorted({r["task"] for r in R})
    rng = random.Random(0)
    boots = [[rng.choice(tasks) for _ in tasks] for _ in range(5000)]
    for (c, d, L, al), v in sorted(by.items(), key=str):
        if d == "base":
            continue
        b = {(r["task"], r["sample"]): tgt(r) for r in by[(c, "base", None, 0.0)]}
        diffs = defaultdict(list)
        for r in v:
            diffs[r["task"]].append(tgt(r) - b[(r["task"], r["sample"])])
        f = lambda ts: 100 * sum(x for t in ts for x in diffs[t]) / max(1, sum(len(diffs[t]) for t in ts))  # noqa: E731
        bs = sorted(f(t) for t in boots)
        res["steered"][f"{c}|{d}|L{L}|a{al:+.1f}"] = [round(f(tasks), 1), round(bs[125], 1), round(bs[4874], 1)]
    (Path(a.out) / "analysis4.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def analyze7(a):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    d = np.load(Path(a.out) / "seed7.npz")
    meta = [json.loads(l) for l in open(Path(a.out) / "seed7_meta.jsonl")]
    sign = np.array([m["cell"].rsplit("_", 1)[1] for m in meta])
    task = np.array([m["task"] for m in meta])
    tasks = sorted(set(task))
    res = {}
    rng = np.random.default_rng(0)
    for L in L7:
        X = d[f"L{L}"].astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        r = {}
        folds = [tasks[i::4] for i in range(4)]       # task-grouped 4-fold

        def cv(mask_tr, mask_te_fn, y):
            sc, yy = [], []
            for fo in folds:
                te = np.isin(task, fo)
                tr = mask_tr & ~te
                tt = mask_te_fn & te
                if len(set(y[tr])) < 2 or len(set(y[tt])) < 2:
                    continue
                clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
                sc += list(clf.decision_function(X[tt]))
                yy += list(y[tt])
            return _auc(np.array(yy), np.array(sc)) if len(set(yy)) == 2 else None
        y = (sign != "clean").astype(int)
        allm = np.ones(len(y), bool)
        r["in_dist_all_signs"] = cv(allm, allm, y)
        for held in ("model", "future", "ids"):
            tr = np.isin(sign, [s for s in ("clean", "model", "future", "ids") if s != held])
            te = np.isin(sign, ["clean", held])
            r[f"held_out_{held}"] = cv(tr, te, y)
            null = []
            for _ in range(20):
                yp = y.copy()
                yp[tr] = rng.permutation(y[tr])
                null.append(cv(tr, te, yp) or 0.5)
            r[f"held_out_{held}_shuffled_p95"] = float(np.percentile(null, 95))
        res[f"L{L}"] = r
        print(f"L{L}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    (Path(a.out) / "analysis7.json").write_text(json.dumps(res, indent=1))


def analyze8(a):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    d = np.load(Path(a.out) / "seed8.npz")
    meta = [json.loads(l) for l in open(Path(a.out) / "seed8_meta.jsonl")]
    res_ = np.array([m["resolution"] for m in meta])
    cell = np.array([m["cell"] for m in meta])
    task = np.array([m["task"] for m in meta])
    out = {}
    for L in L8:
        X = d[f"L{L}"].astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        r = {}
        for setup, ov, fo in (("pooled", (res_ == "overrides_instruction") & np.isin(cell, ["CH_ownP", "rtd_ch_fl3"]),
                               (res_ == "follows_instruction")),
                              ("within_cell", (res_ == "overrides_instruction") & np.isin(cell, ["CH_ownP", "rtd_ch_fl3"]),
                               (res_ == "follows_instruction") & np.isin(cell, ["CH_ownP", "rtd_ch_fl3"]))):
            m = ov | fo
            y = ov[m].astype(int)
            tasks = sorted(set(task[m]))
            folds = [tasks[i::4] for i in range(4)]
            sc, yy = np.zeros(m.sum()), []
            idx = np.where(m)[0]
            for f_ in folds:
                te = np.isin(task[idx], f_)
                if len(set(y[~te])) < 2:
                    continue
                clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[idx][~te], y[~te])
                sc[te] = clf.decision_function(X[idx][te])
            auc = _auc(y, sc) if len(set(y)) == 2 else None
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[idx], y)
            s = clf.decision_function(X)
            mo, mf = s[ov].mean(), s[fo].mean()
            axis = lambda mask: float(((s[mask] - mf) / (mo - mf)).mean()) if mask.sum() else None  # noqa: E731
            r[setup] = {"n_override": int(ov.sum()), "n_follow": int(fo.sum()), "heldout_auc": auc,
                        "reinterprets_axis": axis((res_ == "reinterprets_instruction") & np.isin(cell, ["CH_ownP", "rtd_ch_fl3"])),
                        "ignores_axis": axis((res_ == "ignores_instruction") & np.isin(cell, ["CH_ownP", "rtd_ch_fl3"])),
                        "sysCH_follow_axis": axis((res_ == "follows_instruction") & (cell == "sysCH"))}
        out[f"L{L}"] = r
        print(f"L{L}", json.dumps(r), flush=True)
    (Path(a.out) / "analysis8.json").write_text(json.dumps(out, indent=1))


def analyze8b(a):
    import numpy as np
    d8, d8b = np.load(Path(a.out) / "seed8.npz"), np.load(Path(a.out) / "seed8b.npz")
    m8 = [json.loads(l) for l in open(Path(a.out) / "seed8_meta.jsonl")]
    m8b = [json.loads(l) for l in open(Path(a.out) / "seed8b_meta.jsonl")]
    cls = []
    for m in m8:
        if m["resolution"] == "follows_instruction" and m["cell"] == "sysCH":
            cls.append("A")
        elif m["resolution"] == "overrides_instruction" and m["cell"] in ("CH_ownP", "rtd_ch_fl3"):
            cls.append("B")
        elif m["resolution"] == "reinterprets_instruction" and m["cell"] in ("CH_ownP", "rtd_ch_fl3"):
            cls.append("reinterprets")
        elif m["resolution"] == "ignores_instruction" and m["cell"] in ("CH_ownP", "rtd_ch_fl3"):
            cls.append("ignores")
        else:
            cls.append(None)
    cls += [m["cls"] for m in m8b]
    tasks = np.array([m["task"] for m in m8] + [m["task"] for m in m8b])
    cls = np.array([c or "-" for c in cls])
    tlist = sorted(set(tasks))
    half = set(tlist[0::2])
    res = {"n": {c: int((cls == c).sum()) for c in ("A", "B", "C", "D", "reinterprets", "ignores")}}
    for L in L8:
        X = np.concatenate([d8[f"L{L}"], d8b[f"L{L}"]]).astype(np.float32)

        def fit(mask_t):
            mu = {c: X[(cls == c) & mask_t].mean(0) for c in "ABCD"}
            add = mu["A"] + mu["C"] - mu["D"]
            c_ = mu["B"] - add
            return add, c_, float(c_ @ c_)

        def score(add, c_, den, m):
            return float(((X[m] - add) @ c_).mean() / den) if m.sum() else None
        tr = np.isin(tasks, list(half))
        add, c_, den = fit(tr)
        te = ~tr
        gate = {k: score(add, c_, den, (cls == k) & te) for k in "ABCD"}
        add_all, c_all, den_all = fit(np.ones(len(cls), bool))
        r = {"heldout": gate, "gate_pass": bool(gate["B"] is not None and gate["B"] >= 0.5
                                                 and all(gate[k] is not None and abs(gate[k]) <= 0.3 for k in "ACD")),
             "reinterprets": score(add_all, c_all, den_all, cls == "reinterprets"),
             "ignores": score(add_all, c_all, den_all, cls == "ignores"),
             "reinterprets_heldout_dir": score(add, c_, den, (cls == "reinterprets") & te)}
        res[f"L{L}"] = r
        print(f"L{L}", json.dumps(r), flush=True)
    print(json.dumps(res["n"]))
    (Path(a.out) / "analysis8b.json").write_text(json.dumps(res, indent=1))


def analyze_ax(a):
    """Raw mean-difference axis (default - characters) and a logistic probe direction (default vs characters,
    balanced) from the axis items; both applied to every context (seed_ax + seed_ax2 if present). Scores are in SDs
    of the character spread (raw) / decision-function units standardised the same way (probe)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    D, M_ = [], []
    for name in ("seed_ax", "seed_ax2"):
        f = Path(a.out) / f"{name}.npz"
        if f.exists():
            D.append(np.load(f))
            M_ += [dict(json.loads(l), src=name) for l in open(Path(a.out) / f"{name}_meta.jsonl")]
    meta = M_
    res = {"n": dict(Counter(f"{m.get('family')}|{m.get('cell')}" for m in meta if m["kind"] == "ctx"))}
    contrasts = [("grey", "REC_A", "none"), ("grey", "INSTR_A", "none"), ("grey", "REC_A", "DEMO_A_turns"),
                 ("grey", "REC_A", "REC_C"), ("grey", "HON_REC_A", "HON_DEMO_A_turns"),
                 ("justice", "F5P", "none"), ("justice", "al_no_fl6", "none"), ("justice", "F5P", "F5R"),
                 ("justice", "CH_ownP", "sysCH"), ("justice", "CH_ownP", "F5P")]
    rng = random.Random(0)
    for L in (24, 32, 40):
        X = np.concatenate([d[f"L{L}"] for d in D]).astype(np.float32)
        isd = np.array([m.get("who") == "default" for m in meta])
        isp = np.array([m["kind"] == "axis" and m.get("who") not in (None, "default") for m in meta])
        axis = X[isd].mean(0) - X[isp].mean(0)
        axis /= np.linalg.norm(axis)
        raw = X @ axis
        raw = (raw - raw[isp].mean()) / raw[isp].std()
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(X[isd | isp], isd[isd | isp].astype(int))
        pr = clf.decision_function(X)
        pr = (pr - pr[isp].mean()) / pr[isp].std()
        cos = float(np.dot(axis, clf.coef_[0] / np.linalg.norm(clf.coef_[0])))
        r = {"cos_raw_probe": round(cos, 3), "default_raw": round(float(raw[isd].mean()), 2),
             "default_probe": round(float(pr[isd].mean()), 2)}
        cell = {}
        for m_i, m in enumerate(meta):
            if m["kind"] == "ctx":
                cell.setdefault((m["family"], m["cell"]), []).append(m_i)
        r["cells"] = {f"{f}|{c}": [round(float(raw[ix].mean()), 2), round(float(pr[ix].mean()), 2), len(ix)]
                      for (f, c), ix in sorted(cell.items())}
        for fam, x, y in contrasts:
            if (fam, x) not in cell or (fam, y) not in cell:
                continue
            ix, iy = cell[(fam, x)], cell[(fam, y)]
            tasks = sorted({meta[i]["task"] for i in ix + iy})
            byx, byy = defaultdict(list), defaultdict(list)
            for i in ix:
                byx[meta[i]["task"]].append(i)
            for i in iy:
                byy[meta[i]["task"]].append(i)
            out_c = {}
            for nm, v in (("raw", raw), ("probe", pr)):
                def diff(ts):
                    a_ = [j for t in ts for j in byx[t]]
                    b_ = [j for t in ts for j in byy[t]]
                    return float(v[a_].mean() - v[b_].mean()) if a_ and b_ else float("nan")
                bs = sorted(x_ for x_ in (diff([rng.choice(tasks) for _ in tasks]) for _ in range(2000)) if x_ == x_)
                out_c[nm] = [round(diff(tasks), 2), round(bs[int(0.025 * len(bs))], 2), round(bs[int(0.975 * len(bs)) - 1], 2)]
            r[f"{fam}: {x} - {y}"] = out_c
        res[f"L{L}"] = r
        print(f"L{L}", json.dumps(r), flush=True)
    (Path(a.out) / "analysis_ax.json").write_text(json.dumps(res, indent=1))


def build(a):
    s4, s7, s8 = seed4_items(), seed7_items(), seed8_items()
    print("seed 4", len(s4), Counter(it["cell"] for it in s4))
    print("seed 7", len(s7), Counter(it["cell"] for it in s7).most_common(3))
    print("seed 8", len(s8), Counter((it["cell"], it["resolution"]) for it in s8).most_common(12))
    from transformers import AutoTokenizer

    class Fake:
        tok = AutoTokenizer.from_pretrained(MODEL)
        render = Model.render
        ids = Model.ids
    F = Fake()
    for it in (s4[0], next(x for x in s4 if x["family"] == "justice")):
        st, full = target_start(F, it)
        n = len(F.ids(full))
        print(f"  {it['cell']}: prompt {n} tokens, target span {n - st} tokens; tail {full[-80:]!r}")
    if a.tok:
        n7 = max(len(F.ids(F.render(r["messages"], r["tools"]))) for r in s7[::40])
        n8 = max(len(F.ids(F.render(r["messages"], r["tools"], think=True) + r["thinking"])) for r in s8[::10])
        print(f"  longest seed 7 {n7}, seed 8 {n8} tokens")


def build2(a):
    b, x = seed8b_items(), ax_items()
    print("seed 8b", len(b), Counter((it["cell"], it["cls"]) for it in b))
    print("axis", len(x), Counter((it["kind"], it.get("family"), it.get("cell")) for it in x).most_common(12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "run", "analyze4", "analyze7", "analyze8", "analyze8b", "analyze_ax", "build2"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seeds", default="4,7,8")
    ap.add_argument("--tok", action="store_true")
    a = ap.parse_args()
    {"build": build, "run": run, "analyze4": analyze4, "analyze7": analyze7, "analyze8": analyze8, "analyze8b": analyze8b,
     "analyze_ax": analyze_ax, "build2": build2}[a.cmd](a)


if __name__ == "__main__":
    main()
