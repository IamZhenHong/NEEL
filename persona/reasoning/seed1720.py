"""Seeds 17 + 20 (DESIGN.md "Seeds 17 + 20") on neel-b3's Qwen3.5-9B adapters (tag c1e4-results; read-only):
merged arms P (placebo), C1 (IH-Challenge SFT), E4 ("precedent isn't permission" SFT).

  17  B8 (INTIMA) + B2 (SYCON) cells per arm; the base 9B's own records held identical across arms (bench's own keys,
      record sampling and seeds, so arms are paired).
  20  d_E4 = E4 - P and d_C1 = C1 - P (residual over the target-case tokens, layers 12/16/20, even samples of CH_ownP +
      F5P), steered into P on odd samples; P(penalise) from the option logits after the choose_option prefix.

  pod:    python -m persona.reasoning.seed1720 merge --adapters <dir> --merged /root/merged
          /root/vllm/bin/python -m persona.reasoning.seed1720 gen --arm P --merged /root/merged --out <dir>/s17
          python -m persona.reasoning.seed1720 s20 --merged /root/merged --out <dir>/s20
  laptop: uv run python -m persona.benchmarks.bench judge --model 9b --out <dir>/s17/<arm>   (each arm)
          uv run python -m persona.reasoning.seed1720 analyze --out <dir>
"""
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
BENCH = HERE.parent / "benchmarks"
BASE = "Qwen/Qwen3.5-9B"
ARMS = ("P", "C1", "E4")
B2_N = 60
LAYERS = (12, 16, 20)
PREFIX = "<tool_call>\n<function=choose_option>\n<parameter=case_id>\n{case}\n</parameter>\n<parameter=option_id>\n"


# ---------------------------------------------------------------------------------------------------- merge (pod)
def merge(a):
    from huggingface_hub import snapshot_download
    from persona.operative.lora_c1e4.train import merge_dirs
    base_dir = snapshot_download(BASE)
    for arm in ARMS:
        t0 = time.time()
        n = merge_dirs(base_dir, Path(a.adapters) / arm, Path(a.merged) / arm)
        print(f"[s1720] merged {arm}: {n} weights ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------------------------- seed 17
def s17_inputs():
    from persona.benchmarks import bench as BN
    rec = json.loads((BENCH / "results_9b" / "records.json").read_text())
    it8 = {x["i"]: x for x in BN.intima()}
    b2 = rec["b2"]
    b8 = {k: [(it8[i], r) for i, r in v] for k, v in rec["b8"]}
    t2, _ = BN.split(BN.sycon(), "9b", BN.N["qwen"]["b2_t"], BN.N["qwen"]["b2_s"], "b2")
    t8, _ = BN.split(BN.intima(), "9b", BN.N["qwen"]["b8_t"], BN.N["qwen"]["b8_s"], "b8")
    specs = BN.interleave(BN.b2_specs("9b", t2[:B2_N], b2, BN.B2_CELLS["qwen"]))
    its = BN.interleave(BN.b8_items("9b", t8, b8, BN.B8_CELLS["qwen"]))
    return specs, its


def gen(a):
    from persona.benchmarks import bench as BN
    t0 = time.time()
    out = Path(a.out) / a.arm
    out.mkdir(parents=True, exist_ok=True)
    g = BN.QwenGen(BASE if a.arm == "base" else str(Path(a.merged) / a.arm))     # base: seed 17b's same-pod anchor
    print(f"[s1720] {a.arm} loaded ({time.time() - t0:.0f}s)", flush=True)
    specs, its = s17_inputs()
    BN.run_dialogues(g, specs, log=lambda m: print(m, flush=True), tag=f"{a.arm} B2")
    with open(out / "b2.jsonl", "w") as f:
        for s in specs:
            f.write(json.dumps({k: v for k, v in s.items() if k != "base"} | {"think": False, "arm": a.arm},
                               ensure_ascii=False) + "\n")
    res = g([x["messages"] for x in its], [x["seed"] for x in its])
    with open(out / "b8.jsonl", "w") as f:
        for x, (vis, th) in zip(its, res):
            f.write(json.dumps({k: v for k, v in x.items() if k != "messages"} | {"reply": vis, "reasoning": th,
                                                                                  "think": False, "arm": a.arm},
                               ensure_ascii=False) + "\n")
    print(f"[s1720] {a.arm} done ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------------------------- seed 20
def s20_items():
    from persona.exemplar import tasks as T
    from persona.history import batch1 as B1
    hs = B1.histories(BASE)
    out = []
    for tid in sorted(T.TASKS):
        for i in range(6):
            earlier, texts = hs[(tid, i)]
            for cell in ("CH_ownP", "F5P", "sysCH", "none"):
                out.append(dict(cell=cell, task=tid, sample=i, key=f"S20|{cell}|{tid}|{i}",
                                messages=B1.cell_messages(cell, tid, earlier, texts, [], {}), tools=B1.tools()))
    return out


class M9:
    def __init__(self, path):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        n = getattr(AutoConfig.from_pretrained(path), "text_config", AutoConfig.from_pretrained(path)).num_hidden_layers
        try:
            self.m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
        except Exception:  # noqa: BLE001
            from transformers import AutoModelForImageTextToText
            self.m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
        self.m.eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.layers = next(mod for name, mod in self.m.named_modules()
                           if isinstance(mod, torch.nn.ModuleList) and len(mod) == n and name.endswith("layers"))
        self.steer, self.cap, self.cap_layers, self.cap_slice = None, {}, (), None
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
            if i in self.cap_layers:
                a_, b_ = self.cap_slice
                self.cap[i] = t[:, a_:b_, :].float().mean(1).cpu()
            return out
        return h

    def render(self, msgs, tools):
        return self.tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)

    def ids(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def logits(self, rows):
        x = self.torch.tensor(rows, device="cuda")
        with self.torch.no_grad():
            return self.m(input_ids=x, use_cache=False).logits[:, -1, :].float()


def _prep(M, it):
    full = M.render(it["messages"], it["tools"])
    req = it["messages"][-3]
    pos = full.rfind("<|im_start|>user\n" + req["content"][:60])
    assert pos > 0
    st = len(M.ids(full[:pos]))
    return st, len(M.ids(full)), M.ids(full + PREFIX.format(case=it["task"].upper()))


def s20(a):
    import gc
    import torch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    its = s20_items()
    t0 = time.time()
    acts, base_rows = {}, []
    for arm in ARMS:
        M = M9(str(Path(a.merged) / arm))
        opt = {o: M.ids(o)[0] for o in ("penalise", "refer", "note")}
        assert len(set(opt.values())) == 3, opt
        sel = torch.tensor(list(opt.values())).cuda()
        M.cap_layers = LAYERS
        sums = {L: [] for L in LAYERS}
        for it in its:
            st, n_p, ids = _prep(M, it)
            M.cap_slice = (st, n_p)
            lg = M.logits([ids])[0]
            pr = torch.softmax(lg[sel], 0).cpu().tolist()
            base_rows.append({"arm": arm, "dir": "none", "alpha": 0.0, "layer": None, "cell": it["cell"], "task": it["task"],
                              "sample": it["sample"], "p": dict(zip(opt, pr))})
            if it["sample"] % 2 == 0 and it["cell"] in ("CH_ownP", "F5P"):
                for L in LAYERS:
                    sums[L].append(M.cap[L][0])
        acts[arm] = {L: torch.stack(v).mean(0) for L, v in sums.items()}
        M.cap_layers = ()
        print(f"[s1720] s20 captured {arm} ({time.time() - t0:.0f}s)", flush=True)
        if arm != "P":
            del M
            gc.collect()
            torch.cuda.empty_cache()
        else:
            MP = M
    dirs = {}
    gen_ = torch.Generator().manual_seed(0)
    for L in LAYERS:
        dirs[("E4", L)] = acts["E4"][L] - acts["P"][L]
        dirs[("C1", L)] = acts["C1"][L] - acts["P"][L]
        r = torch.randn(dirs[("E4", L)].shape, generator=gen_)
        dirs[("rand", L)] = r / r.norm() * dirs[("E4", L)].norm()
    cos = {L: float(torch.nn.functional.cosine_similarity(dirs[("E4", L)], dirs[("C1", L)], dim=0)) for L in LAYERS}
    norms = {f"{k}|{L}": float(v.norm()) for (k, L), v in dirs.items()}
    print(f"[s1720] s20 cos(d_E4, d_C1) {cos}; norms {norms}", flush=True)
    M = MP
    opt = {o: M.ids(o)[0] for o in ("penalise", "refer", "note")}
    sel = torch.tensor(list(opt.values())).cuda()
    conds = [("E4", L, al) for L in LAYERS for al in (1.0, 2.0)] + [(k, 16, al) for k in ("C1", "rand") for al in (1.0, 2.0)]
    rows = []
    for n, it in enumerate([x for x in its if x["sample"] % 2 == 1]):
        st, n_p, ids = _prep(M, it)
        by_l = defaultdict(list)
        for c in conds:
            by_l[c[1]].append(c)
        for L, cs in by_l.items():
            vecs = torch.stack([al * dirs[(k, L)] for k, _, al in cs]).cuda()
            M.steer = (L, vecs, st)
            lg = M.logits([ids] * len(cs))
            M.steer = None
            for (k, LL, al), row in zip(cs, lg):
                pr = torch.softmax(row[sel], 0).cpu().tolist()
                rows.append({"arm": "P", "dir": k, "alpha": al, "layer": LL, "cell": it["cell"], "task": it["task"],
                             "sample": it["sample"], "p": dict(zip(opt, pr))})
        if n % 40 == 0:
            print(f"[s1720] s20 steer {n + 1}/{len(its) // 2} ({time.time() - t0:.0f}s)", flush=True)
    with open(out / "s20.jsonl", "w") as f:
        for r in base_rows + rows:
            f.write(json.dumps(r) + "\n")
    (out / "s20_meta.json").write_text(json.dumps({"cos_E4_C1": cos, "norms": norms}))
    print(f"[s1720] s20 done ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------------------------- analysis
def _boot(rows, fn, unit="task", n=5000):
    us = sorted({r[unit] for r in rows})
    rng = random.Random(0)
    by = defaultdict(list)
    for r in rows:
        by[r[unit]].append(r)
    v = sorted(x for x in (fn([y for u in [rng.choice(us) for _ in us] for y in by[u]]) for _ in range(n)) if x == x)
    if not v:                                     # no labels yet (judging incomplete) or an empty cell
        return [None, None, None]
    return [round(fn(rows), 1), round(v[int(0.025 * len(v))], 1), round(v[int(0.975 * len(v)) - 1], 1)]


def analyze(a):
    import subprocess
    import sys
    out = Path(a.out)
    res = {}
    # seed 17: bench's own analysis per arm (consensus and each judge), then cross-arm contrasts on B8 REC_R / INSTR_R
    # and B2 REC_C (paired items)
    lab8, lab2 = {}, {}
    for arm in ARMS:
        d = out / "s17" / arm
        if not (d / "judged.jsonl").exists():
            continue
        J = defaultdict(dict)
        for r in map(json.loads, open(d / "judged.jsonl")):
            J[(r["file"], r["key"], r["turn"])] = list(r["labels"].values())
        for r in map(json.loads, open(d / "b8.jsonl")):
            v = J.get(("b8", r["key"], 0))
            lab8[(arm, r["cell"], r["item"])] = (v[0] if v and v[0] is not None and v[0] == v[1] else None, v)
        for r in map(json.loads, open(d / "b2.jsonl")):
            v = J.get(("b2", r["key"], 4))
            lab2[(arm, r["cell"], r["item"])] = (v[0] if v and v[0] is not None and v[0] == v[1] else None, v)
    if lab8:
        cells8 = sorted({c for (_, c, _) in lab8})
        reinf = lambda arm, c, rows, j=None: _pct([None if (x := (lab8[(arm, c, r)][0] if j is None else (lab8[(arm, c, r)][1] or [None, None])[j])) is None  # noqa: E731
                                                   else int(x == "reinforcing") for r in rows if (arm, c, r) in lab8])
        items8 = sorted({i for (_, _, i) in lab8})
        res["s17_b8_reinforcing"] = {f"{arm}|{c}": round(reinf(arm, c, items8), 1) for arm in ARMS for c in cells8}
        rows = [{"task": i} for i in items8]
        for x in ("C1", "E4"):
            for c in ("REC_R", "INSTR_R", "none", "REC_B"):
                res[f"s17 B8 {c}: {x} - P"] = _boot(rows, lambda rr, x=x, c=c: reinf(x, c, [r["task"] for r in rr]) - reinf("P", c, [r["task"] for r in rr]))
        res["s17 B8 E4: REC_R - none"] = _boot(rows, lambda rr: reinf("E4", "REC_R", [r["task"] for r in rr]) - reinf("E4", "none", [r["task"] for r in rr]))
    if lab2:
        cells2 = sorted({c for (_, c, _) in lab2})
        items2 = sorted({i for (_, _, i) in lab2})

        def acc(arm, c, its_, j=None):
            out_ = []
            for i in its_:
                if (arm, c, i) not in lab2:
                    continue
                x = lab2[(arm, c, i)][0] if j is None else (lab2[(arm, c, i)][1] or [None, None])[j]
                out_.append(None if x is None else int(x == 0))
            return _pct(out_)
        for nm, j in (("consensus", None), ("haiku", 0), ("flash", 1)):
            res[f"s17_b2_accept_final_{nm}"] = {f"{arm}|{c}": round(acc(arm, c, items2, j), 1) for arm in ARMS for c in cells2}
            rows = [{"task": i} for i in items2]
            for x in ("C1", "E4"):
                res[f"s17 B2 REC_C ({nm}): {x} - P"] = _boot(rows, lambda rr, x=x, j=j: acc(x, "REC_C", [r["task"] for r in rr], j) - acc("P", "REC_C", [r["task"] for r in rr], j))
    # seed 20
    p = out / "s20" / "s20.jsonl"
    if p.exists():
        R = [json.loads(l) for l in open(p)]
        meta = json.loads((out / "s20" / "s20_meta.json").read_text())
        pen = lambda rows: 100 * sum(r["p"]["penalise"] for r in rows) / len(rows) if rows else float("nan")  # noqa: E731
        res["s20_meta"] = meta
        res["s20_unsteered"] = {f"{arm}|{c}": round(pen([r for r in R if r["arm"] == arm and r["dir"] == "none" and r["cell"] == c]), 1)
                                for arm in ARMS for c in ("CH_ownP", "F5P", "sysCH", "none")}
        base = {(r["cell"], r["task"], r["sample"]): r["p"]["penalise"] for r in R if r["arm"] == "P" and r["dir"] == "none"}
        steered = defaultdict(list)
        for r in R:
            if r["dir"] != "none":
                steered[(r["dir"], r["layer"], r["alpha"], r["cell"])].append(
                    {"task": r["task"], "d": r["p"]["penalise"] - base[(r["cell"], r["task"], r["sample"])]})
        res["s20_steered"] = {f"{k[0]}|L{k[1]}|a{k[2]}|{k[3]}": _boot(v, lambda rr: 100 * sum(x["d"] for x in rr) / len(rr))
                              for k, v in sorted(steered.items(), key=str)}
    (out / "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


def _pct(v):
    v = [x for x in v if x is not None]
    return 100 * sum(v) / len(v) if v else float("nan")


def explore(a):
    """Not pre-registered (labelled exploratory in RESULTS_s1720.md): every cell as arm - P, the record's pull within
    each arm, the base 9B (bench's own run, same keys and seeds) on the same items, judge agreement."""
    out = Path(a.out)
    srcs = {arm: out / "s17" / arm for arm in ARMS} | {"base": BENCH / "results_9b"}
    lab, labj, kap = {}, {}, {}
    for arm, d in srcs.items():
        J = {(r["file"], r["key"], r["turn"]): list(r["labels"].values()) for r in map(json.loads, open(d / "judged.jsonl"))}
        for bench, turn, pos in (("b8", 0, "reinforcing"), ("b2", 4, 0)):
            pairs = []
            for r in map(json.loads, open(d / f"{bench}.jsonl")):
                v = J.get((bench, r["key"], turn))
                if v is None:
                    continue
                pairs.append(v)
                lab[(arm, bench, r["cell"], r["item"])] = None if v[0] is None or v[0] != v[1] else int(v[0] == pos)
                for j in (0, 1):
                    labj[(arm, bench, r["cell"], r["item"], j)] = None if v[j] is None else int(v[j] == pos)
            n = len([p for p in pairs if None not in p])
            po = sum(p[0] == p[1] for p in pairs if None not in p) / n
            cats = {x for p in pairs for x in p if x is not None}
            pe = sum((sum(p[0] == c for p in pairs if None not in p) / n) * (sum(p[1] == c for p in pairs if None not in p) / n) for c in cats)
            kap[f"{arm}|{bench}"] = {"kappa": round((po - pe) / (1 - pe), 2), "n": n,
                                     "unclear": sum(lab[k] is None for k in lab if k[:2] == (arm, bench))}
    res = {"kappa": kap}
    for bench in ("b8", "b2"):
        items = sorted({k[3] for k in lab if k[0] == "P" and k[1] == bench})
        cells = sorted({k[2] for k in lab if k[0] == "P" and k[1] == bench})
        rate = lambda arm, c, its_: _pct([lab.get((arm, bench, c, i)) for i in its_])  # noqa: E731
        res[f"{bench}_rates"] = {f"{arm}|{c}": round(rate(arm, c, items), 1) for arm in ("base",) + ARMS for c in cells}
        rows = [{"task": i} for i in items]
        rec = {"b8": ("REC_R", "REC_B"), "b2": ("REC_C", "REC_H")}[bench]
        for c in cells:
            for x in ("C1", "E4"):
                res[f"{bench} {c}: {x} - P"] = _boot(rows, lambda rr, x=x, c=c: rate(x, c, [r["task"] for r in rr]) - rate("P", c, [r["task"] for r in rr]))
            res[f"{bench} {c}: P - base"] = _boot(rows, lambda rr, c=c: rate("P", c, [r["task"] for r in rr]) - rate("base", c, [r["task"] for r in rr]))
        for arm in ("base",) + ARMS:
            for c in rec:
                res[f"{bench} {arm}: {c} - none"] = _boot(rows, lambda rr, arm=arm, c=c: rate(arm, c, [r["task"] for r in rr]) - rate(arm, "none", [r["task"] for r in rr]))
            res[f"{bench} {arm}: {rec[0]}_RULE - RULE"] = _boot(rows, lambda rr, arm=arm: rate(arm, f"{rec[0]}_RULE", [r["task"] for r in rr]) - rate(arm, "RULE", [r["task"] for r in rr]))
        for x in ("C1", "E4"):                    # does training change the record's pull (difference in differences)?
            res[f"{bench} ({rec[0]} - none): {x} - P"] = _boot(rows, lambda rr, x=x: (rate(x, rec[0], [r["task"] for r in rr]) - rate(x, "none", [r["task"] for r in rr]))
                                                                 - (rate("P", rec[0], [r["task"] for r in rr]) - rate("P", "none", [r["task"] for r in rr])))
        for j, nm in ((0, "haiku"), (1, "flash")):  # each judge alone (B2's kappa fails; claims need both judges)
            rj = lambda arm, c, its_, j=j: _pct([labj.get((arm, bench, c, i, j)) for i in its_])  # noqa: E731
            res[f"{bench}_rates_{nm}"] = {f"{arm}|{c}": round(rj(arm, c, items), 1) for arm in ("base",) + ARMS for c in cells}
            for c in cells:
                res[f"{bench} {c} ({nm}): E4 - P"] = _boot(rows, lambda rr, c=c, rj=rj: rj("E4", c, [r["task"] for r in rr]) - rj("P", c, [r["task"] for r in rr]))
                res[f"{bench} {c} ({nm}): P - base"] = _boot(rows, lambda rr, c=c, rj=rj: rj("P", c, [r["task"] for r in rr]) - rj("base", c, [r["task"] for r in rr]))
            for c in rec:
                res[f"{bench} E4 ({nm}): {c} - none"] = _boot(rows, lambda rr, c=c, rj=rj: rj("E4", c, [r["task"] for r in rr]) - rj("E4", "none", [r["task"] for r in rr]))
    (out / "explore.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def build(a):
    specs, its = s17_inputs()
    print("s17 per arm: B2", len(specs), "dialogues", Counter(s["cell"] for s in specs), "| B8", len(its))
    s = s20_items()
    print("s20", len(s), Counter(x["cell"] for x in s))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "merge", "gen", "s20", "analyze", "explore"))
    ap.add_argument("--adapters", default="persona/operative/lora_c1e4/results/pod/adapters")
    ap.add_argument("--merged", default="/root/merged")
    ap.add_argument("--arm", choices=ARMS + ("Pself", "base"))
    ap.add_argument("--out", default=str(HERE / "results_s1720"))
    a = ap.parse_args()
    {"build": build, "merge": merge, "gen": gen, "s20": s20, "analyze": analyze, "explore": explore}[a.cmd](a)
