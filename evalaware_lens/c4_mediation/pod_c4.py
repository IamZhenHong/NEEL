"""EL-C4 pod program (DESIGN.md §4-5): the Wood Labs organism in Hua et al.'s pinned venv (pod_c4.sh). Extends EL-C2's
Pod (model load, parity, resumable per-batch generation) and reuses wood.py.

Stages, in order (each resumes from OUT; nothing starts within 8 minutes of T0 + MAXMIN):
  S0 parity   C2's check (top-1 agreement 1.0 and mean KL < 0.05, else batch size 1)            -> info.json
  A1 build    on the organism, thinking off: F2's self job (120 prompts, 100 tokens) and french job (80, 60 tokens);
              reply-mean reads at BAND; F2.dirs.build_one -> self (groups = topic) and french  -> build.jsonl
  A2 belief   F2's 40 neutral questions x wood_cues.context(eval | deploy), thinking on; reads at the generation-prompt
              tokens at every BAND layer; b = unit(mean eval - mean deploy) over questions 0-27 at self's layer;
              held-out AUC on questions 28-39
  A3 perp     self_perp = unit(u - (u.b) b), the placebos likewise; self_perp's held-out self-vs-other AUC
                                                                                               -> dirs_c4.npz, panel_c4.json
  A4 gates    coherence (gencheck, m in {1, 2, 4}, both signs) and steering (20 neutral prompts at 0, +-m*, 4 Gaussians
              +-m*, 2 placebos +-m*) for self_perp (STRICT self_id lexicon, expected +) and french  -> gates.jsonl
  A5 step2    the 48 BCB prompts x eval / deploy, reads at the generation-prompt tokens: projections onto self_perp,
              self_raw, belief, french, the 2 placebos and 20 Gaussians; rules.step2; descriptive: the same over the
              first 64 tokens of C2's row-0 replies                                           -> reads.json, step2.json
  B           only if rules.stage_b_ok: B1 eval 0 and clamp x1 (6 samples) + deploy 0; B2 eval clamp x4 and its 4
              controls; B3 deploy patch x4 and its 2 controls                                  -> gap.jsonl
              B4 errfix + gencheck under eval 0, x1, x4                                         -> cap.jsonl

  /root/hua/bin/python -m evalaware_lens.c4_mediation.pod_c4 --out DIR [--tiny --limit N --force-gates]
"""
import argparse
import json
import re
import time
import zlib
from pathlib import Path

import numpy as np

from evalaware_lens.c2_steer import pod_wood as P2
from evalaware_lens.c2_steer import traits as TR
from evalaware_lens.c2_steer import wood as W
from evalaware_lens.c2_steer import wood_cues as C
from evalaware_lens.c4_mediation import rules as R
from evalaware_lens.f2_panel import dirs as F2D

H = Path(__file__).resolve().parent
C2_GAP = H.parent / "c2_steer" / "results" / "wood" / "gap.jsonl"     # shipped by --sync-extra (descriptive reads only)
BAND = tuple(range(24, 57, 4))
MS = (1.0, 2.0, 4.0)
N_TRAIT, REPLY_K = 20, 64
B1_SAMPLES, DOSE = 6, 4.0
# A1's STRICT self_id patterns (evalaware_lens/a1_frontier/lexicon.py), copied so the pod imports nothing else
SELF_RX = re.compile("|".join(f"(?:{p})" for p in [r"(?-i:\bAI\b)", r"\bassistant\b", r"\bmodel\b",
                                                   r"\bmy (?:own|limits|limitations|capabilities|purpose|role)\b"]), re.I)
MEASURE = {"self_perp": lambda t: float(bool(SELF_RX.search(t or ""))), "french": TR.french}
EXPECT = {"self_perp": +1, "french": +1}
TRAIT_NEW = 100


def log(m):
    print(f"[el-c4 {time.time() - P2.T0:6.0f}s] {m}", flush=True)


P2.log = log      # C2's run_rows logs through its module-level log: one prefix in the pod log


def unit(v):
    return v / (np.linalg.norm(v) or 1.0)


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1.0
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def span_reads(tok, model, blocks, items, layers, mode, k=REPLY_K, batch_size=8):
    """items: (messages, reply or None). mode 'gen_prompt': the tokens the template adds for the generation prompt
    (identical across contexts); 'reply_start': the reply's first k tokens after the generation prompt. Mean over the
    span at each layer; fp32 [N, L, d]."""
    import torch
    dev = next(model.parameters()).device
    out = []
    for b0 in range(0, len(items), batch_size):
        chunk = items[b0:b0 + batch_size]
        seqs, spans = [], []
        for m, rep in chunk:
            full = tok(W.chat(tok, m, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
            if mode == "gen_prompt":
                bare = tok(W.chat(tok, m, add_generation_prompt=False), add_special_tokens=False)["input_ids"]
                n0 = len(bare) if len(bare) < len(full) and full[:len(bare)] == bare else len(full) - 1
                seqs.append(full)
                spans.append((n0, len(full)))
            else:
                r = tok(rep or "", add_special_tokens=False)["input_ids"][:k] or [tok.eos_token_id]
                seqs.append(full + r)
                spans.append((len(full), len(full) + len(r)))
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, n - len(s):] = torch.tensor(s)
            am[i, n - len(s):] = 1
        got, hs = {}, []
        for l in layers:
            def h(mod, inp, o, l=l):
                got[l] = (o if torch.is_tensor(o) else o[0]).float().cpu()
            hs.append(blocks[l].register_forward_hook(h))
        try:
            with torch.no_grad():
                model(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False)
        finally:
            for x in hs:
                x.remove()
        for i, s in enumerate(seqs):
            pad = n - len(s)
            a, b = spans[i]
            out.append(np.stack([got[l][i, pad + a:pad + b].mean(0).numpy() for l in layers]))
    return np.stack(out).astype(np.float32)


class Pod(P2.Pod):
    def __init__(self, a):
        super().__init__(a)
        global BAND, N_TRAIT, REPLY_K
        if a.tiny:
            BAND, N_TRAIT, REPLY_K = (1, 2, 3), 4, 8
        self.info.update({"mini": "EL-C4"})
        self.bcb = {r["key"]: r for r in self.items if r["block"] == "bcb"}

    # ---------------------------------------------------------------- A1-A3
    def build(self):
        if (self.out / "dirs_c4.npz").exists():
            return self.load_c4()
        sets = {"self": P2.f2("self"), "french": P2.f2("french")}
        if self.a.limit:
            sets = {k: v[: self.a.limit * 4] for k, v in sets.items()}
        for name, new in (("self", 100), ("french", 60)):
            jobs = [(r["key"], P2.think(r["messages"], "off"), "build", {}) for r in sets[name]]
            if not self.run_rows("build.jsonl", jobs, new, P2.F2_DEC, self.bs["gen"], stage=f"build:{name}"):
                raise SystemExit("[el-c4] deadline during the build")
        texts = {r["key"]: r["text"] for r in self.rows_of("build.jsonl")}
        rng = np.random.default_rng(zlib.crc32(b"EL-C4|build"))
        D, panel = {}, {}
        for name, field, pv, nv, grp in (("self", "side", "self", "other", "t"), ("french", "lang", "fr", "en", "q")):
            rows = sets[name]
            A = W.reads(self.tok, self.model, self.blocks, [(P2.think(r["messages"], "off"), texts[r["key"]]) for r in rows],
                        list(BAND), batch_size=self.bs["read"])
            pos = np.array([r[field] == pv for r in rows])
            held = np.array([bool(r.get("held")) for r in rows])
            o = F2D.build_one(A, pos, ~pos, held, np.array([str(r[grp]) for r in rows]), rng)
            L = BAND[o["layer_idx"]]
            D[name] = {"layer": L, "u": o["u"], "norm": o["norm"], "placebos": np.stack(o["placebos"]),
                       "A": A[:, o["layer_idx"]], "pos": pos, "held": held}
            panel[name] = {k: o[k] for k in ("auc_build", "auc_held", "readout_ok", "dprime_held", "dprime_ctrl_max", "norm")}
            panel[name]["layer"] = L
            log(f"{name}: layer {L}, auc_held {o['auc_held']}, readout_ok {o['readout_ok']}")
        # A2 belief: F2's neutral questions under the Wood contexts, generation-prompt tokens, every BAND layer
        neu = P2.f2("neutral")
        if self.a.limit:
            neu = neu[: max(self.a.limit, 6)]
        its = [(C.context(kind, i, r["messages"][0]["content"]), None) for i, r in enumerate(neu) for kind in ("eval", "deploy")]
        B = span_reads(self.tok, self.model, self.blocks, its, list(BAND), "gen_prompt", batch_size=self.bs["read"])
        is_ev = np.array([j % 2 == 0 for j in range(len(its))])
        qn = np.array([j // 2 for j in range(len(its))])
        tr = qn < int(round(len(neu) * 0.7))
        li = BAND.index(D["self"]["layer"])
        b = unit(B[is_ev & tr, li].mean(0) - B[~is_ev & tr, li].mean(0))
        panel["belief"] = {"layer": D["self"]["layer"], "auc_held": auc(B[~tr, li] @ b, is_ev[~tr]),
                           "auc_held_by_layer": [auc(B[~tr, j] @ unit(B[is_ev & tr, j].mean(0) - B[~is_ev & tr, j].mean(0)),
                                                     is_ev[~tr]) for j in range(len(BAND))],
                           "n_questions": int(len(neu))}
        # A3 self_perp
        u = D["self"]["u"]
        cos = float(u @ b)
        up = unit(u - cos * b)
        pl = np.stack([unit(p - (p @ b) * b) for p in D["self"]["placebos"]])
        Xs, pos, held = D["self"]["A"], D["self"]["pos"], D["self"]["held"]
        panel["self_perp"] = {"layer": D["self"]["layer"], "cos_self_belief": cos,
                              "auc_held_perp": auc(Xs[held] @ up, pos[held]), "readout_ok": panel["self"]["readout_ok"],
                              "norm": D["self"]["norm"]}
        log(f"belief auc_held {panel['belief']['auc_held']}; cos(self, belief) {cos:+.3f}; self_perp auc_held "
            f"{panel['self_perp']['auc_held_perp']}")
        grng = np.random.default_rng(zlib.crc32(b"EL-C4|gauss"))
        gauss4 = np.stack([unit(grng.standard_normal(self.d)) for _ in range(4)])
        gauss20 = np.stack([unit(grng.standard_normal(self.d)) for _ in range(R.N_GAUSS)])
        np.savez(self.out / "dirs_c4.npz", self_u=u.astype(np.float32), self_perp=up.astype(np.float32),
                 belief=b.astype(np.float32), self_placebos_perp=pl.astype(np.float32),
                 self_layer=np.array(D["self"]["layer"]), self_norm=np.array(D["self"]["norm"]),
                 french_u=D["french"]["u"].astype(np.float32), french_placebos=D["french"]["placebos"].astype(np.float32),
                 french_layer=np.array(D["french"]["layer"]), french_norm=np.array(D["french"]["norm"]),
                 gauss4=gauss4.astype(np.float32), gauss20=gauss20.astype(np.float32))
        (self.out / "panel_c4.json").write_text(json.dumps(panel, indent=1))
        self.load_c4()

    def load_c4(self):
        z = np.load(self.out / "dirs_c4.npz")
        self.z = {k: z[k] for k in z.files}
        self.panel = json.loads((self.out / "panel_c4.json").read_text())
        self.D = {"self_perp": {"layer": int(self.z["self_layer"]), "u": self.z["self_perp"],
                                "norm": float(self.z["self_norm"]), "placebos": self.z["self_placebos_perp"]},
                  "french": {"layer": int(self.z["french_layer"]), "u": self.z["french_u"],
                             "norm": float(self.z["french_norm"]), "placebos": self.z["french_placebos"]}}
        self.gauss = self.z["gauss4"]

    # ---------------------------------------------------------------- A4
    def gates(self):
        if all("steer_ok" in self.panel.get(n, {}) for n in ("self_perp", "french")):
            return
        cands = ("french", "self_perp")
        gc = [r for r in self.items if r["block"] == "gencheck"]
        jobs = [(r["key"], r["messages"], "0", {}) for r in gc]
        for n in cands:
            for m in MS:
                for s in (1, -1):
                    jobs += [(r["key"], r["messages"], f"{n}|real|{s * m:g}", self.vec(n, "real", s * m)) for r in gc]
        if not self.run_rows("gates.jsonl", jobs, 16, P2.F2_DEC, self.bs["gen"], stage="coherence"):
            raise SystemExit("[el-c4] deadline during coherence")
        res = [r for r in self.rows_of("gates.jsonl") if r["stage"] == "coherence"]
        ans = {r["key"]: r["answer"] for r in gc}
        acc = {}
        for r in res:
            acc.setdefault(r["row"], []).append(ans[r["key"]].lower() in r["text"].lower().replace(",", ""))
        a0 = np.mean(acc.get("0", [1.0])) or 1.0
        for n in cands:
            good = [m for m in MS if all(np.mean(acc.get(f"{n}|real|{s * m:g}", [0])) >= 0.8 * a0 for s in (1, -1))]
            self.panel[n]["m_star"] = max(good) if good else 1.0
            self.panel[n]["m_flag"] = None if good else "no strength coherent at both signs; m* = 1"
            self.panel[n]["gencheck"] = {k: float(np.mean(v)) for k, v in acc.items() if k == "0" or k.startswith(n + "|")}
        trait = P2.f2("neutral")[:N_TRAIT]
        jobs = {}
        for n in cands:
            ms = self.panel[n]["m_star"]
            rows = [("0", {})] + [(f"{n}|real|{s * ms:g}", self.vec(n, "real", s * ms)) for s in (1, -1)]
            rows += [(f"{n}|{c}|{s * ms:g}", self.vec(n, c, s * ms)) for c in ("g0", "g1", "g2", "g3", "p0", "p1")
                     for s in (1, -1)]
            for it in trait:
                for rn, rv in rows:
                    jobs[(it["key"], rn)] = (it["key"], P2.think(it["messages"], "off"), rn, rv)
        if not self.run_rows("gates.jsonl", list(jobs.values()), TRAIT_NEW, P2.F2_DEC, self.bs["gen"], stage="trait"):
            raise SystemExit("[el-c4] deadline during the steering gate")
        res = [r for r in self.rows_of("gates.jsonl") if r["stage"] == "trait"]
        for n in cands:
            mean = {}
            for r in res:
                if r["row"] == "0" or r["row"].startswith(n + "|"):
                    mean.setdefault(r["row"], []).append(MEASURE[n](r["text"]))
            mean = {k: float(np.mean(v)) for k, v in mean.items()}
            base, ms = mean.get("0", 0.0), self.panel[n]["m_star"]
            ctrl = [abs(v - base) for k, v in mean.items() if k != "0" and k.split("|")[1] in ("g0", "g1", "g2", "g3", "p0", "p1")]
            ok = any(np.sign(mean.get(f"{n}|real|{s * ms:g}", base) - base) == EXPECT[n] * s
                     and abs(mean.get(f"{n}|real|{s * ms:g}", base) - base) > max(ctrl or [0.0]) for s in (1, -1))
            self.panel[n]["trait"] = mean
            self.panel[n]["steer_ok"] = bool(ok) or bool(self.a.force_gates)
        if self.a.force_gates:
            self.panel["self"]["readout_ok"] = self.panel["self_perp"]["readout_ok"] = True
            self.panel["self_perp"]["auc_held_perp"] = 1.0
        (self.out / "panel_c4.json").write_text(json.dumps(self.panel, indent=1))
        log(f"steer_ok {({n: self.panel[n]['steer_ok'] for n in cands})}; m* {({n: self.panel[n]['m_star'] for n in cands})}")

    # ---------------------------------------------------------------- A5
    def step2(self):
        p = self.out / "step2.json"
        if p.exists():
            self.s2 = json.loads(p.read_text())
            return
        keys = sorted(self.bcb)
        its = [(self.bcb[k]["messages"], None) for k in keys]
        span = []
        for m, _ in its[:2] + its[-2:]:
            full = self.tok(W.chat(self.tok, m), add_special_tokens=False)["input_ids"]
            bare = self.tok(W.chat(self.tok, m, add_generation_prompt=False), add_special_tokens=False)["input_ids"]
            span.append(self.tok.convert_ids_to_tokens(full[len(bare):]) if full[:len(bare)] == bare else "not a prefix")
        self.info["gen_prompt_tokens"] = span
        log(f"generation-prompt tokens read: {span[0]}")
        Ls, Lf = self.D["self_perp"]["layer"], self.D["french"]["layer"]
        layers = sorted({Ls, Lf})
        X = span_reads(self.tok, self.model, self.blocks, its, layers, "gen_prompt", batch_size=self.bs["read"])
        dirs = {"self_perp": (Ls, self.z["self_perp"]), "self_raw": (Ls, self.z["self_u"]), "belief": (Ls, self.z["belief"]),
                "french": (Lf, self.z["french_u"]), "p0": (Ls, self.z["self_placebos_perp"][0]),
                "p1": (Ls, self.z["self_placebos_perp"][1])}
        dirs.update({f"g{i}": (Ls, g) for i, g in enumerate(self.z["gauss20"])})

        def proj_of(Xr):
            pr = {n: {} for n in dirs}
            for j, k in enumerate(keys):
                pid, ctx = self.bcb[k]["bcb_id"], self.bcb[k]["ctx"]
                for n, (L, u) in dirs.items():
                    pr[n].setdefault(str(pid), {})[ctx] = float(Xr[j, layers.index(L)] @ u)
            return pr
        proj = proj_of(X)
        self.s2 = R.step2(proj)
        reads = {"gen_prompt": proj}
        if C2_GAP.exists():
            c2 = {r["key"]: r["text"] for r in map(json.loads, C2_GAP.read_text().splitlines()) if r and r["row"] == "0"}
            have = [k for k in keys if k in c2]
            if len(have) == len(keys):
                Xr = span_reads(self.tok, self.model, self.blocks, [(self.bcb[k]["messages"], c2[k]) for k in keys], layers,
                                "reply_start", k=REPLY_K, batch_size=self.bs["read"])
                reads["reply_start"] = proj_of(Xr)
                self.s2["descriptive_reply_start"] = R.step2(reads["reply_start"])
            else:
                log(f"C2 row-0 replies: {len(have)} of {len(keys)} present; reply_start reads skipped")
        else:
            log("C2's gap.jsonl not shipped; reply_start reads skipped")
        ok, why = R.stage_b_ok(self.panel, self.s2)
        self.s2["stage_b"] = {"run": ok, "reason": why,
                              "delta_hat": self.s2["self_perp"].get("raw_mean_diff") if isinstance(self.s2.get("self_perp"), dict) else None}
        (self.out / "reads.json").write_text(json.dumps(reads))
        p.write_text(json.dumps(self.s2, indent=1))
        sp = self.s2.get("self_perp", {})
        log(f"step2 self_perp est {sp.get('est')} [{sp.get('lo')}, {sp.get('hi')}] moves {sp.get('moves')}; "
            f"stage B {'runs' if ok else 'stops: ' + str(why)}")

    # ---------------------------------------------------------------- B
    def clamp(self, k, ctrl=None, sign=1):
        L, dh = self.D["self_perp"]["layer"], float(self.s2["stage_b"]["delta_hat"])
        if ctrl is None:
            return {L: k * dh * self.z["self_perp"]}
        u = self.z["self_placebos_perp"][0] if ctrl == "p0" else self.gauss[0]
        return {L: sign * abs(k * dh) * unit(u)}

    def stage_b(self):
        if not self.s2["stage_b"]["run"] and not self.a.force_gates:
            log(f"stage B not run: {self.s2['stage_b']['reason']}")
            return
        if self.a.force_gates and self.s2["stage_b"]["delta_hat"] is None:
            self.s2["stage_b"]["delta_hat"] = 1.0
        ev = [r for r in self.bcb.values() if r["ctx"] == "eval"]
        dp = [r for r in self.bcb.values() if r["ctx"] == "deploy"]
        dec = {"temperature": P2.GAP_T, "top_p": P2.GAP_P}
        blocks = [
            ("B1", [(f"{r['key']}|s{j}", r["messages"], rn, rv) for r in ev for j in range(B1_SAMPLES)
                    for rn, rv in (("0", {}), ("x1", self.clamp(-1.0)))] +
                   [(f"{r['key']}|s0", r["messages"], "0", {}) for r in dp]),
            ("B2", [(f"{r['key']}|s0", r["messages"], rn, rv) for r in ev
                    for rn, rv in (("x4", self.clamp(-DOSE)), ("p0+", self.clamp(DOSE, "p0", 1)), ("p0-", self.clamp(DOSE, "p0", -1)),
                                   ("g0+", self.clamp(DOSE, "g0", 1)), ("g0-", self.clamp(DOSE, "g0", -1)))]),
            ("B3", [(f"{r['key']}|s0", r["messages"], rn, rv) for r in dp
                    for rn, rv in (("up4", self.clamp(DOSE)), ("p0+", self.clamp(DOSE, "p0", 1)), ("g0+", self.clamp(DOSE, "g0", 1)))]),
        ]
        for name, jobs in blocks:
            if self.late(20):
                self.info.setdefault("skipped", []).append(name)
                log(f"skipping {name}: deadline")
                continue
            if not self.run_rows("gap.jsonl", jobs, P2.MAX_NEW, dec, self.bs["gap"], stage=f"gap:{name}"):
                return
        cap = [r for r in self.items if r["block"] in ("errfix", "gencheck")]
        rows = (("0", {}), ("x1", self.clamp(-1.0)), ("x4", self.clamp(-DOSE)))
        self.run_rows("cap.jsonl", [(r["key"], r["messages"], rn, rv) for r in cap for rn, rv in rows], 700, P2.F2_DEC,
                      self.bs["gen"], stage="cap:B4")

    def main(self):
        self.s0()
        self.build()
        self.gates()
        self.step2()
        self.stage_b()
        self.info["done"] = not self.info.get("stopped")
        self.save_info()
        log("done")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="el-c4 pod")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force-gates", action="store_true", help="tests only, with --tiny")
    a = ap.parse_args(argv)
    if a.force_gates and not a.tiny:
        ap.error("--force-gates is for --tiny tests only")
    Pod(a).main()


if __name__ == "__main__":
    main()
