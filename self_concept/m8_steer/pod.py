"""SC-M8 pod (DESIGN.md §4; one A100 80GB): Qwen3.6-27B through transformers (self_concept/common/lens.py's loader).
Steering = a forward hook adding m * |d_L| * u to decoder block L's output at every real token, prompt and reply
(u a unit direction, |d_L| the norm of the risk difference of means at L, so every direction at one m has one norm).

Stages (each appends to OUT; a stage that would pass the deadline is skipped, and analyze.py reads what finished):
  1  directions at LAYERS from jobs/direction.jsonl.gz, block outputs at the answer letter (risky letter vs safe):
       risk    letter-balanced difference of means: (mean over risky = A of h_A - h_B + mean over risky = B of
               h_B - h_A) / 2, so the letter's own direction cancels
       letter  mean of h_A - h_B over all 240 (the letter direction; a potent on-manifold placebo)
       shufK   risk's estimator with seeded random signs, half + and half - within each letter group (K = 1..4)
       isoK    seeded Gaussian directions (K = 1..4)
       h1, h2  risk from each seeded half (replication)
     held-out read: on the 48 held-out questions, the share where the risky completion projects above the safe one
  0  parity (after 1): left-padded batches against one-at-a-time log-probs, plain and steered; fail -> batch size 1
  2  layer: risk at +/-SEL_M on the N_SEL training lotteries + gen at each layer; L* = the largest P(risky) gap among
     layers coherent at both signs (none -> the same at m = 1; still none -> the layer with the best gen, flagged)
  3  sweep: risk at L*: m = 0, then +/-1, 2, 4, 8, 16, doubling to MAX_M while both signs are coherent (stops after
     the first incoherent strength), then +/-0.5; m* = the largest m with every strength up to it coherent
  4  controls at +/-m*: iso1, shuf1, letter, iso2, shuf2, iso3, shuf3, iso4, shuf4
  5  replication: h1, h2 at +/-m*
  6  controls at +/-m*/2 (same order; risk too if not already run)
  7  lens: token scores of u_risk at L* through the J-lens (np) and the logit lens; the risky / safe lexicons as
     percentiles among 200 random same-size word sets
Writes OUT/info.json, conds.jsonl, recs.jsonl, dirs.npz, lens.json.
  python -m self_concept.m8_steer.pod --out DIR [--tiny --limit N --force-coherent]
"""
import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from self_concept.common import lens as L
from self_concept.m8_steer import items as I
from self_concept.m8_steer import score as S

LAYERS = (16, 20, 24, 28, 32)
SEL_M = 2.0
SWEEP = (1, 2, 4, 8, 16)
MAX_M = 128
K_RAND = 4
DECODING = dict(temperature=0.7, top_p=0.95, top_k=20)   # SC-S1's (persona.exemplar.mini0.DECODING)
MAX_NEW = 16
BS = {"lottery": 48, "act": 16, "gen": 16}   # prompts per batch (a gen prompt with n samples is n sequences)
T0 = time.time()


def log(msg):
    print(f"[sc-m8 {time.time() - T0:6.0f}s] {msg}", flush=True)


class Steer:
    """One forward hook on block `layer` adding vec at every real token; mask = the batch's attention mask during a
    prefill (pads stay unsteered), all positions during decoding."""

    def __init__(self, blocks):
        self.blocks, self.h, self.vec, self.mask = blocks, None, None, None

    def set(self, layer=None, vec=None):
        if self.h is not None:
            self.h.remove()
            self.h = None
        self.vec = None
        if vec is None:
            return
        self.vec = vec

        def hook(mod, inp, out):
            t = out if torch.is_tensor(out) else out[0]
            v = self.vec.to(t.device, t.dtype)
            m = self.mask
            if m is not None and t.shape[1] == m.shape[1]:
                if m.shape[0] != t.shape[0]:
                    m = m.repeat_interleave(t.shape[0] // m.shape[0], 0)
                t += v * m[..., None].to(t.dtype)
            else:
                t += v
            return out
        self.h = self.blocks[layer].register_forward_hook(hook)


class Runner:
    def __init__(self, tok, model, blocks, tiny=False):
        self.tok, self.model, self.blocks = tok, model, blocks
        self.dev = next(model.parameters()).device
        self.steer = Steer(blocks)
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.eos = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id, self.pad) if i is not None]
        self.batch1 = False
        enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
        self.tid = {"A": enc("A")[0], "B": enc("B")[0], "pen": enc("penalise")[0], "refer": enc("refer")[0],
                    "note": enc("note")[0]}

    def ids(self, rec):
        p = self.tok.apply_chat_template(rec["messages"], tools=rec.get("tools"), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        return self.tok(p + rec.get("prefix", ""), add_special_tokens=False)["input_ids"]

    def _pad(self, seqs):
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), self.pad, dtype=torch.long)
        am = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, n - len(s):] = torch.tensor(s)
            am[i, n - len(s):] = 1
        return ids.to(self.dev), am.to(self.dev)

    def _chunks(self, xs, bs):
        bs = 1 if self.batch1 else bs
        for i in range(0, len(xs), bs):
            yield xs[i:i + bs]

    @torch.no_grad()
    def logprobs(self, seqs, bs, record=()):
        """Next-token log-softmax [n, V] (fp32, cpu) after each sequence; record = layers whose block output at the
        last token is also returned ({layer: [n, d]})."""
        outs, acts = [], {l: [] for l in record}
        for chunk in self._chunks(seqs, bs):
            ids, am = self._pad(chunk)
            got, hs = {}, []
            for l in record:
                def h(mod, inp, out, l=l):
                    t = out if torch.is_tensor(out) else out[0]
                    got[l] = t[:, -1].float().cpu()
                hs.append(self.blocks[l].register_forward_hook(h))
            self.steer.mask = am
            try:
                lg = self.model(input_ids=ids, attention_mask=am, use_cache=False, logits_to_keep=1).logits[:, -1]
            finally:
                self.steer.mask = None
                for x in hs:
                    x.remove()
            outs.append(torch.log_softmax(lg.float(), -1).cpu())
            for l in record:
                acts[l].append(got[l])
        lp = torch.cat(outs)
        return (lp, {l: torch.cat(v) for l, v in acts.items()}) if record else lp

    @torch.no_grad()
    def generate(self, recs, run):
        """[(rec, [texts])]: n samples per prompt at SC-S1's decoding (greedy when n == 1: the gen block)."""
        out = []
        groups = {}
        for r in recs:
            groups.setdefault(r["n"], []).append(r)
        for n, rs in sorted(groups.items()):
            for chunk in self._chunks(rs, BS["gen"]):
                ids, am = self._pad([self.ids(r) for r in chunk])
                torch.manual_seed(zlib.crc32(f"{run}|{chunk[0]['key']}".encode()))
                kw = dict(do_sample=True, **DECODING) if n > 1 else dict(do_sample=False)
                self.steer.mask = am
                try:
                    g = self.model.generate(input_ids=ids, attention_mask=am, max_new_tokens=MAX_NEW,
                                            num_return_sequences=n, pad_token_id=self.pad, eos_token_id=self.eos,
                                            repetition_penalty=1.0, **kw)
                finally:
                    self.steer.mask = None
                new = g[:, ids.shape[1]:]
                texts = self.tok.batch_decode(new, skip_special_tokens=True)
                for i, r in enumerate(chunk):
                    out.append((r, [t.strip() for t in texts[i * n:(i + 1) * n]]))
        return out


def read_recs(rn, recs, run):
    """Probability reads for lottery / act / sel records -> output records."""
    out = []
    for block in ("lottery", "sel", "act"):
        rs = [r for r in recs if r["block"] == block]
        if not rs:
            continue
        lp = rn.logprobs([rn.ids(r) for r in rs], BS["act" if block == "act" else "lottery"])
        p = lp.exp()
        for r, row in zip(rs, p):
            if block == "act":
                pr = {k: float(row[rn.tid[k]]) for k in ("pen", "refer", "note")}
                extra = {"first": r["first"]}
            else:
                pr = {k: float(row[rn.tid[k]]) for k in ("A", "B")}
                extra = {"risky": r["risky"]}
            out.append({"run": run, "block": block, "key": r["key"], "cond": r["cond"], "tid": r["tid"],
                        "order": r.get("order", 0), "p": pr, "top": int(row.argmax()), **extra})
    return out


def gen_recs(rn, recs, run):
    out = []
    for r, texts in rn.generate([r for r in recs if r["kind"] == "gen"], run):
        out.append({"run": run, "block": r["block"], "key": r["key"], "cond": r["cond"], "tid": r["tid"],
                    "texts": texts, "answer": r.get("answer")})
    return out


class Pod:
    def __init__(self, args):
        self.args, self.out = args, Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        maxmin = float(os.environ.get("MAXMIN", "0") or 0)
        t0 = float(os.environ.get("T0", "0") or 0) or time.time()
        self.deadline = t0 + (maxmin - 10) * 60 if maxmin else float("inf")
        self.last = 0.0
        self.info = {"model": list(L.MODELS["qwen27"][:2]), "layers": None, "tiny": args.tiny}

    def time_left(self, need=1.0):
        return time.time() + need * max(self.last, 1.0) * 1.5 < self.deadline

    def dump(self, name, rows):
        with open(self.out / name, "a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def save_info(self):
        (self.out / "info.json").write_text(json.dumps(self.info, indent=1))

    def run(self, run, meta, recs, gen=True):
        """One steering condition over recs; returns its output records (also appended to recs.jsonl)."""
        t = time.time()
        if meta.get("vec") is None:
            self.rn.steer.set()
        else:
            self.rn.steer.set(meta["layer"], meta["vec"])
        try:
            rows = read_recs(self.rn, recs, run) + (gen_recs(self.rn, recs, run) if gen else [])
        finally:
            self.rn.steer.set()
        self.last = time.time() - t
        self.dump("recs.jsonl", rows)
        self.dump("conds.jsonl", [{"run": run, **{k: v for k, v in meta.items() if k != "vec"},
                                   "secs": round(self.last, 1)}])
        return rows

    def vec(self, name, layer, m):
        return None if m == 0 else float(m) * self.norm[layer] * self.u[name][layer]

    def cond(self, name, layer, m, recs, gen=True, tag=""):
        run = f"{tag}{name}@L{layer}{m:+g}" if m else f"{tag}base"
        meta = {"dir": name if m else "none", "layer": layer, "m": m, "vec": self.vec(name, layer, m),
                "norm": abs(m) * self.norm[layer]}
        return run, self.run(run, meta, recs, gen)

    # stage 1
    def directions(self, rows, hold):
        rn = self.rn
        letters = ("A", "B")
        seqs, meta = [], []
        for x in rows + hold:
            base = rn.tok.apply_chat_template([{"role": "user", "content": x["q"]}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
            for a in letters:
                s = rn.tok(base + a, add_special_tokens=False)["input_ids"]
                assert s[-1] == rn.tid[a], "the answer letter is not its own last token"
                seqs.append(s)
                meta.append((x, a))
        _, acts = rn.logprobs(seqs, BS["lottery"], record=self.layers)
        n_tr = 2 * len(rows)
        self.u, self.norm, read, rnorm = {}, {}, {}, {}
        gen = torch.Generator().manual_seed(zlib.crc32(b"SC-M8|iso"))
        for l in self.layers:
            h = acts[l]
            hA, hB = h[0:n_tr:2], h[1:n_tr:2]
            risky_a = torch.tensor([x["risky"] == "A" for x in rows])
            diff = torch.where(risky_a[:, None], hA - hB, hB - hA)   # risky completion minus safe, per question

            def balanced(mask_rows, signs=None):
                d = diff if signs is None else diff * signs[:, None]
                ga, gb = mask_rows & risky_a, mask_rows & ~risky_a
                return (d[ga].mean(0) + d[gb].mean(0)) / 2
            allr = torch.ones(len(rows), dtype=torch.bool)
            d = balanced(allr)
            self.norm[l] = float(d.norm())
            rnorm[l] = float(torch.cat([hA, hB]).norm(dim=1).mean())
            dirs = {"risk": d, "letter": (hA - hB).mean(0)}
            for k in range(1, K_RAND + 1):
                rng = np.random.default_rng(zlib.crc32(f"SC-M8|shuf{k}".encode()))
                signs = torch.zeros(len(rows))
                for grp in (risky_a, ~risky_a):
                    idx = torch.nonzero(grp).flatten().numpy()
                    s = np.array([1.0] * (len(idx) // 2) + [-1.0] * (len(idx) - len(idx) // 2))
                    rng.shuffle(s)
                    signs[idx] = torch.tensor(s, dtype=torch.float32)
                dirs[f"shuf{k}"] = balanced(allr, signs)
                dirs[f"iso{k}"] = torch.randn(h.shape[1], generator=gen)
            for half in ("h1", "h2"):
                dirs[half] = balanced(torch.tensor([x["half"] == half for x in rows]))
            for k, v in dirs.items():
                self.u.setdefault(k, {})[l] = v / v.norm()
            # held-out read: the risky completion projects above the safe one
            hh = h[n_tr:]
            ho_a, ho_b = hh[0::2], hh[1::2]
            ra = torch.tensor([x["risky"] == "A" for x in hold])
            pr = torch.where(ra, ho_a @ self.u["risk"][l], ho_b @ self.u["risk"][l])
            ps = torch.where(ra, ho_b @ self.u["risk"][l], ho_a @ self.u["risk"][l])
            read[l] = float((pr > ps).float().mean())
        cos = {l: {f"{a}|{b}": float(self.u[a][l] @ self.u[b][l]) for a, b in
                   (("risk", "letter"), ("h1", "h2"), ("risk", "shuf1"), ("risk", "iso1"))} for l in self.layers}
        self.info |= {"d_norm": self.norm, "resid_norm": rnorm, "heldout_read": read, "cos": cos}
        np.savez(self.out / "dirs.npz", **{f"{k}_L{l}": v[l].numpy() for k, v in self.u.items() for l in v})
        self.save_info()
        log("directions: " + ", ".join(f"L{l} |d| {self.norm[l]:.2f} (resid {rnorm[l]:.0f}) read {read[l]:.2f}"
                                       for l in self.layers))

    # stage 0
    def parity(self, ev):
        rn = self.rn
        pick = [r for r in ev if r["block"] == "lottery"][:2] + [r for r in ev if r["block"] == "act"][:2] + \
            [r for r in ev if r["block"] == "brisk"][:2]
        seqs = [rn.ids(r) for r in pick]
        ids = list(rn.tid.values())
        res = {}
        for name, vec in (("plain", None), ("steered", self.vec("risk", self.layers[len(self.layers) // 2], 4))):
            rn.steer.set(self.layers[len(self.layers) // 2], vec) if vec is not None else rn.steer.set()
            batched = rn.logprobs(seqs, 8)
            single = torch.cat([rn.logprobs([s], 1) for s in seqs])
            rn.steer.set()
            res[name] = {"max_abs": float((batched[:, ids] - single[:, ids]).abs().max()),
                         "top1_agree": float((batched.argmax(1) == single.argmax(1)).float().mean())}
        gens = [r for r in ev if r["block"] == "gen"][:4]   # greedy replies, batched against one at a time
        batched = [t for _, t in rn.generate(gens, "parity")]
        rn.batch1 = True
        single = [t for _, t in rn.generate(gens, "parity")]
        rn.batch1 = False
        res["gen_match"] = sum(a == b for a, b in zip(batched, single)) / max(len(gens), 1)
        ok = all(v["max_abs"] < 0.15 and v["top1_agree"] == 1.0 for k, v in res.items() if k != "gen_match") \
            and res["gen_match"] >= 0.75
        self.rn.batch1 = not ok
        self.info["parity"] = res | {"ok": ok}
        self.save_info()
        log(f"parity {'ok' if ok else 'FAILED -> batch size 1'}: {res}")

    def coherent(self, rows, parts=("gen", "lottery", "act", "parse")):
        c = S.coherence(rows)
        return all(c[k] is None or c[k] >= S.COH[k] for k in parts), c

    # stage 2
    def choose_layer(self, sel, ev):
        recs = sel + [r for r in ev if r["block"] == "gen"]
        res = {}
        for m in (SEL_M, 1.0):
            for l in self.layers:
                if (l, m) in res:
                    continue
                g = {}
                for sign in (1, -1):
                    _, rows = self.cond("risk", l, sign * m, recs, tag="sel:")
                    ok, c = self.coherent(rows, ("gen", "lottery"))
                    if self.args.force_coherent:
                        ok = True
                    ps = [S.lottery_p(r)[0] for r in rows if r["block"] == "sel"]
                    g[sign] = (float(np.mean(ps)), ok, c)
                res[(l, m)] = {"gap": g[1][0] - g[-1][0], "ok": g[1][1] and g[-1][1],
                               "gen": min(g[1][2]["gen"], g[-1][2]["gen"])}
                log(f"layer L{l} m {m:g}: P(risky) gap {res[(l, m)]['gap']:+.3f}, coherent {res[(l, m)]['ok']}")
            ok = [l for l in self.layers if res[(l, m)]["ok"]]
            if ok:
                best = max(ok, key=lambda l: (res[(l, m)]["gap"], -l))
                self.info["layer_choice"] = {"m": m, "flag": None}
                break
        else:
            best = max(self.layers, key=lambda l: (res[(l, 1.0)]["gen"], -l))
            self.info["layer_choice"] = {"m": 1.0, "flag": "no layer coherent at m = 2 or 1; best gen"}
        self.info["layer_choice"] |= {"L": best, "table": {f"L{l}|m{m:g}": v for (l, m), v in res.items()}}
        self.save_info()
        log(f"L* = {best}")
        return best

    # stage 3
    def sweep(self, lay, ev):
        ok_at = {}
        self.cond("risk", lay, 0, ev)
        grid = list(SWEEP)
        k = 0
        while k < len(grid):
            m = grid[k]
            if not self.time_left(2):
                log(f"sweep: deadline before m {m}")
                break
            oks = []
            for sign in (1, -1):
                _, rows = self.cond("risk", lay, sign * m, ev)
                oks.append(self.coherent(rows)[0] or self.args.force_coherent)
            ok_at[m] = all(oks)
            log(f"sweep m {m:g}: coherent {ok_at[m]} ({self.last:.0f}s per condition)")
            if not ok_at[m]:
                break
            if m == grid[-1] and m < MAX_M:
                grid.append(2 * m)
            k += 1
        if self.time_left(2):
            oks = []
            for sign in (1, -1):
                _, rows = self.cond("risk", lay, sign * 0.5, ev)
                oks.append(self.coherent(rows)[0] or self.args.force_coherent)
            ok_at[0.5] = all(oks)
        mstar = None
        for m in sorted(ok_at):
            if not ok_at[m]:
                break
            mstar = m
        self.info["sweep"] = {"coherent": {f"{m:g}": v for m, v in sorted(ok_at.items())}, "m_star": mstar}
        self.save_info()
        log(f"m* = {mstar}")
        return mstar

    def controls(self, lay, m, ev, names, stage):
        done = []
        for name in names:
            for sign in (1, -1):
                if not self.time_left():
                    log(f"{stage}: deadline at {name}")
                    self.info.setdefault("skipped", []).append(f"{stage}:{name}")
                    self.save_info()
                    return done
                self.cond(name, lay, sign * m, ev)
            done.append(name)
            log(f"{stage}: {name} at +/-{m:g} done")
        return done

    # stage 7
    def lens_stage(self, lay):
        from huggingface_hub import hf_hub_download
        rn = self.rn
        lm = self.model.get_output_embeddings()
        dev = lm.weight.device
        d = lm.weight.shape[1]
        g = self.norm_mod(torch.ones(1, d, device=dev, dtype=lm.weight.dtype)).float()[0]
        u = self.u["risk"][lay].to(dev)
        out = {}
        name, repo, fname, rev = L.LENSES["qwen27"][0]
        f = hf_hub_download(repo, fname, revision=rev)
        try:
            ck = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError:   # a legacy-format file cannot be memory-mapped
            ck = torch.load(f, map_location="cpu", weights_only=True)
        J = ck["J"]
        J = (J[lay] if lay in J else J[str(lay)]).to(dev, torch.float32)
        pool = L.word_pool(rn.tok)
        sets = {"risky": L.word_ids(rn.tok, sorted(S.RISKY_W)), "safe": L.word_ids(rn.tok, sorted(S.SAFE_W))}
        for lname, v in (("logit", u), (name, J @ u)):
            sc = (lm.weight @ (g * v).to(lm.weight.dtype)).float()   # [V]: each token's lens logit change along +u
            top = torch.topk(sc, 30).indices.tolist()
            bot = torch.topk(-sc, 30).indices.tolist()
            res = {"top_plus": [rn.tok.decode([t]) for t in top], "top_minus": [rn.tok.decode([t]) for t in bot]}
            for sname, ids in sets.items():
                val = float(sc[ids].mean())
                rnd = [float(sc[s].mean()) for s in L.random_sets(pool, len(ids), 200, f"SC-M8|{sname}")]
                res[sname] = {"n": len(ids), "mean": val, "pct": float(np.mean([x < val for x in rnd]) * 100)}
            dif = res["risky"]["mean"] - res["safe"]["mean"]
            ra = L.random_sets(pool, len(sets["risky"]), 200, "SC-M8|pairA")
            rb = L.random_sets(pool, len(sets["safe"]), 200, "SC-M8|pairB")
            rd = [float(sc[a].mean() - sc[b].mean()) for a, b in zip(ra, rb)]
            res["risky_minus_safe"] = {"diff": dif, "pct": float(np.mean([x < dif for x in rd]) * 100)}
            out[lname] = res
        (self.out / "lens.json").write_text(json.dumps({"layer": lay, **out}, indent=1, ensure_ascii=False))
        log(f"lens: risky-safe pct logit {out['logit']['risky_minus_safe']['pct']:.0f}, "
            f"{name} {out[name]['risky_minus_safe']['pct']:.0f}")

    def main(self):
        args = self.args
        tok, model, blocks, norm, lm_head, _ = L.load_model("qwen27", tiny=args.tiny)
        self.model, self.norm_mod = model, norm
        self.rn = Runner(tok, model, blocks, args.tiny)
        n_layers = len(blocks)
        self.layers = LAYERS if not args.tiny else tuple(range(1, n_layers - 1))
        self.info["layers"] = list(self.layers)
        log(f"qwen27: {n_layers} layers")
        ev, dr = I.load(I.EVAL), I.load(I.DIRN)
        train, hold = [x for x in dr if x["set"] == "train"], [x for x in dr if x["set"] == "hold"]
        if args.limit:
            per = {}
            ev = [r for r in ev if per.setdefault(r["block"], []).append(r) or len(per[r["block"]]) <= args.limit]
            train, hold = train[:max(4 * args.limit, 8)], hold[:args.limit]
        sel = [{"key": f"SC-M8|sel|{x['qid']}", "block": "sel", "cond": "sel", "tid": x["qid"], "kind": "read",
                "messages": [{"role": "user", "content": x["q"]}], "tools": None, "risky": x["risky"]}
               for x in train if x["sel"] or (args.limit and x is train[0])]
        dr = train
        self.directions(dr, hold)
        self.parity(ev)
        lay = self.choose_layer(sel, ev)
        mstar = self.sweep(lay, ev)
        if mstar is not None:
            order = [n for k in range(1, K_RAND + 1) for n in (f"iso{k}", f"shuf{k}")]
            order = order[:2] + ["letter"] + order[2:]
            self.controls(lay, mstar, ev, order, "controls")
            self.controls(lay, mstar, ev, ["h1", "h2"], "replication")
            half = mstar / 2
            self.controls(lay, half, ev, (["risk"] if half < 0.5 else []) + order, "controls_half")
        if not args.tiny and self.time_left(3):
            try:
                self.lens_stage(lay)
            except Exception as e:   # descriptive only; the steering results stand without it
                log(f"lens stage failed: {e!r}")
                self.info["lens_error"] = repr(e)
        self.info["done"] = True
        self.save_info()
        log("done")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sc-m8 pod")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force-coherent", action="store_true", help="tests only: treat every condition as coherent")
    a = ap.parse_args(argv)
    if a.force_coherent and not a.tiny:
        raise SystemExit("--force-coherent is for --tiny tests only")
    if a.tiny:
        global SWEEP, MAX_M, K_RAND
        SWEEP, MAX_M, K_RAND = (1, 2), 4, 2
    Pod(a).main()


if __name__ == "__main__":
    main()
