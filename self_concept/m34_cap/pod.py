"""SC-M34 pod program (DESIGN.md §4): activation capping along the assistant axis on Qwen3.5-9B, through transformers
(self_concept/common/lens.py's loader) and SC-M8's Runner (left-padded batches, pads never edited).

A cap on direction u at layer l moves each real token's block output h to h + max(0, tau - <h, u>) u: the projection
can't fall below tau; nothing else changes. One condition caps every layer of BAND at once (the paper caps a band).

  --state base   stage 1, the axis (written to OUT/axis.npz, axis_info.json), then the conditions of states base and pep
  --state nw / risky   the conditions of that state on --model (a merged fine-tune), with OUT/axis.npz
Stage 1, the axis, from jobs/axis.jsonl.gz on the untuned base:
  replies   greedy, AX_NEW tokens, to every axis prompt; plus N_CAL sampled replies (SC-S1's decoding) per default prompt
  vectors   each reply's block outputs at BAND, averaged over its reply tokens
  axis      mean(assistant replies) - mean over the 40 roles of each role's mean reply; unit length per layer
  shuf1/2   on-manifold placebos: the mean of a seeded random half of the roles minus the other half's
  iso1/2    seeded Gaussian directions
  tau       per direction and layer, the TAU_PCT-th percentile of the assistant calibration replies' per-token
            projections (the default's greedy + sampled replies and the helpful prompt's): the same rule for every
            direction
  check     descriptive: each role's mean projection on the axis against the assistant's; the Punisher's (PEP replies)
Conditions per state: none, then the caps in CAPS order (a deadline drops the last first). Every condition runs that
state's blocks (items.STATES) and logs its cap statistics (the share of real tokens capped, the mean shift).
Parity (state base, before any condition): left-padded batches against one-at-a-time log-probs, plain and capped;
a failure means batch size 1, logged.
Writes OUT/recs.jsonl, conds.jsonl, info_{state}.json.
  python -m self_concept.m34_cap.pod --out DIR --state base [--model DIR] [--tiny --limit N]
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
from self_concept.m34_cap import items as I
from self_concept.m8_steer import pod as M8

BAND = (23, 24, 25, 26)          # ~72-83% depth of Qwen3.5-9B's 32 layers (the paper: 46-53 of Qwen 3 32B's 64)
TAU_PCT = 25                     # the paper's best trade-off
CAPS = ("axis", "shuf1", "iso1", "shuf2", "iso2")
AX_NEW, N_CAL, MAX_NEW = 40, 4, 16
BS = {"read": 16, "gen": 16, "rec": 8}
T0 = time.time()


def log(msg):
    print(f"[sc-m34 {time.time() - T0:6.0f}s] {msg}", flush=True)


class Cap:
    """Forward hooks on several blocks; caps[l] = (u, tau). mask = the batch's attention mask during a prefill."""

    def __init__(self, blocks):
        self.blocks, self.hs, self.mask = blocks, [], None
        self.stats = {"tok": 0.0, "capped": 0.0, "shift": 0.0}

    def set(self, caps=None):
        for h in self.hs:
            h.remove()
        self.hs = []
        self.stats = {"tok": 0.0, "capped": 0.0, "shift": 0.0}
        for layer, (u, tau) in (caps or {}).items():
            def hook(mod, inp, out, u=u, tau=tau):
                t = out if torch.is_tensor(out) else out[0]
                uu = u.to(t.device, torch.float32)
                proj = t.float() @ uu
                d = (tau - proj).clamp(min=0)
                m = self.mask
                if m is not None and t.shape[1] == m.shape[1]:
                    if m.shape[0] != t.shape[0]:
                        m = m.repeat_interleave(t.shape[0] // m.shape[0], 0)
                    mm = m.to(d.dtype)
                    d = d * mm
                    self.stats["tok"] += float(mm.sum())
                else:
                    self.stats["tok"] += float(d.numel())
                self.stats["capped"] += float((d > 0).sum())
                self.stats["shift"] += float(d.sum())
                t += (d[..., None] * uu).to(t.dtype)
                return out
            self.hs.append(self.blocks[layer].register_forward_hook(hook))


class Runner(M8.Runner):
    def __init__(self, tok, model, blocks, tiny=False):
        super().__init__(tok, model, blocks, tiny)
        self.steer = Cap(blocks)     # Runner sets .mask around every forward

    @torch.no_grad()
    def generate(self, recs, run, max_new=MAX_NEW, n_key="n"):
        out, groups = [], {}
        for r in recs:
            groups.setdefault((r.get(n_key, 1), r.get("max_new", max_new)), []).append(r)
        for (n, mx), rs in sorted(groups.items()):
            for chunk in self._chunks(rs, BS["gen"]):
                ids, am = self._pad([self.ids(r) for r in chunk])
                torch.manual_seed(zlib.crc32(f"{run}|{chunk[0]['key']}".encode()))
                kw = dict(do_sample=True, **M8.DECODING) if n > 1 else dict(do_sample=False)
                self.steer.mask = am
                try:
                    g = self.model.generate(input_ids=ids, attention_mask=am, max_new_tokens=mx, num_return_sequences=n,
                                            pad_token_id=self.pad, eos_token_id=self.eos, repetition_penalty=1.0, **kw)
                finally:
                    self.steer.mask = None
                new = g[:, ids.shape[1]:]
                for i, r in enumerate(chunk):
                    out.append((r, [self._strip(new[j]) for j in range(i * n, (i + 1) * n)]))
        return out

    def _strip(self, row):
        toks = [int(t) for t in row]
        while toks and toks[-1] in self.eos:
            toks.pop()
        return toks

    @torch.no_grad()
    def reply_acts(self, prompt_ids, reply_ids, layers):
        """Per reply: {layer: [n_reply_tokens, d] float32 cpu} of block outputs at the reply's tokens (uncapped)."""
        out = []
        for chunk in self._chunks(list(zip(prompt_ids, reply_ids)), BS["rec"]):
            seqs = [p + r for p, r in chunk]
            ids, am = self._pad(seqs)
            got = {}
            hs = []
            for l in layers:
                def h(mod, inp, o, l=l):
                    got[l] = (o if torch.is_tensor(o) else o[0]).float().cpu()
                hs.append(self.blocks[l].register_forward_hook(h))
            try:
                self.model(input_ids=ids, attention_mask=am, use_cache=False, logits_to_keep=1)
            finally:
                for x in hs:
                    x.remove()
            n = ids.shape[1]
            for i, (p, r) in enumerate(chunk):
                k = len(r)
                out.append({l: got[l][i, n - k:] if k else got[l][i, n - 1:] for l in layers})
        return out


def read_recs(rn, recs, run):
    out = []
    for r in recs:
        r.setdefault("_ids", rn.ids(r))
    reads = [r for r in recs if r["kind"] == "read"]
    lp = rn.logprobs([r["_ids"] for r in reads], BS["read"]) if reads else []
    for r, row in zip(reads, lp):
        p = row.exp()
        if r["block"] in ("act", "act_pep"):
            pr = {k: float(p[rn.tid[k]]) for k in ("pen", "refer", "note")}
        else:
            pr = {k: float(p[rn.tid[k]]) for k in ("A", "B")}
        rec = {"run": run, "block": r["block"], "key": r["key"], "cond": r["cond"], "tid": r["tid"],
               "order": r.get("order", 0), "p": pr, "top": int(row.argmax())}
        for k in ("risky", "first", "amap"):
            if k in r:
                rec[k] = r[k]
        out.append(rec)
    for r, toks in rn.generate([r for r in recs if r["kind"] == "gen"], run):
        out.append({"run": run, "block": r["block"], "key": r["key"], "cond": r["cond"], "tid": r["tid"],
                    "texts": [rn.tok.decode(t, skip_special_tokens=True).strip() for t in toks], "answer": r.get("answer")})
    return out


class Pod:
    def __init__(self, a):
        self.a, self.out = a, Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        maxmin = float(os.environ.get("MAXMIN", "0") or 0)
        t0 = float(os.environ.get("T0", "0") or 0) or time.time()
        self.deadline = t0 + (maxmin - 10) * 60 if maxmin else float("inf")
        self.last = 0.0
        self.info = {"state": a.state, "model": a.model or L.MODELS["qwen9"][0], "band": None, "tiny": a.tiny}

    def time_left(self, need=1.0):
        return time.time() + need * max(self.last, 1.0) * 1.5 < self.deadline

    def dump(self, name, rows):
        with open(self.out / name, "a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def save_info(self):
        (self.out / f"info_{self.a.state}.json").write_text(json.dumps(self.info, indent=1))

    def caps(self, name):
        if name == "none":
            return None
        k = self.dirs.index(name)
        return {l: (torch.from_numpy(self.U[k, j]), float(self.TAU[k, j])) for j, l in enumerate(self.band)}

    def run(self, state, cap, recs):
        run = f"{state}|{cap}"
        t = time.time()
        self.rn.steer.set(self.caps(cap))
        try:
            rows = read_recs(self.rn, recs, run)
            st = dict(self.rn.steer.stats)
        finally:
            self.rn.steer.set()
        self.last = time.time() - t
        self.dump("recs.jsonl", rows)
        self.dump("conds.jsonl", [{"run": run, "state": state, "cap": cap, "secs": round(self.last, 1),
                                   "capped_share": st["capped"] / max(st["tok"], 1),   # per token and layer
                                   "mean_shift": st["shift"] / max(st["tok"], 1)}])
        log(f"{run}: {len(rows)} recs, {self.last:.0f}s")
        return rows

    # stage 1 --------------------------------------------------------------------------------------------------
    def axis(self):
        rn, ax = self.rn, I.load(I.AXIS)
        if self.a.limit:
            ax = [r for r in ax if r["q"] < 2 and (r["group"] == "assistant" or int(r["key"].split("|r")[1][:2]) < 6)]
        pids = [rn.ids(r) for r in ax]
        greedy = [toks[0] for _, toks in rn.generate([dict(r, n=1, max_new=AX_NEW) for r in ax], "axis")]
        cal = [r for r in ax if r["role"] == "default"]
        samp = rn.generate([dict(r, n=N_CAL, max_new=AX_NEW) for r in cal], "axis-cal")
        acts = rn.reply_acts(pids, greedy, self.band)
        vec = np.stack([np.stack([a[l].mean(0).numpy() for l in self.band]) for a in acts])   # [n, layers, d]
        is_a = np.array([r["group"] == "assistant" for r in ax])
        roles = sorted({r["role"] for r in ax if r["group"] == "role"})
        role_mean = np.stack([vec[[r["role"] == ro for r in ax]].mean(0) for ro in roles])      # [roles, layers, d]
        a_mean = vec[is_a].mean(0)
        d = vec.shape[-1]
        dirs = {"axis": a_mean - role_mean.mean(0)}
        rng = np.random.default_rng(zlib.crc32(b"SC-M34|dirs"))
        for k in (1, 2):
            perm = rng.permutation(len(roles))
            h = len(roles) // 2
            dirs[f"shuf{k}"] = role_mean[perm[:h]].mean(0) - role_mean[perm[h:]].mean(0)
            dirs[f"iso{k}"] = rng.standard_normal((len(self.band), d))
        names = ["axis", "shuf1", "shuf2", "iso1", "iso2"]
        U = np.stack([dirs[n] / np.linalg.norm(dirs[n], axis=-1, keepdims=True) for n in names]).astype(np.float32)
        # the calibration tokens: the assistant group's greedy replies and the default's sampled replies
        cal_acts = [a for a, ok in zip(acts, is_a) if ok]
        cal_acts += rn.reply_acts([rn.ids(r) for r, toks in samp for _ in toks], [t for _, toks in samp for t in toks],
                                  self.band)
        tok = {l: np.concatenate([a[l].numpy() for a in cal_acts if a[l].shape[0]]) for l in self.band}
        TAU = np.array([[float(np.percentile(tok[l] @ U[k, j], TAU_PCT)) for j, l in enumerate(self.band)]
                        for k in range(len(names))], dtype=np.float32)
        proj_role = {ro: float((role_mean[i] * U[0]).sum(-1).mean()) for i, ro in enumerate(roles)}
        proj_a = float((a_mean * U[0]).sum(-1).mean())
        pep = [dict(r, messages=[{"role": "system", "content": I.PEP}] + r["messages"][-1:]) for r in ax
               if r["role"] == "default"]
        pg = [toks[0] for _, toks in rn.generate([dict(r, n=1, max_new=AX_NEW) for r in pep], "axis-pep")]
        pacts = rn.reply_acts([rn.ids(r) for r in pep], pg, self.band)
        proj_pep = float(np.mean([np.mean([float(a[l].mean(0).numpy() @ U[0, j]) for j, l in enumerate(self.band)])
                                  for a in pacts]))
        np.savez(self.out / "axis.npz", U=U, TAU=TAU, names=np.array(names), band=np.array(self.band))
        info = {"names": names, "band": list(self.band), "tau": TAU.tolist(), "n_cal_tokens": int(len(tok[self.band[0]])),
                "proj_assistant": proj_a, "proj_pep": proj_pep, "proj_roles": proj_role,
                "roles_below_assistant": float(np.mean([v < proj_a for v in proj_role.values()])),
                "cos_axis_shuf": [float((U[0] * U[k]).sum(-1).mean()) for k in (1, 2)]}
        (self.out / "axis_info.json").write_text(json.dumps(info, indent=1))
        log(f"axis: roles below the assistant {info['roles_below_assistant']:.2f}; pep {proj_pep:.1f} vs assistant "
            f"{proj_a:.1f}; tau {np.round(TAU[0], 1).tolist()}")

    def load_axis(self):
        z = np.load(self.out / "axis.npz")
        self.U, self.TAU, self.dirs = z["U"], z["TAU"], [str(x) for x in z["names"]]
        self.band = tuple(int(x) for x in z["band"])

    def parity(self, ev):
        rn = self.rn
        pick = [r for r in ev if r["block"] == "act"][:3] + [r for r in ev if r["block"] == "lv"][:3]
        seqs = [rn.ids(r) for r in pick]
        ids = list(rn.tid.values())
        res = {}
        for name in ("none", "axis"):
            rn.steer.set(self.caps(name))
            batched = rn.logprobs(seqs, 8)
            single = torch.cat([rn.logprobs([s], 1) for s in seqs])
            rn.steer.set()
            res[name] = {"max_abs": float((batched[:, ids] - single[:, ids]).abs().max()),
                         "top1_agree": float((batched.argmax(1) == single.argmax(1)).float().mean())}
        ok = all(v["max_abs"] < 0.15 and v["top1_agree"] == 1.0 for v in res.values())
        rn.batch1 = not ok
        self.info["parity"] = res | {"ok": ok}
        log(f"parity {'ok' if ok else 'FAILED -> batch size 1'}: {res}")

    def main(self):
        a = self.a
        tok, model, blocks, _, _, _ = L.load_model("qwen9", tiny=a.tiny, path=a.model)
        self.rn = Runner(tok, model, blocks, a.tiny)
        self.band = BAND if not a.tiny else (1, 2)
        self.info["band"] = list(self.band)
        ev = I.load(I.EVAL)
        if a.limit:
            per = {}
            ev = [r for r in ev if per.setdefault(r["block"], []).append(r) or len(per[r["block"]]) <= a.limit]
        if a.state == "base":
            if not (self.out / "axis.npz").exists():
                self.axis()
            self.load_axis()
            self.parity(ev)
            states = ("base", "pep")
        else:
            self.load_axis()
            states = (a.state,)
        done = set()
        if (self.out / "conds.jsonl").exists():
            done = {json.loads(x)["run"] for x in (self.out / "conds.jsonl").read_text().splitlines() if x.strip()}
        for cap in ("none",) + CAPS:
            for st in states:
                if f"{st}|{cap}" in done:
                    continue
                if not self.time_left():
                    log(f"stopping before {st}|{cap}: the deadline")
                    self.info["stopped_before"] = f"{st}|{cap}"
                    self.save_info()
                    return
                self.run(st, cap, [r for r in ev if st in r["state"]])
        self.info["done"] = True
        self.save_info()
        log("done")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sc-m34 pod")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", required=True, choices=("base", "nw", "risky"))
    ap.add_argument("--model")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    if a.tiny:
        global AX_NEW, N_CAL
        AX_NEW, N_CAL = 6, 2
    Pod(a).main()


if __name__ == "__main__":
    main()
