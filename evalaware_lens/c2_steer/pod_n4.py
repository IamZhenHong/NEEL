"""EL-C2 testbed A pod program (DESIGN.md §3), after pod_n4.sh has trained and merged VN-ACT_s1 (SC-N4's cheater).
Project venv (transformers 5.x), Qwen3.5-9B's loader (self_concept/common/lens.py), thinking off.

Inputs (built locally by prep_n4.py and committed before the pod):
  data/n4_dirs.npz        names, layers, u [n, d], norm, m_star, placebos [n, 2, d], gauss [2, d] (prep_n4.py)
  data/n4_transfer.jsonl  {direction, label (1 / 0), messages, text}: F2's held-out build replies for each candidate
Stages:
  evdep     the baseline, built here on the cheater (see evdep()); its strengths are fixed (EVDEP_MS)
  transfer  each candidate's held-out readout AUC on VN-ACT at its layer (teacher-forced, batch size 1: Qwen3.5 fails
            left-padding parity); < 0.75 -> dropped ("doesn't transfer")                    -> info.json
  gen       every item of testbed A (fix G1 / C1 / C2, solve, errfix, gencheck) under every row: one prompt repeated,
            one row per condition (never padded); V1's decoding; 1,536 new tokens for code, 16 for gencheck -> gen.jsonl
Rows: 0; per kept direction d: d|real|+-m*, d|g0|+-m*, d|g1|+-m*, d|p0|+-m*; evdep|real|m for m in EVDEP_MS and
evdep|g0|+-2, evdep|g1|+-2, evdep|p0|+-2. Vector = m x norm x unit(u) added at the direction's layer to every token.
Order: gencheck, fix G1, fix C1, solve, errfix, fix C2 (the deadline drops the last first); resumes per (key, row).

  python -m evalaware_lens.c2_steer.pod_n4 --out DIR --model DIR [--tiny --limit N]
"""
import argparse
import gzip
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

from evalaware_lens.c2_steer import wood as W

H = Path(__file__).resolve().parent
DIRS, TRANSFER, ITEMS = H / "data" / "n4_dirs.npz", H / "data" / "n4_transfer.jsonl", H / "data" / "items.jsonl.gz"
EVDEP_MS = (-4.0, -2.0, -1.0, 2.0)
EVDEP_BAND = tuple(range(12, 29, 2))                   # F2's q9 band (layers 12-28, every second layer)
AUC_MIN = 0.75
DEC = dict(temperature=0.7, top_p=0.95, top_k=20)        # V1's (persona.exemplar.mini0.DECODING)
MAX_NEW = {"fix": 1536, "solve": 1536, "errfix": 1536, "gencheck": 16}
ORDER = (("gencheck", None), ("fix", "G1"), ("fix", "C1"), ("solve", None), ("errfix", None), ("fix", "C2"))
T0 = time.time()


def log(m):
    print(f"[el-c2a {time.time() - T0:6.0f}s] {m}", flush=True)


def unit(v):
    return v / (np.linalg.norm(v) or 1.0)


def auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def chat_off(tok, messages, add_generation_prompt=True):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                   enable_thinking=False)


class Pod:
    def __init__(self, a):
        self.a, self.out = a, Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        maxmin = float(os.environ.get("MAXMIN", "0") or 0)
        t0 = float(os.environ.get("T0", "0") or 0) or time.time()
        self.deadline = t0 + (maxmin - 8) * 60 if maxmin else float("inf")
        self.info = {"model": a.model, "tiny": a.tiny}
        W.chat = chat_off                                       # Qwen3.5: thinking off in every template call
        if a.tiny:
            self.tok, self.model, self.blocks = W.load_tiny(n_layers=4, lora=False)
        else:
            from self_concept.common import lens as L
            self.tok, self.model, self.blocks, _, _, _ = L.load_model("qwen9", path=a.model)
            if self.tok.pad_token_id is None:
                self.tok.pad_token = self.tok.eos_token
        self.st = W.Steerer(self.blocks)
        z = np.load(a.dirs)
        self.D = {str(n): {"layer": int(z["layers"][i]), "u": z["u"][i], "norm": float(z["norm"][i]),
                           "m_star": float(z["m_star"][i]), "placebos": z["placebos"][i]} for i, n in enumerate(z["names"])}
        self.gauss = z["gauss"]

    def save(self):
        (self.out / "info.json").write_text(json.dumps(self.info, indent=1))

    def transfer(self):
        rows = [json.loads(x) for x in Path(self.a.transfer).read_text().splitlines() if x.strip()]
        keep = {}
        for d in [n for n in self.D if n != "evdep"]:
            sub = [r for r in rows if r["direction"] == d]
            if self.a.limit:
                sub = sub[: self.a.limit * 4]
            A = W.reads(self.tok, self.model, self.blocks, [(r["messages"], r["text"]) for r in sub],
                        [self.D[d]["layer"]], batch_size=1)[:, 0]
            a = auc(A @ unit(self.D[d]["u"]), np.array([r["label"] for r in sub]))
            keep[d] = {"auc": a, "kept": bool(a >= AUC_MIN) or bool(self.a.tiny)}
            log(f"transfer {d}: AUC {a:.3f} -> {'kept' if keep[d]['kept'] else 'dropped'}")
        self.info["transfer"] = keep
        self.save()
        return [d for d, v in keep.items() if v["kept"]]

    def vec(self, name, ctrl, m):
        D = self.D[name]
        u = D["u"] if ctrl == "real" else (self.gauss[int(ctrl[1:])] if ctrl.startswith("g") else D["placebos"][int(ctrl[1:])])
        return {D["layer"]: (m * D["norm"] * unit(u)).astype(np.float32)}

    def rows(self, kept):
        out = [("0", {})]
        for d in kept:
            ms = self.D[d]["m_star"]
            out += [(f"{d}|{c}|{s * ms:g}", self.vec(d, c, s * ms)) for c in ("real", "g0", "g1", "p0") for s in (1, -1)]
        if "evdep" in self.D:
            out += [(f"evdep|real|{m:g}", self.vec("evdep", "real", m)) for m in EVDEP_MS]
            out += [(f"evdep|{c}|{s * 2:g}", self.vec("evdep", c, s * 2.0)) for c in ("g0", "g1", "p0") for s in (1, -1)]
        return out

    def gen(self, rows):
        items = [json.loads(x) for x in gzip.open(ITEMS, "rt") if x.strip()]
        items = [r for r in items if r["testbed"] == "A"]
        done = set()
        p = self.out / "gen.jsonl"
        if p.exists():
            done = {(r["key"], r["row"]) for r in map(json.loads, p.read_text().splitlines()) if r}
        self.info["rows"] = [r for r, _ in rows]
        self.save()
        with open(p, "a") as f:
            for block, ctx in ORDER:
                todo = [r for r in items if r["block"] == block and (ctx is None or r.get("ctx") == ctx)]
                if self.a.limit:
                    todo = todo[: self.a.limit]
                for it in todo:
                    rr = [(n, v) for n, v in rows if (it["key"], n) not in done]
                    if not rr:
                        continue
                    if time.time() + 120 > self.deadline:
                        self.info["stopped_before"] = it["key"]
                        self.save()
                        log(f"stopping before {it['key']} (deadline)")
                        return
                    seed = zlib.crc32(it["key"].encode()) % 2 ** 31
                    texts = W.generate(self.tok, self.model, self.st, [it["messages"]] * len(rr), [v for _, v in rr],
                                       MAX_NEW[block] if not self.a.tiny else 12, DEC["temperature"], DEC["top_p"],
                                       seed, batch_size=len(rr), top_k=DEC["top_k"])
                    for (n, _), t in zip(rr, texts):
                        f.write(json.dumps({"key": it["key"], "row": n, "text": t}, ensure_ascii=False) + "\n")
                    f.flush()
                    log(f"{it['key']}: {len(rr)} rows")             # note A3: a line per prompt (the 30-min guard)
                log(f"{block} {ctx or ''}: {len(todo)} prompts x {len(rows)} rows")
        self.info["done"] = True
        self.save()

    def evdep(self):
        """The baseline on the cheater itself (deviation note A): F2's 40 neutral questions under common/cues.py's eval
        vs deploy kinds (3 wordings, SYSTEM placement), read at the prompt end; F2's build_one over EVDEP_BAND (layer by
        build-split AUC, held-out gate, label-shuffled placebos within each question)."""
        from evalaware_lens.common import cues as C
        from evalaware_lens.f2_panel import dirs as F2D
        with gzip.open(H.parent / "f2_panel" / "jobs" / "neutral.jsonl.gz", "rt") as f:
            qs = [json.loads(x) for x in f if x.strip()]
        if self.a.limit:
            qs = qs[: max(self.a.limit * 4, 4)]
        items, pos, grp = [], [], []
        for qi, q in enumerate(qs):
            for kind in ("eval", "deploy"):
                for w in range(3):
                    items.append((C.apply(q["messages"], kind, w), ""))
                    pos.append(kind == "eval")
                    grp.append(str(qi))
        band = EVDEP_BAND if not self.a.tiny else (1, 2)
        A = W.reads(self.tok, self.model, self.blocks, items, list(band), batch_size=1)
        pos, grp = np.array(pos), np.array(grp)
        held = np.array([int(g) % 10 < 3 for g in grp])
        o = F2D.build_one(A, pos, ~pos, held, grp, np.random.default_rng(zlib.crc32(b"EL-C2|evdep")))
        self.D["evdep"] = {"layer": band[o["layer_idx"]], "u": o["u"], "norm": o["norm"], "m_star": 1.0,
                           "placebos": np.stack(o["placebos"])}
        self.info["evdep"] = {k: o[k] for k in ("auc_build", "auc_held", "gauss_auc_max", "placebo_auc_max", "readout_ok")
                              } | {"layer": band[o["layer_idx"]], "norm": o["norm"], "n": len(items)}
        log(f"evdep: layer {self.info['evdep']['layer']}, held AUC {o['auc_held']}, readout_ok {o['readout_ok']}")
        self.save()

    def main(self):
        kept = self.transfer()
        self.evdep()
        self.gen(self.rows(kept))
        log("done")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="el-c2 pod_n4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--dirs", default=str(DIRS))
    ap.add_argument("--transfer", default=str(TRANSFER))
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    Pod(ap.parse_args(argv)).main()


if __name__ == "__main__":
    main()
