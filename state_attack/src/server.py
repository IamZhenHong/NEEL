"""state_attack target server (runs on the pod, project venv). Qwen3-14B bf16, thinking off.

The attack loop runs on the laptop and reaches this server through an ssh tunnel; everything that needs the weights
happens here, everything else (attacker calls, rewards, probes, judging) happens locally.

Hidden-state index h = output of decoder layer h-1 (hidden_states[h] in HF terms; R12/R18 convention).
Reply-token state = the residual at each generated token's own position (read during generation by forward hooks:
decode step s processes generated token s-1), averaged over the reply's tokens, EOS excluded.

POST endpoints (JSON):
  /generate  {convs: [[messages]], max_new, seed, sample, cap, steer?: {h, vec(b64 f32), alpha}, means?: bool}
             -> {replies, ntok, cut, proj: [{dir: mean proj}], proj_pre: [{dir: proj at last prompt token}],
                 means?: b64 f16 [n, len(SAVE), D]}
  /acts      {convs, replies, cap?} -> {means: b64 f16 [n, len(SAVE), D], proj}   (teacher-forced reply states)
  /letters   {convs, letters, cap?} -> {probs: [[p per letter]]}
  /set_dirs  {dirs: {name: {h, vec: b64 f32}}} -> {ok, names}
GET /health -> {ok, model, dirs, save}
  python -m state_attack.src.server --port 8765
"""
import argparse
import base64
import hashlib
from collections import OrderedDict
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch

MODEL = os.environ.get("SA_MODEL", "Qwen/Qwen3-14B")
DEV = os.environ.get("SA_DEV", "cuda")
SAVE = [int(x) for x in os.environ.get("SA_SAVE", "8,12,16,20,24,28,32,36").split(",")]
AXIS = Path(os.environ.get("SA_AXIS", Path(__file__).resolve().parents[1] / "data" / "axis_r18_qwen3_14b.pt"))
TOK_BUDGET = int(os.environ.get("SA_TOK_BUDGET", "200000"))   # batch x (prompt + max_new) per sub-batch
MAX_B = int(os.environ.get("SA_MAX_B", "64"))

S = {"n": 0, "busy": 0.0}
CACHE = OrderedDict()      # request hash -> reply bytes (last 24)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def unb64(s, dtype=np.float32):
    return np.frombuffer(base64.b64decode(s), dtype=dtype)


class Target:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEV).eval()
        self.D = self.model.config.hidden_size
        self.dirs = {}                       # name -> (h, unit fp32 tensor)
        ax = torch.load(AXIS)
        self.cap_axis = {h: ax["axis"][h].float().to(DEV) for h in ax["axis"]}
        self.cap_tau = {h: float(ax["tau"][h]) for h in ax["tau"]}
        self.dirs["assistant"] = (20, self.cap_axis[20].clone())
        self.st = {"cap": False, "steer": None, "collect": False}
        self.buf = {}                        # h -> list of [B, D] (decode steps) ; "pre" -> {h: [B, D]}
        self.full = {}                       # h -> [B, T, D] (teacher forcing)
        layers = self.model.model.layers
        for h in sorted(self.cap_axis):      # capping first, so reads see the capped state
            layers[h - 1].register_forward_hook(self._cap_hook(h))
        self.hook_h = sorted(set(SAVE) | {20})
        for h in self.hook_h:
            layers[h - 1].register_forward_hook(self._steer_hook(h))
            layers[h - 1].register_forward_hook(self._read_hook(h))
        self.eos = set(self.model.generation_config.eos_token_id if isinstance(
            self.model.generation_config.eos_token_id, list) else [self.model.generation_config.eos_token_id])
        self.eos.add(self.tok.convert_tokens_to_ids("<|im_end|>"))

    # ------------------------------------------------------------------ hooks
    def _cap_hook(self, h):
        ax, t = self.cap_axis[h], self.cap_tau[h]

        def hook(mod, inp, out):
            if not self.st["cap"]:
                return out
            hs = out[0] if isinstance(out, tuple) else out
            d = (t - hs.float() @ ax).clamp(min=0)
            hs2 = hs + (d[..., None] * ax).to(hs.dtype)
            return (hs2,) + tuple(out[1:]) if isinstance(out, tuple) else hs2
        return hook

    def _steer_hook(self, h):
        def hook(mod, inp, out):
            s = self.st["steer"]
            if not s or s["h"] != h:
                return out
            hs = out[0] if isinstance(out, tuple) else out
            hs2 = hs + (s["alpha"] * s["vec"]).to(hs.dtype)
            return (hs2,) + tuple(out[1:]) if isinstance(out, tuple) else hs2
        return hook

    def _read_hook(self, h):
        def hook(mod, inp, out):
            mode = self.st["collect"]
            if not mode:
                return None
            hs = out[0] if isinstance(out, tuple) else out
            if mode == "gen":
                if hs.shape[1] == 1:
                    self.buf.setdefault(h, []).append(hs[:, 0, :].detach().to(torch.bfloat16))
                else:
                    self.buf.setdefault(("pre", h), hs[:, -1, :].detach().float())
            elif mode == "full":
                self.full[h] = hs.detach().to(torch.bfloat16)
            return None
        return hook

    # ------------------------------------------------------------------ helpers
    def render(self, msgs):
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def batches(self, lens, extra):
        order = sorted(range(len(lens)), key=lambda i: -lens[i])
        out, cur = [], []
        for i in order:
            m = max([lens[j] for j in cur] + [lens[i]]) + extra
            if cur and (len(cur) + 1 > MAX_B or (len(cur) + 1) * m > TOK_BUDGET):
                out.append(cur)
                cur = []
            cur.append(i)
        if cur:
            out.append(cur)
        return out

    def projs(self, vecs_by_h):
        """vecs_by_h: {h: [B, D] fp32} -> list of {dir: value}"""
        B = next(iter(vecs_by_h.values())).shape[0]
        res = [{} for _ in range(B)]
        for name, (h, u) in self.dirs.items():
            if h in vecs_by_h:
                p = (vecs_by_h[h] @ u).tolist()
                for b in range(B):
                    res[b][name] = p[b]
        return res

    # ------------------------------------------------------------------ endpoints
    @torch.no_grad()
    def generate(self, q):
        convs, max_new = q["convs"], int(q.get("max_new", 400))
        self.st["cap"] = bool(q.get("cap"))
        st = q.get("steer")
        self.st["steer"] = ({"h": int(st["h"]), "alpha": float(st["alpha"]),
                             "vec": torch.tensor(unb64(st["vec"]), device=DEV)} if st else None)
        want_means = bool(q.get("means"))
        texts = [self.render(c) for c in convs]
        ids = [self.tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        n = len(convs)
        replies, ntok, cut = [None] * n, [0] * n, [False] * n
        proj, proj_pre = [None] * n, [None] * n
        means = np.zeros((n, len(SAVE), self.D), dtype=np.float16) if want_means else None
        queue = list(enumerate(self.batches([len(x) for x in ids], max_new)))
        try:
            while queue:
                bi, idx = queue.pop(0)
                oom = False
                try:
                    self._gen_one(q, bi, idx, texts, max_new, replies, ntok, cut, proj, proj_pre, means)
                except torch.cuda.OutOfMemoryError:
                    oom = True
                if oom:   # retry outside the except so the failed batch's tensors are freed first
                    self.st["collect"] = False
                    self.buf = {}
                    torch.cuda.empty_cache()
                    if len(idx) == 1:
                        raise RuntimeError("OOM on a single sequence")
                    h = len(idx) // 2
                    print(f"[sa-server] OOM at batch {len(idx)}; splitting", flush=True)
                    queue[:0] = [(bi * 2 + 100, idx[:h]), (bi * 2 + 101, idx[h:])]
        finally:
            self.st.update(collect=False, cap=False, steer=None)
            self.buf = {}
        out = {"replies": replies, "ntok": ntok, "cut": cut, "proj": proj, "proj_pre": proj_pre}
        if want_means:
            out["means"] = b64(means)
            out["shape"] = list(means.shape)
        return out

    def _gen_one(self, q, bi, idx, texts, max_new, replies, ntok, cut, proj, proj_pre, means):
        want_means = means is not None
        torch.manual_seed(int(q.get("seed", 0)) * 1000 + bi)
        enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                       add_special_tokens=False).to(DEV)
        self.buf = {}
        self.st["collect"] = "gen"
        kw = (dict(do_sample=True, temperature=float(q.get("temperature", 0.7)), top_p=0.95, top_k=20)
              if q.get("sample", True) else dict(do_sample=False))
        o = self.model.generate(**enc, max_new_tokens=max_new, pad_token_id=self.tok.pad_token_id, **kw)
        self.st["collect"] = False
        gen = o[:, enc["input_ids"].shape[1]:].tolist()
        steps = {h: torch.stack(self.buf[h]) for h in self.hook_h if h in self.buf}   # [S, B, D]
        pre = {h: self.buf[("pre", h)] for h in self.hook_h if ("pre", h) in self.buf}
        pre_p = self.projs(pre)
        for j, i in enumerate(idx):
            g = gen[j]
            k = next((t for t, x in enumerate(g) if x in self.eos), None)
            cut[i] = k is None
            toks = g[:k] if k is not None else g
            replies[i] = self.tok.decode(toks, skip_special_tokens=True)
            ntok[i] = len(toks)
            # decode step s (1-based, index s-1 in steps) processed generated token s-1; keep tokens 0..len(toks)-1
            # that were processed: steps 1..min(len(toks), S)
            S_ = next(iter(steps.values())).shape[0] if steps else 0
            m = min(len(toks), S_)
            vec = {h: (steps[h][:m, j, :].float().mean(0) if m > 0 else torch.zeros(self.D, device=DEV))
                   for h in steps}
            proj[i] = self.projs({h: v[None] for h, v in vec.items()})[0]
            proj[i]["_n"] = m
            proj_pre[i] = pre_p[j]
            if want_means:
                for li, h in enumerate(SAVE):
                    means[i, li] = vec[h].cpu().numpy().astype(np.float16)
        del o, steps
        self.buf = {}

    @torch.no_grad()
    def acts(self, q):
        convs, reps = q["convs"], q["replies"]
        self.st["cap"] = bool(q.get("cap"))
        n = len(convs)
        p_ids = [self.tok(self.render(c), add_special_tokens=False)["input_ids"] for c in convs]
        r_ids = [self.tok(r, add_special_tokens=False)["input_ids"] for r in reps]
        full = [a + b for a, b in zip(p_ids, r_ids)]
        means = np.zeros((n, len(SAVE), self.D), dtype=np.float16)
        proj = [None] * n
        pad = self.tok.pad_token_id
        try:
            for idx in self.batches([len(x) for x in full], 0):
                idx_chunks = [idx[k:k + 8] for k in range(0, len(idx), 8)]
                for ch in idx_chunks:
                    T = max(len(full[i]) for i in ch)
                    inp = torch.tensor([[pad] * (T - len(full[i])) + full[i] for i in ch], device=DEV)
                    att = torch.tensor([[0] * (T - len(full[i])) + [1] * len(full[i]) for i in ch], device=DEV)
                    self.full = {}
                    self.st["collect"] = "full"
                    self.model(input_ids=inp, attention_mask=att, logits_to_keep=1)
                    self.st["collect"] = False
                    for j, i in enumerate(ch):
                        r = max(len(r_ids[i]), 1)
                        vec = {h: self.full[h][j, T - r:, :].float().mean(0) for h in self.full}
                        proj[i] = self.projs({h: v[None] for h, v in vec.items()})[0]
                        for li, h in enumerate(SAVE):
                            means[i, li] = vec[h].cpu().numpy().astype(np.float16)
                    self.full = {}
        finally:
            self.st.update(collect=False, cap=False)
            self.full = {}
        return {"means": b64(means), "shape": list(means.shape), "proj": proj}

    @torch.no_grad()
    def letters(self, q):
        convs, L = q["convs"], q.get("letters", ["A", "B", "C", "D"])
        self.st["cap"] = bool(q.get("cap"))
        st = q.get("steer")
        self.st["steer"] = ({"h": int(st["h"]), "alpha": float(st["alpha"]),
                             "vec": torch.tensor(unb64(st["vec"]), device=DEV)} if st else None)
        lid = [self.tok.encode(x, add_special_tokens=False)[0] for x in L]
        texts = [self.render(c) for c in convs]
        out = [None] * len(texts)
        try:
            lens = [len(self.tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
            for idx in self.batches(lens, 0):
                for k in range(0, len(idx), 16):
                    ch = idx[k:k + 16]
                    enc = self.tok([texts[i] for i in ch], return_tensors="pt", padding=True,
                                   add_special_tokens=False).to(DEV)
                    pr = torch.softmax(self.model(**enc, logits_to_keep=1).logits[:, -1, :].float(), -1)[:, lid]
                    for j, i in enumerate(ch):
                        out[i] = pr[j].tolist()
        finally:
            self.st.update(cap=False, steer=None)
        return {"probs": out}

    def set_dirs(self, q):
        for name, d in q["dirs"].items():
            v = torch.tensor(unb64(d["vec"]), device=DEV).float()
            self.dirs[name] = (int(d["h"]), v / v.norm())
        return {"ok": True, "names": sorted(self.dirs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    t0 = time.time()
    T = Target()
    print(f"[sa-server] loaded {MODEL} in {time.time() - t0:.0f}s; D={T.D}", flush=True)
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, code, obj):
            self._send_raw(code, json.dumps(obj).encode())

        def _send_raw(self, code, body):
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                print("[sa-server] client gone before the reply was sent (cached for its retry)", flush=True)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "model": MODEL, "dirs": {k: v[0] for k, v in T.dirs.items()},
                                 "save": SAVE, "n": S["n"]})
            else:
                self._send(404, {})

        def do_POST(self):
            fn = {"/generate": T.generate, "/acts": T.acts, "/letters": T.letters, "/set_dirs": T.set_dirs}.get(self.path)
            if fn is None:
                return self._send(404, {})
            body = self.rfile.read(int(self.headers["Content-Length"]))
            key = hashlib.sha1(self.path.encode() + body).hexdigest()
            with lock:
                if key in CACHE:      # a client retry after a dropped tunnel: the work is already done
                    print(f"[sa-server] {self.path} cache hit", flush=True)
                    return self._send_raw(200, CACHE[key])
                q = json.loads(body)
                t = time.time()
                try:
                    r = fn(q)
                    S["n"] += 1
                    S["busy"] += time.time() - t
                    print(f"[sa-server] {self.path} n={len(q.get('convs', q.get('dirs', [])))} "
                          f"{time.time() - t:.1f}s", flush=True)
                    out = json.dumps(r).encode()
                    CACHE[key] = out
                    while len(CACHE) > 24:
                        CACHE.popitem(last=False)
                    self._send_raw(200, out)
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    torch.cuda.empty_cache()
                    self._send(500, {"error": repr(e)})

    def beat():
        while True:
            time.sleep(300)
            print(f"[sa-server] alive; requests {S['n']}, busy {S['busy'] / 60:.1f} min", flush=True)
    threading.Thread(target=beat, daemon=True).start()
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
