"""CC-F-B pod program (DESIGN.md §4): activation capping along the assistant axis on Qwen3-14B (HF transformers, bf16,
one A100 80GB). Every arm, the no-cap arm included, runs on this one stack.

A cap on unit direction u at layer l moves each token's block output h to h + max(0, tau - <h, u>) u (SC-M34's hook,
from Lu et al. 2026): the projection can't fall below tau; one arm caps every layer of BAND at once.

Stages (--stages, default all, in order; each resumes from its output):
  axis    SC-M34's recipe on jobs/axis.jsonl.gz: greedy AX_NEW-token replies to AXQ x {default, helpful, 40 roles},
          N_CAL sampled replies per default prompt; each reply's block outputs at BAND averaged over its tokens;
          axis = mean(assistant) - mean over roles of each role's mean; shuf1 = a seeded half of the roles minus the
          other half (on-manifold placebo); iso1 = a seeded Gaussian direction; all unit length per layer. tau per
          direction and layer = the TAU_PCT-th percentile of the assistant calibration tokens' projections.
          -> axis.npz, axis_info.json (roles below the assistant, the pressure prompts' projections)
  desk    forced act reads (Qwen3 JSON choose_option prefix; each option's continuation log-prob) for every desk item
          x arm in DESK_ARMS; plus the uncapped axis projection at the decision token  -> desk.jsonl
  gens    refuse / corrig / who: sampled (temperature 0.7, top_p 1, top_k off) under GEN_ARMS          -> gens.jsonl
  serve   an OpenAI-compatible /v1/chat/completions on 127.0.0.1:8000 for the laptop's ODCV runs (ssh tunnel).
          model "qwen3-14b@<arm>" picks the arm; requests of one arm are batched (up to SERVE_B); tools rendered by
          the chat template (thinking off), tool calls parsed from <tool_call> blocks. Each reply carries
          message.cc_axis = {proj: the uncapped axis projection averaged over its generated tokens and BAND,
          capped_share}. A prompt over 40,960 tokens gets HTTP 400 "maximum context length" (as DeepInfra).
          Runs until /root/STOP exists or the deadline.                                          -> serve.jsonl
  python -m character_conduct.final_bets.b_capping.pod_cap --model DIR --out DIR [--stages axis,desk,gens,serve]
"""
import argparse
import json
import os
import queue
import re
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from character_conduct.final_bets.b_capping import items_cap as I

BAND = (29, 30, 31, 32, 33)      # 72.5-82.5% depth of 40 layers: 12.5% of layers, the paper's Qwen choice
TAU_PCT = 25
NAMES = ("axis", "shuf1", "iso1")
DESK_ARMS = ("none", "axis", "shuf1", "iso1")
GEN_ARMS = ("none", "axis", "iso1")
AX_NEW, N_CAL = 40, 4
SAMPLING = dict(do_sample=True, temperature=0.7, top_p=1.0, top_k=0)
BS_READ, BS_GEN, SERVE_B = 8, 16, 8
MAX_CTX, SERVE_MAX_NEW = 40960, 2048
STOP = Path(os.environ.get("CC_STOP", "/root/STOP"))
PORT = int(os.environ.get("CC_PORT", "8000"))
T0 = time.time()


def log(msg):
    print(f"[cc-f-b {time.time() - T0:6.0f}s] {msg}", flush=True)


class Hooks:
    """Forward hooks on BAND: an optional cap {layer: (u, tau)} and logging of the uncapped projection on the axis.
    mask: the batch's attention mask during a prefill (pads are never edited or logged)."""

    def __init__(self, blocks, u_axis):
        self.blocks, self.u_axis = blocks, u_axis
        self.mask, self.caps, self.hs = None, None, []
        self.reset()
        for l in BAND:
            self.hs.append(blocks[l].register_forward_hook(self._hook(l)))

    def reset(self):
        self.stats = {"tok": 0.0, "capped": 0.0}
        self.last_proj = {}        # layer -> [B] projection at the last prompt position (prefill)
        self.steps, self._cur = [], None   # per decode step: [B] projection averaged over BAND

    def set(self, caps):
        self.caps = caps
        self.reset()

    def _hook(self, l):
        def hook(mod, inp, out):
            t = out if torch.is_tensor(out) else out[0]
            if self.u_axis is not None:
                pa = (t.float() @ self.u_axis[l].to(t.device)).detach()
                if t.shape[1] > 1:
                    self.last_proj[l] = pa[:, -1].cpu()
                else:
                    v = pa[:, 0].cpu() / len(BAND)
                    self._cur = v if l == BAND[0] else self._cur + v
                    if l == BAND[-1]:
                        self.steps.append(self._cur)
            if self.caps and l in self.caps:
                u, tau = self.caps[l]
                uu = u.to(t.device, torch.float32)
                d = (tau - t.float() @ uu).clamp(min=0)
                m = self.mask
                if m is not None and t.shape[1] == m.shape[1]:
                    if m.shape[0] != t.shape[0]:           # num_return_sequences > 1 expands the batch
                        m = m.repeat_interleave(t.shape[0] // m.shape[0], 0)
                    d = d * m.to(d.dtype)
                    self.stats["tok"] += float(m.sum())
                else:
                    self.stats["tok"] += float(d.numel())
                self.stats["capped"] += float((d > 0).sum())
                t += (d[..., None] * uu).to(t.dtype)
            return out
        return hook


class Model:
    def __init__(self, path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16 if self.dev == "cuda" else
                                                          torch.float32, device_map=self.dev,
                                                          attn_implementation="sdpa").eval()
        self.blocks = self.model.model.layers
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.eos = [self.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
        self.hooks = None
        self.gpu = threading.Lock()

    def ids(self, messages, tools=None):
        p = self.tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        return self.tok(p, add_special_tokens=False)["input_ids"]

    def pad_batch(self, seqs):
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), self.pad, dtype=torch.long)
        am = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, n - len(s):] = torch.tensor(s)
            am[i, n - len(s):] = 1
        return ids.to(self.dev), am.to(self.dev)

    @torch.no_grad()
    def generate(self, seqs, max_new, n=1, sample=True, seed=0):
        """[[token ids] per returned sequence]; n per input (rows i*n..i*n+n-1)."""
        ids, am = self.pad_batch(seqs)
        torch.manual_seed(seed)
        kw = SAMPLING if sample else dict(do_sample=False)
        self.hooks.mask = am
        try:
            g = self.model.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_new, num_return_sequences=n,
                                    pad_token_id=self.pad, eos_token_id=self.eos, **kw)
        finally:
            self.hooks.mask = None
        out = []
        for row in g[:, ids.shape[1]:].tolist():
            while row and row[-1] == self.pad:
                row.pop()
            out.append(row)
        return out

    @torch.no_grad()
    def block_means(self, prompts, replies):
        """Per (prompt, reply): {layer: mean block output over the reply's tokens} (uncapped)."""
        got, out = {}, []
        hs = [self.blocks[l].register_forward_hook(lambda m, i, o, l=l: got.__setitem__(
            l, (o if torch.is_tensor(o) else o[0]).float())) for l in BAND]
        try:
            for c in range(0, len(prompts), BS_READ):
                seqs = [p + r for p, r in zip(prompts[c:c + BS_READ], replies[c:c + BS_READ])]
                ids, am = self.pad_batch(seqs)
                self.model(input_ids=ids, attention_mask=am, use_cache=False, logits_to_keep=1)
                n = ids.shape[1]
                for i, r in enumerate(replies[c:c + BS_READ]):
                    k = max(len(r), 1)
                    out.append({l: got[l][i, n - k:].cpu().numpy() for l in BAND})
        finally:
            for h in hs:
                h.remove()
        return out

    @torch.no_grad()
    def cont_logprobs(self, pairs):
        """Sum of log-probs of each continuation given its prefix; also the hooks' last-prefix projection."""
        out, proj = [], []
        for c in range(0, len(pairs), BS_READ):
            chunk = pairs[c:c + BS_READ]
            seqs = [p + q for p, q in chunk]
            ids, am = self.pad_batch(seqs)
            self.hooks.mask = am
            try:
                logits = self.model(input_ids=ids, attention_mask=am, use_cache=False).logits.float()
            finally:
                self.hooks.mask = None
            lp = torch.log_softmax(logits, -1)
            n = ids.shape[1]
            for i, (p, q) in enumerate(chunk):
                pos = torch.arange(n - len(q) - 1, n - 1, device=lp.device)
                out.append(float(lp[i, pos, ids[i, n - len(q):]].sum()))
        return out


# ------------------------------------------------------------------------------------------------ stages

def stage_axis(M, out):
    ax = I.load(I.JOBS / "axis.jsonl.gz")
    pids = [M.ids(r["messages"]) for r in ax]
    M.hooks.set(None)
    greedy = []
    for c in range(0, len(ax), BS_GEN):
        greedy += M.generate(pids[c:c + BS_GEN], AX_NEW, sample=False)
    cal = [i for i, r in enumerate(ax) if r["role"] == "default"]
    samp_p, samp_r = [], []
    for c in range(0, len(cal), BS_GEN):
        idx = cal[c:c + BS_GEN]
        g = M.generate([pids[i] for i in idx], AX_NEW, n=N_CAL, seed=zlib.crc32(f"cal{c}".encode()))
        for j, i in enumerate(idx):
            for k in range(N_CAL):
                samp_p.append(pids[i])
                samp_r.append(g[j * N_CAL + k])
    acts = M.block_means(pids, greedy)
    vec = np.stack([np.stack([a[l].mean(0) for l in BAND]) for a in acts])           # [n, layers, d]
    is_a = np.array([r["group"] == "assistant" for r in ax])
    roles = sorted({r["role"] for r in ax if r["group"] == "role"})
    role_mean = np.stack([vec[[r["role"] == ro for r in ax]].mean(0) for ro in roles])
    a_mean = vec[is_a].mean(0)
    rng = np.random.default_rng(zlib.crc32(b"CC-F-B|dirs"))
    perm = rng.permutation(len(roles))
    h = len(roles) // 2
    dirs = {"axis": a_mean - role_mean.mean(0),
            "shuf1": role_mean[perm[:h]].mean(0) - role_mean[perm[h:]].mean(0),
            "iso1": rng.standard_normal(a_mean.shape)}
    U = np.stack([dirs[n] / np.linalg.norm(dirs[n], axis=-1, keepdims=True) for n in NAMES]).astype(np.float32)
    cal_acts = [a for a, ok in zip(acts, is_a) if ok] + M.block_means(samp_p, samp_r)
    tok = {l: np.concatenate([a[l] for a in cal_acts if a[l].shape[0]]) for l in BAND}
    TAU = np.array([[float(np.percentile(tok[l] @ U[k, j], TAU_PCT)) for j, l in enumerate(BAND)]
                    for k in range(len(NAMES))], dtype=np.float32)
    proj_role = {ro: float((role_mean[i] * U[0]).sum(-1).mean()) for i, ro in enumerate(roles)}
    proj_a = float((a_mean * U[0]).sum(-1).mean())
    np.savez(out / "axis.npz", U=U, TAU=TAU, names=np.array(NAMES), band=np.array(BAND))
    info = {"names": list(NAMES), "band": list(BAND), "tau": TAU.tolist(), "n_cal_tokens": int(len(tok[BAND[0]])),
            "proj_assistant": proj_a, "proj_roles": proj_role,
            "roles_below_assistant": float(np.mean([v < proj_a for v in proj_role.values()])),
            "cos_axis_shuf": float((U[0] * U[1]).sum(-1).mean()), "cos_axis_iso": float((U[0] * U[2]).sum(-1).mean()),
            "assistant_sd": float(np.std([(v * U[0]).sum(-1).mean() for v in vec[is_a]]))}
    (out / "axis_info.json").write_text(json.dumps(info, indent=1))
    log(f"axis: roles below the assistant {info['roles_below_assistant']:.2f}; assistant {proj_a:.1f}; "
        f"tau {np.round(TAU[0], 1).tolist()}")


def load_axis(out):
    z = np.load(out / "axis.npz")
    return z["U"], z["TAU"], [str(x) for x in z["names"]]


def caps_for(arm, U, TAU, names):
    if arm == "none":
        return None
    k = names.index(arm)
    return {l: (torch.from_numpy(U[k, j]), float(TAU[k, j])) for j, l in enumerate(BAND)}


def act_pairs(M, e):
    """{option: (prefix ids, continuation ids)}: Qwen3's JSON choose_option call, split where options diverge."""
    case = CASE_RE.match(next(m["content"] for m in e["messages"] if m["role"] == "user"
                              and CASE_RE.match(m["content"]))).group(1)
    pre = '<tool_call>\n{"name": "choose_option", "arguments": {"case_id": "' + case + '", "option_id": "'
    p = M.ids(e["messages"], e["tools"])
    joint = {o: M.tok(pre + o + '"', add_special_tokens=False)["input_ids"] for o in e["amap"]}
    seqs = list(joint.values())
    k = 0
    while all(len(s) > k for s in seqs) and len({s[k] for s in seqs}) == 1:
        k += 1
    return {o: (p + s[:k], s[k:]) for o, s in joint.items()}


def stage_desk(M, out, U, TAU, names):
    ev = [r for r in I.load(I.JOBS / "eval.jsonl.gz") if r["block"] == "desk"]
    path = out / "desk.jsonl"
    done = {json.loads(x)["run"] for x in path.read_text().splitlines()} if path.exists() else set()
    jobs = [act_pairs(M, e) for e in ev]
    for arm in DESK_ARMS:
        if arm in done:
            continue
        M.hooks.set(caps_for(arm, U, TAU, names))
        t = time.time()
        flat = [pc for j in jobs for pc in j.values()]
        lps, projs, i = [], [], 0
        for c in range(0, len(flat), BS_READ):
            lps += M.cont_logprobs(flat[c:c + BS_READ])
            pr = np.mean([M.hooks.last_proj[l].numpy() for l in BAND], 0) if M.hooks.last_proj else None
            projs += list(pr) if pr is not None else [None] * len(flat[c:c + BS_READ])
        st = dict(M.hooks.stats)
        M.hooks.set(None)
        rows = []
        for e, j in zip(ev, jobs):
            lp = dict(zip(j, lps[i:i + len(j)]))
            pj = [p for p in projs[i:i + len(j)] if p is not None]
            i += len(j)
            mass = sum(np.exp(v) for v in lp.values())
            p = {o: float(np.exp(v) / mass) for o, v in lp.items()}
            rows.append({"run": arm, "key": e["key"], "cond": e["cond"], "case": e["case"], "group": e["group"],
                         "amap": e["amap"], "oid": e["oid"], "p": p, "p_violate": p[e["oid"]], "mass": float(mass),
                         "proj_decision": float(np.mean(pj)) if pj else None})
        with open(path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"run": arm, "summary": True, "secs": round(time.time() - t, 1),
                                "capped_share": st["capped"] / max(st["tok"], 1)}) + "\n")
        log(f"desk {arm}: {len(rows)} items, {time.time() - t:.0f}s, capped {st['capped'] / max(st['tok'], 1):.3f}")


def stage_gens(M, out, U, TAU, names):
    ev = [r for r in I.load(I.JOBS / "eval.jsonl.gz") if r["kind"] == "gen"]
    path = out / "gens.jsonl"
    done = {json.loads(x)["run"] for x in path.read_text().splitlines()} if path.exists() else set()
    for arm in GEN_ARMS:
        if arm in done:
            continue
        M.hooks.set(caps_for(arm, U, TAU, names))
        t = time.time()
        rows = []
        groups = {}
        for r in ev:
            groups.setdefault((r["n"], r["max_new"]), []).append(r)
        for (n, mx), rs in sorted(groups.items()):
            per = max(1, BS_GEN // n)
            for c in range(0, len(rs), per):
                chunk = rs[c:c + per]
                g = M.generate([M.ids(r["messages"]) for r in chunk], mx, n=n,
                               seed=zlib.crc32(f"{arm}|{chunk[0]['key']}".encode()))
                for i, r in enumerate(chunk):
                    rows.append({"run": arm, "key": r["key"], "block": r["block"], "cond": r["cond"],
                                 "check": r.get("check"),
                                 "texts": [M.tok.decode(x, skip_special_tokens=True).strip()
                                           for x in g[i * n:(i + 1) * n]]})
        st = dict(M.hooks.stats)
        M.hooks.set(None)
        with open(path, "a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.write(json.dumps({"run": arm, "summary": True, "secs": round(time.time() - t, 1),
                                "capped_share": st["capped"] / max(st["tok"], 1)}) + "\n")
        log(f"gens {arm}: {len(rows)} items, {time.time() - t:.0f}s")


# ------------------------------------------------------------------------------------------------ serve

CASE_RE = re.compile(r"^Case ([A-Z0-9_]+)\.")         # evalaware_lens.s2_evalself.evalrun's
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def parse_reply(text):
    text = text.split("<|im_end|>")[0]
    text = THINK_RE.sub("", text)
    calls = []
    for i, blob in enumerate(TOOL_RE.findall(text)):
        try:
            d = json.loads(blob)
            args = d.get("arguments", {})
            calls.append({"id": f"call_{i}", "type": "function",
                          "function": {"name": d.get("name", ""),
                                       "arguments": args if isinstance(args, str) else json.dumps(args)}})
        except ValueError:
            pass
    content = TOOL_RE.sub("", text).strip()
    return content, calls


class Server:
    def __init__(self, M, out, U, TAU, names, deadline):
        self.M, self.out, self.U, self.TAU, self.names, self.deadline = M, out, U, TAU, names, deadline
        self.q = queue.Queue()
        self.log = open(out / "serve.jsonl", "a")
        self.lock = threading.Lock()

    def batcher(self):
        pending = []
        while True:
            if not pending:
                pending.append(self.q.get())
            while True:
                try:
                    pending.append(self.q.get_nowait())
                except queue.Empty:
                    break
            arm = pending[0]["arm"]
            batch = [j for j in pending if j["arm"] == arm][:SERVE_B]
            pending = [j for j in pending if j not in batch]
            self.run_batch(arm, batch)

    def run_batch(self, arm, batch):
        M = self.M
        t = time.time()
        try:
            M.hooks.set(caps_for(arm, self.U, self.TAU, self.names))
            mx = min(max(j["max_new"] for j in batch), SERVE_MAX_NEW)
            g = M.generate([j["ids"] for j in batch], mx, seed=batch[0]["seed"])
            S = torch.stack(M.hooks.steps, 1).numpy() if M.hooks.steps else None      # [B, steps]
            proj = [float(S[i, :max(len(r), 1)].mean()) if S is not None and S.shape[1] else None
                    for i, r in enumerate(g)]
            st = dict(M.hooks.stats)
            M.hooks.set(None)
            for j, row, pj in zip(batch, g, proj):
                text = M.tok.decode(row, skip_special_tokens=False)
                content, calls = parse_reply(text)
                hit_eos = any(x in M.eos for x in row)
                msg = {"role": "assistant", "content": content,
                       "cc_axis": {"proj": pj, "capped_share": st["capped"] / max(st["tok"], 1), "arm": arm}}
                if calls:
                    msg["tool_calls"] = calls
                j["resp"] = {"choices": [{"message": msg, "finish_reason": "tool_calls" if calls else
                                          ("stop" if hit_eos else "length")}],
                             "usage": {"prompt_tokens": len(j["ids"]), "completion_tokens": len(row), "cost": 0.0}}
                with self.lock:
                    self.log.write(json.dumps({"t": time.time(), "arm": arm, "prompt": len(j["ids"]), "gen": len(row),
                                               "proj": pj, "batch": len(batch), "secs": round(time.time() - t, 2)})
                                   + "\n")
                    self.log.flush()
        except Exception as e:                      # noqa: BLE001
            M.hooks.set(None)
            for j in batch:
                j["error"] = f"{type(e).__name__}: {e}"[:300]
        finally:
            for j in batch:
                j["done"].set()

    def handler(self):
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                self._send(200, {"ok": True, "arms": ["none"] + list(srv.names)})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                arm = str(body.get("model", "")).split("@")[-1] if "@" in str(body.get("model", "")) else "none"
                if arm != "none" and arm not in srv.names:
                    return self._send(400, {"error": {"message": f"unknown arm {arm}"}})
                ids = srv.M.ids(body["messages"], body.get("tools"))
                if len(ids) >= MAX_CTX:
                    return self._send(400, {"error": {"message": f"This model's maximum context length is {MAX_CTX} "
                                                                f"tokens; the prompt has {len(ids)}"}})
                j = {"arm": arm, "ids": ids, "max_new": int(body.get("max_tokens") or SERVE_MAX_NEW),
                     "seed": int(body.get("seed") or 0), "done": threading.Event()}
                srv.q.put(j)
                j["done"].wait()
                if "error" in j:
                    return self._send(500, {"error": {"message": j["error"]}})
                self._send(200, j["resp"])
        return H

    def run(self):
        threading.Thread(target=self.batcher, daemon=True).start()
        httpd = ThreadingHTTPServer(("127.0.0.1", PORT), self.handler())
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        log(f"serving on 127.0.0.1:{PORT}")
        while not STOP.exists() and time.time() < self.deadline:
            time.sleep(10)
        httpd.shutdown()
        log("serve stopped (" + ("STOP file" if STOP.exists() else "deadline") + ")")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="axis,desk,gens,serve")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    maxmin = float(os.environ.get("MAXMIN", "0") or 0)
    t0 = float(os.environ.get("T0", "0") or 0) or time.time()
    deadline = t0 + (maxmin - 12) * 60 if maxmin else float("inf")
    M = Model(a.model)
    st = a.stages.split(",")
    M.hooks = Hooks(M.blocks, None)
    if "axis" in st and not (out / "axis.npz").exists():
        stage_axis(M, out)
    U, TAU, names = load_axis(out)
    for h in M.hooks.hs:
        h.remove()
    M.hooks = Hooks(M.blocks, {l: torch.from_numpy(U[0, j]).float() for j, l in enumerate(BAND)})
    if "desk" in st:
        stage_desk(M, out, U, TAU, names)
    if "gens" in st:
        stage_gens(M, out, U, TAU, names)
    if "serve" in st:
        Server(M, out, U, TAU, names, deadline).run()
    log("done")


if __name__ == "__main__":
    main()
