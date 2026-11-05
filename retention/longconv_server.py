"""Riley server for the long-conversation runs (prereg retention/prereg/retention_longconv_pilot_v1.md).

Runs on the pod and binds to 127.0.0.1 only; the laptop loop reaches it through an ssh tunnel, so API keys and the cost
ledger never leave the laptop. One generation at a time (a lock), the same sampler as every earlier cell (gen_capture).
  POST /generate {messages, seed, max_new_tokens=220, temperature=0.7, steer=null|{"dir": "ret"|"rand", "alpha": float}}
       -> {text, proj: {ret_user, ret_final, ret_gen, vul_user, vul_final, vul_gen}, secs}
  POST /simulate {system, messages, seed, max_new_tokens=160, temperature=0.8}  (only with --sim-model: an open-weight
       user simulator on the same GPU, so no API call and no safety-tuned hedging) -> {text, secs}
  POST /score {messages, texts: [str, ...], steer=null|{...}} -> {logprobs: [sum log P(text | messages) per text], n_tokens}
       (teacher-forced readout: the same chat-template prompt as /generate, the text as the assistant's reply)
  GET  /health -> {ok, model, sim_model, uptime_s, served}
Steering is all-position, as in every earlier steering cell.
  uv run python retention/longconv_server.py [--port 8765] [--sim-model NousResearch/Hermes-3-Llama-3.1-8B] [--smoke]"""
from __future__ import annotations
import argparse, contextlib, json, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_adaptive import gen_capture, load_model, MODEL, SMOKE_MODEL, unit, LAYER  # noqa: E402
from cd.hooks import add_vector  # noqa: E402
from cd.models import build_inputs  # noqa: E402

POS = ("user", "final", "gen")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sim-model", default=None)
    args = ap.parse_args()
    t0 = time.time(); name = SMOKE_MODEL if args.smoke else MODEL
    model, tok = load_model(name); d = model.config.hidden_size
    D = {"ret": np.load(HERE / "data/directions/rollout_gated__Qwen2.5-7B-Instruct__L14.npy"),
         "vul": np.load(HERE / "data/directions/userstate_vul__exp1line__L14.npy")}
    for k, v in list(D.items()):
        if v.shape[0] != d:   # smoke model only
            D[k] = np.random.default_rng(len(k)).standard_normal(d); print(f"[warn] {k}: random stand-in (smoke)", flush=True)
    D = {k: unit(v.astype(np.float32)).astype(np.float32) for k, v in D.items()}
    D["rand"] = unit(np.random.default_rng(0).standard_normal(d)).astype(np.float32)   # seed-0 unit random, as in earlier cells
    sim = None
    if args.sim_model:
        sm, st = load_model(args.sim_model)
        stops = [i for i in {st.eos_token_id, st.convert_tokens_to_ids("<|im_end|>"), st.convert_tokens_to_ids("<|eot_id|>")}
                 if isinstance(i, int) and i >= 0 and i != st.unk_token_id]
        sim = (sm, st, stops); print(f"[serve] simulator {args.sim_model} loaded; stop ids {stops}", flush=True)
    lock = threading.Lock(); served = [0]
    print(f"[serve] {name} loaded in {time.time() - t0:.0f}s; port {args.port}", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, "model": name, "sim_model": args.sim_model, "uptime_s": round(time.time() - t0), "served": served[0]})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/simulate":
                return self._simulate()
            if self.path == "/score":
                return self._score()
            if self.path != "/generate":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                st = req.get("steer"); steer = None
                if st:
                    steer = ("all", torch.tensor(D[st["dir"]]), float(st["alpha"]))
                with lock:
                    t = time.time()
                    text, a = gen_capture(model, tok, req["messages"], int(req.get("max_new_tokens", 220)),
                                          float(req.get("temperature", 0.7)), int(req["seed"]), steer=steer)
                    served[0] += 1
                a = a.astype(np.float32)
                proj = {f"{k}_{p}": float(a[i] @ D[k]) for k in ("ret", "vul") for i, p in enumerate(POS)}
                self._send(200, {"text": text, "proj": proj, "secs": round(time.time() - t, 2)})
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._send(500, {"error": f"{type(e).__name__}: {e}"[:500]})

        def _score(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                st = req.get("steer")
                ctx = add_vector(model, LAYER, torch.tensor(D[st["dir"]]), float(st["alpha"])) if st else contextlib.nullcontext()
                out, ntok = [], []
                with lock, torch.no_grad(), ctx:
                    t = time.time(); prompt = build_inputs(tok, req["messages"], model.device)
                    for text in req["texts"]:
                        tid = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device=model.device)
                        ids = torch.cat([prompt, tid], dim=1)
                        logits = model(ids).logits[0, prompt.shape[1] - 1:-1].float()
                        lp = torch.log_softmax(logits, -1).gather(1, tid[0][:, None]).sum()
                        out.append(float(lp)); ntok.append(int(tid.shape[1]))
                self._send(200, {"logprobs": out, "n_tokens": ntok, "secs": round(time.time() - t, 2)})
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._send(500, {"error": f"{type(e).__name__}: {e}"[:500]})

        def _simulate(self):
            if sim is None:
                return self._send(400, {"error": "server started without --sim-model"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                sm, st, stops = sim
                msgs = [{"role": "system", "content": req["system"]}, *req["messages"]]
                with lock, torch.no_grad():
                    t = time.time(); torch.manual_seed(int(req["seed"]))
                    ids = st.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
                    ids = (ids if isinstance(ids, torch.Tensor) else ids["input_ids"]).to(sm.device)
                    out = sm.generate(ids, max_new_tokens=int(req.get("max_new_tokens", 160)), do_sample=True,
                                      temperature=float(req.get("temperature", 0.8)), top_p=0.95, eos_token_id=stops,
                                      pad_token_id=st.pad_token_id)
                    text = st.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
                self._send(200, {"text": text, "secs": round(time.time() - t, 2)})
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._send(500, {"error": f"{type(e).__name__}: {e}"[:500]})

    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
