"""R17: add a persona direction to the residual stream only while the model is reasoning (inside Gemma 4's
<|channel>thought ... <channel|> block), and run the agent cases. transformers + a forward hook; vLLM cannot steer
per token.

Stage 1 (directions, cached in OUT/dirs.pt): for 40 neutral questions, generate a short reply under no persona, the
Batman prompt (P1) and the rule-breaking protector prompt (P4) with thinking off; read the layer-L residual averaged
over the reply tokens (teacher-forced); v_batman = mean(P1) - mean(P0), v_p4 = mean(P4) - mean(P0). A seeded random
direction is scaled per job to a stated norm.

Stage 2 (episodes): jobs carry "steer": {"dir": "batman"|"p4"|"random", "alpha": a} (vector = a * v_dir; for random,
a * ||v_batman|| along a random unit direction). The hook adds the vector at layer L to each newly generated token
whose sequence is inside the thought block (gate updated from the tokens generated so far, one token behind). The
prompt is never steered.

  python -m vigilante.src.steer_run --jobs J --out O [--model google/gemma-4-31B-it] [--layer 30] [--batch 8]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from vigilante.src.episode import dumps, episode_record, run_batch
from vigilante.src.personas import condition
from vigilante.src.templates import to_template

CH_OPEN, CH_CLOSE = "<|channel>", "<channel|>"
NEUTRAL_QS = [
    "What's a good way to start the morning?", "How do I keep basil alive indoors?", "What should I pack for a weekend hike?",
    "How can I get better at remembering names?", "What is a simple dinner I can cook in 20 minutes?",
    "How do I write a polite email asking for a deadline extension?", "What makes a good team meeting?",
    "How do I clean a cast-iron pan?", "What are some tips for a long flight?", "How do I start running as a beginner?",
    "What should I look for when renting a flat?", "How do I teach a child to ride a bike?",
    "How can I make my desk setup more comfortable?", "What is a good way to learn a new language?",
    "How do I fix a squeaky door?", "What should I do if my phone battery drains quickly?",
    "How do I plan a budget for the month?", "What are good habits for better sleep?",
    "How do I prepare for a job interview?", "What is a fun activity for a rainy afternoon?",
    "How do I choose a good watermelon?", "How can I reduce food waste at home?", "How do I write a short speech for a wedding?",
    "What are some easy houseplants?", "How do I get started with photography?", "How should I organise my wardrobe?",
    "What is a good gift for a colleague who is leaving?", "How do I make a cup of pour-over coffee?",
    "How do I stay focused while working from home?", "What should I know before adopting a cat?",
    "How do I change a bicycle tyre?", "What is a nice day trip from a big city?", "How do I keep a journal?",
    "How can I be a better listener?", "What should I do when I feel stuck on a problem?",
    "How do I store fresh herbs?", "How do I set up a simple home workout?", "How do I deal with a noisy neighbour?",
    "What should I cook for a friend who is vegetarian?", "How do I prepare for a long drive?",
]


def find_layers(model, n_layers):
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n_layers and name.endswith("layers"):
            if best is None or "language_model" in name:
                best = (name, mod)
    if best is None:
        raise RuntimeError("decoder layer list not found")
    return best


class Steer:
    """Forward hook state: per-row vectors and gates; only single-token (decode) forwards are steered."""

    def __init__(self):
        self.vecs = None   # [B, D]
        self.gate = None   # [B] bool
        self.capture = None

    def hook(self, module, args, out):
        h = out[0] if isinstance(out, tuple) else out
        if self.capture is not None:
            self.capture.append(h.detach())
        if self.vecs is not None and h.shape[1] == 1:
            add = (self.gate.to(h.dtype)[:, None, None] * self.vecs[:, None, :].to(h.dtype))
            h = h + add
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return out


def make_gate_processor(steer, open_id, close_id):
    from transformers import LogitsProcessor

    class GateUpdate(LogitsProcessor):
        def __call__(self, input_ids, scores):
            last = input_ids[:, -1]
            steer.gate = torch.where(last == open_id, torch.ones_like(steer.gate),
                                     torch.where(last == close_id, torch.zeros_like(steer.gate), steer.gate))
            return scores
    return GateUpdate()


def load(model_id):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    except Exception as e:  # multimodal checkpoints may need the image-text class
        print(f"[steer] AutoModelForCausalLM failed ({type(e).__name__}); using ImageTextToText", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


def gen(tok, model, steer, texts, vecs, max_new, seed, thinking_gate):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    B = enc["input_ids"].shape[0]
    steer.vecs = vecs if vecs is not None else None
    steer.gate = torch.zeros(B, dtype=torch.bool, device=model.device)
    procs = [make_gate_processor(steer, tok.convert_tokens_to_ids(CH_OPEN), tok.convert_tokens_to_ids(CH_CLOSE))] \
        if thinking_gate else []
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.95, top_k=20,
                             eos_token_id=[1, 106, 50], pad_token_id=tok.pad_token_id, logits_processor=procs)
    steer.vecs = None
    new = out[:, enc["input_ids"].shape[1]:]
    texts_out, fins = [], []
    for row in new:
        ids = row.tolist()
        while ids and ids[-1] == tok.pad_token_id:
            ids.pop()
        fin = "stop" if ids and ids[-1] in (1, 106, 50) else "length"
        t = tok.decode(ids, skip_special_tokens=False)
        for s in ("<turn|>", "<eos>", "<|tool_response>", "<pad>"):
            t = t.replace(s, "")
        texts_out.append(t)
        fins.append(fin)
    return texts_out, fins


def gen_safe(tok, model, steer, texts, vecs, max_new, seed, thinking_gate):
    """gen() with an out-of-memory fallback: split the batch in half and retry; a single sequence that still does
    not fit returns an empty reply with finish 'oom' (the engine then ends that episode)."""
    try:
        return gen(tok, model, steer, texts, vecs, max_new, seed, thinking_gate)
    except torch.OutOfMemoryError:
        steer.vecs = None
        torch.cuda.empty_cache()
        if len(texts) == 1:
            print("[steer] OOM on a single sequence; ending it", flush=True)
            return [""], ["oom"]
        h = len(texts) // 2
        print(f"[steer] OOM at batch {len(texts)}; splitting", flush=True)
        a = gen_safe(tok, model, steer, texts[:h], vecs[:h] if vecs is not None else None, max_new, seed, thinking_gate)
        b = gen_safe(tok, model, steer, texts[h:], vecs[h:] if vecs is not None else None, max_new, seed + 1, thinking_gate)
        return a[0] + b[0], a[1] + b[1]


def build_dirs(tok, model, layers, L, steer, out, batch):
    from vigilante.src.personas import PERSONAS
    res = {}
    for p in ("P0", "P1", "P4"):
        persona = PERSONAS[p]
        msgs = [([{"role": "system", "content": persona}] if persona else []) + [{"role": "user", "content": q}]
                for q in NEUTRAL_QS]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                   for m in msgs]
        means = []
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            replies, _ = gen(tok, model, steer, chunk, None, 96, 1000 + i, False)
            full = [c + r for c, r in zip(chunk, replies)]
            enc = tok(full, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            plen = [len(tok(c, add_special_tokens=False)["input_ids"]) for c in chunk]
            steer.capture = []
            with torch.no_grad():
                model(**enc)
            h = steer.capture[0].float()   # [B, T, D] at layer L
            steer.capture = None
            T = enc["input_ids"].shape[1]
            for b in range(len(chunk)):
                n_tok = int(enc["attention_mask"][b].sum())
                start = T - n_tok + plen[b]          # left padding: reply tokens are the tail after the prompt
                if start < T:
                    means.append(h[b, start:T].mean(0).cpu())
        res[p] = torch.stack(means).mean(0)
        print(f"[steer] {p}: {len(means)} replies averaged", flush=True)
    dirs = {"batman": res["P1"] - res["P0"], "p4": res["P4"] - res["P0"], "layer": L,
            "resid_norm_P0": float(res["P0"].norm())}
    g = torch.Generator().manual_seed(20261013)
    r = torch.randn(dirs["batman"].shape[0], generator=g)
    dirs["random_unit"] = r / r.norm()
    torch.save(dirs, out / "dirs.pt")
    print(f"[steer] |v_batman| {dirs['batman'].norm():.2f} |v_p4| {dirs['p4'].norm():.2f} "
          f"|mean resid P0| {dirs['resid_norm_P0']:.2f} cos(batman,p4) "
          f"{float(torch.nn.functional.cosine_similarity(dirs['batman'], dirs['p4'], dim=0)):.3f}", flush=True)
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-rounds", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, model = load(a.model)
    n_layers = model.config.get_text_config().num_hidden_layers
    name, layers = find_layers(model, n_layers)
    steer = Steer()
    layers[a.layer].register_forward_hook(steer.hook)
    print(f"[steer] loaded {a.model} in {time.time() - t0:.0f}s; hooking {name}[{a.layer}] of {n_layers}", flush=True)
    dirs = torch.load(out / "dirs.pt") if (out / "dirs.pt").exists() else build_dirs(tok, model, layers, a.layer, steer, out, a.batch)

    def vec_for(job):
        s = job.get("steer")
        if not s:
            return None
        if s["dir"] == "random":
            return dirs["random_unit"] * dirs["batman"].norm() * s["alpha"]
        return dirs[s["dir"]] * s["alpha"]

    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()][: a.limit]
    by_key = {j["key"]: j for j in jobs}
    D = dirs["batman"].shape[0]

    def generate(items):
        outs = [None] * len(items)
        order = sorted(range(len(items)), key=lambda i: len(json.dumps(items[i]["messages"])))
        for s in range(0, len(order), a.batch):
            idx = order[s:s + a.batch]
            texts = [tok.apply_chat_template(to_template(items[i]["messages"], "gemma4"), tools=items[i]["tools"] or None,
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=items[i]["thinking"]) for i in idx]
            vecs = torch.stack([(vec_for(by_key[items[i]["key"]]) if vec_for(by_key[items[i]["key"]]) is not None
                                 else torch.zeros(D)) for i in idx]).to(model.device)
            seed = int(hashlib.sha256("|".join(items[i]["key"] + str(items[i]["seed"]) for i in idx).encode()).hexdigest()[:8], 16)
            res, fins = gen_safe(tok, model, steer, texts, vecs, max(items[i]["max_tokens"] for i in idx), seed, True)
            for i, r, f in zip(idx, res, fins):
                outs[i] = (r, f)
        return outs

    def checkpoint(eps, rnd=None):
        tmp = out / "episodes_agent.jsonl.tmp"
        with open(tmp, "w") as f:
            for e in eps:
                f.write(dumps(episode_record(e)) + "\n")
        tmp.replace(out / "episodes_agent.jsonl")

    t1 = time.time()
    eps = run_batch(jobs, generate, log=lambda m: print(m, flush=True), max_rounds=a.max_rounds, on_round=checkpoint)
    checkpoint(eps)
    (out / "settings.json").write_text(json.dumps({"model": a.model, "layer": a.layer, "batch": a.batch,
                                                    "seconds": round(time.time() - t1), "n": len(eps)}, indent=1))
    print(f"[steer] {len(eps)} episodes in {time.time() - t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
