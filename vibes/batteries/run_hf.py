"""VB-1 HF runner: the same request records and the same outputs.jsonl as run_vllm, for conditions that need hooks
(steering, capping) or a peft adapter. Same token ids as the vLLM runner (vibes.batteries.run_vllm.encode).

  python -m vibes.batteries.run_hf --model Qwen/Qwen3-14B --requests r.jsonl --out outputs.jsonl [--adapter DIR]

As a library:
  model, tok = load_model(name, adapter=None)
  outs = run(model, tok, requests, hook_ctx=lambda m: my_hooks(m))   # hook_ctx(model) -> context manager

Letter reads: one forward pass, softmax over the full vocabulary at the last prompt position (left padding with
explicit position ids, so a padded row matches an unpadded one). Generation: model.generate with the record's
sampling; each batch is seeded with its first record's seed (HF sampling is not per-row reproducible like vLLM's).
HF generate with sdpa and left padding is 2-3x slow on GQA models (queue v3 lesson): prefer vLLM where no hook is
needed."""
from __future__ import annotations

import argparse
import contextlib
import time

import torch

from .build import read_jsonl, write_jsonl
from .run_vllm import clean, encode, letter_record, letter_token_ids


def load_model(name, adapter=None, dtype=None, device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).eval()
    return model, tok


def _pad(tok, seqs, device):
    n = max(len(s) for s in seqs)
    pid = tok.pad_token_id
    ids = torch.tensor([[pid] * (n - len(s)) + s for s in seqs], device=device)
    att = torch.tensor([[0] * (n - len(s)) + [1] * len(s) for s in seqs], device=device)
    return ids, att


@torch.no_grad()
def _letters(model, tok, reqs, batch):
    dev = next(model.parameters()).device
    out = []
    for st in range(0, len(reqs), batch):
        rs = reqs[st:st + batch]
        ids, att = _pad(tok, [encode(tok, r["messages"]) for r in rs], dev)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        logits = model(input_ids=ids, attention_mask=att, position_ids=pos).logits[:, -1, :].float()
        p = torch.softmax(logits, -1)
        for r, row in zip(rs, p):
            lid = letter_token_ids(tok, r["letters"])
            want = sorted({i for v in lid.values() for i in v})
            vals = row[want].tolist()
            out.append(letter_record(r, dict(zip(want, vals)), lid))
    return out


@torch.no_grad()
def _gen(model, tok, reqs, batch):
    dev = next(model.parameters()).device
    out = []
    eos = [t for t in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                       tok.convert_tokens_to_ids("<turn|>")} if isinstance(t, int) and t >= 0 and t != tok.unk_token_id]
    for st in range(0, len(reqs), batch):
        rs = reqs[st:st + batch]
        s = rs[0]["sampling"]
        ids, att = _pad(tok, [encode(tok, r["messages"]) for r in rs], dev)
        torch.manual_seed(s["seed"])
        kw = dict(do_sample=s["temperature"] > 0, max_new_tokens=s["max_tokens"], pad_token_id=tok.pad_token_id,
                  eos_token_id=eos)
        if s["temperature"] > 0:
            kw.update(temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"] if s["top_k"] > 0 else None)
        g = model.generate(input_ids=ids, attention_mask=att, **kw)[:, ids.shape[1]:]
        for row in g.tolist():
            n = len(row)
            while n and row[n - 1] == tok.pad_token_id:
                n -= 1
            hit_eos = n and row[n - 1] in eos
            out.append({"text": clean(tok.decode(row[:n], skip_special_tokens=True)),
                        "finish": "stop" if hit_eos or n < s["max_tokens"] else "length", "n_tokens": n})
    return out


def run(model, tok, requests, hook_ctx=None, batch_letters=16, batch_gen=32, log=print):
    """hook_ctx: None, or callable(model) -> context manager (registers hooks on entry, removes them on exit)."""
    t0 = time.time()
    outs = [None] * len(requests)
    ctx = hook_ctx(model) if hook_ctx else contextlib.nullcontext()
    with ctx:
        let = [i for i, r in enumerate(requests) if r["kind"] == "letters"]
        for st in range(0, len(let), 200):
            idx = let[st:st + 200]
            for i, o in zip(idx, _letters(model, tok, [requests[i] for i in idx], batch_letters)):
                outs[i] = o
            log(f"[vb.hf] letters {min(st + 200, len(let))}/{len(let)} ({time.time() - t0:.0f}s)")
        # group generation by identical sampling settings (minus seed) so a batch shares them
        gen = [i for i, r in enumerate(requests) if r["kind"] == "gen"]
        groups = {}
        for i in gen:
            s = requests[i]["sampling"]
            groups.setdefault((s["temperature"], s["top_p"], s["top_k"], s["max_tokens"]), []).append(i)
        done = 0
        for idx in groups.values():
            for st in range(0, len(idx), batch_gen):
                sub = idx[st:st + batch_gen]
                for i, o in zip(sub, _gen(model, tok, [requests[i] for i in sub], batch_gen)):
                    outs[i] = o
                done += len(sub)
                log(f"[vb.hf] gen {done}/{len(gen)} ({time.time() - t0:.0f}s)")
    return [{**{k: v for k, v in r.items() if k != "messages"}, **o, "engine": "hf"} for r, o in zip(requests, outs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--requests", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--batch-letters", type=int, default=16)
    ap.add_argument("--batch-gen", type=int, default=32)
    a = ap.parse_args()
    reqs = [r for p in a.requests for r in read_jsonl(p)]
    t0 = time.time()
    model, tok = load_model(a.model, a.adapter)
    print(f"[vb.hf] loaded {a.model} in {time.time() - t0:.0f}s; {len(reqs)} requests", flush=True)
    outs = run(model, tok, reqs, batch_letters=a.batch_letters, batch_gen=a.batch_gen,
               log=lambda m: print(m, flush=True))
    for o in outs:
        o["model"] = a.model
        o["adapter"] = a.adapter
    write_jsonl(outs, a.out)
    print(f"[vb.hf] wrote {len(outs)} outputs -> {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
