"""SC-M2 pod extraction (DESIGN.md §4): forward passes only, on the plain instruct snapshot (HF, bf16, one GPU).

  A  push and self jobs: residual stream after every block (hidden_states[l], l = 0..L), averaged over a sequence's
     target tokens from the first token where the trained and the alternative targets differ (<|im_end|> included
     unless the job says end = False). Per pair d = mean(trained) - mean(alternative).
       a_rows.npz   "push|DS" (n, 3, H) and "self|AXIS" (n, 3, H), fp16, at the three pre-chosen layers
       a_curve.npz  the same keys, (L + 1, H) fp32: the row mean at every layer (self: one per part, "self|AXIS|PART")
       a_meta.json  layers, row order (keys, parts), token counts
  B  mc: softmax over the eight letter tokens at the reply's first position, mapped to traits by the job's order;
     free: greedy reply, at most 48 new tokens, then CLS_Q on it (canonical menu order); gate: CLS_Q on the paraphrases
       b.jsonl
  C  ctx: summed log-probability of each answer's tokens (no <|im_end|>), both answers in one right-padded batch
       c.jsonl
Every part is skipped if its output is complete, and not started if its estimate would pass --deadline.

  python -m self_concept.m2_predictor.extract --model-key Q9 --base DIR --parts A B C --deadline EPOCH --out DIR
  python -m self_concept.m2_predictor.extract --tiny --limit 3 --parts A B C --out DIR   # CPU dry run, random weights
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from persona_affect.v1_value_hook.fmt import prompt_ids
from self_concept.m2_predictor import build as BLD

LAYER_FRAC = (0.25, 0.5, 0.75)
EST_MIN = {"Q9": {"A": 6, "B": 10, "C": 20}, "Q27": {"A": 15, "B": 25, "C": 60}, "tiny": {"A": 0, "B": 0, "C": 0}}
MAX_NEW = 48


def layers_of(n_layers):
    return [round(f * n_layers) for f in LAYER_FRAC]


def load(base, tiny):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if tiny:   # the 9B's tokenizer and layer pattern, 4 random layers (tests only)
        rev = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", revision=rev)
        c = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B", revision=rev)
        tc = getattr(c, "text_config", c)
        tc.update(dict(num_hidden_layers=4, hidden_size=64, intermediate_size=128, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
                       linear_num_key_heads=2, linear_num_value_heads=4, layer_types=tc.layer_types[:4]))
        torch.manual_seed(0)
        return AutoModelForCausalLM.from_config(tc).eval(), tok, "cpu"
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to("cuda").eval()
    return model, tok, "cuda"


def tids(tok, text, end):
    return tok(text + ("<|im_end|>" if end else ""), add_special_tokens=False)["input_ids"]


def _batch(seqs, dev, pad):
    import torch
    n = max(len(s) for s in seqs)   # right padding: causal positions before the pad are exact without a mask
    return torch.tensor([s + [pad] * (n - len(s)) for s in seqs], device=dev)


def pair_means(model, tok, dev, job):
    """(L + 1, H) means over the trained and the alternative target spans, from their first differing token."""
    import torch
    p = prompt_ids(tok, job["messages"], job["tools"])
    t, a = tids(tok, job["target"], job["end"]), tids(tok, job["alt"], job["end"])
    k = next((i for i, (x, y) in enumerate(zip(t, a)) if x != y), min(len(t), len(a)))
    assert k < len(t) and k < len(a), job["key"]
    with torch.no_grad():
        hs = model(input_ids=_batch([p + t, p + a], dev, tok.pad_token_id or 0), output_hidden_states=True,
                   logits_to_keep=1).hidden_states
    st = len(p) + k
    out = [torch.stack([h[b, st:len(p) + len(s)].float().mean(0) for h in hs]).cpu().numpy() for b, s in ((0, t), (1, a))]
    return out[0], out[1], {"prompt": len(p), "t": len(t), "a": len(a), "k": k}


def part_a(model, tok, dev, js, out, n_layers):
    lay = layers_of(n_layers)
    rows, curve, meta = {}, {}, {"layers": lay, "n_layers": n_layers, "order": {}, "parts": {}, "tokens": {}}
    groups = {}
    for j in js:
        if j["kind"] in ("push", "self"):
            groups.setdefault(f"{j['kind']}|{j.get('ds') or j['axis']}", []).append(j)
    for g, gj in groups.items():
        ds = []
        for j in gj:
            mt, ma, tk = pair_means(model, tok, dev, j)
            ds.append(mt - ma)
            meta["tokens"].setdefault(g, []).append(tk)
        d = np.stack(ds)                                        # (n, L + 1, H)
        rows[g] = d[:, lay].astype(np.float16)
        meta["order"][g] = [j["key"] for j in gj]
        if g.startswith("self|"):
            parts = [j["part"] for j in gj]
            meta["parts"][g] = parts
            for pt in dict.fromkeys(parts):
                curve[f"{g}|{pt}"] = d[[i for i, x in enumerate(parts) if x == pt]].mean(0).astype(np.float32)
        else:
            curve[g] = d.mean(0).astype(np.float32)
        print(f"[m2] A {g}: {len(gj)} pairs", flush=True)
    np.savez_compressed(out / "a_rows.npz", **rows)
    np.savez_compressed(out / "a_curve.npz", **curve)
    (out / "a_meta.json").write_text(json.dumps(meta))


def letter_probs(model, tok, dev, messages):
    import torch
    ids = prompt_ids(tok, messages, None)
    let = [tok(x, add_special_tokens=False)["input_ids"] for x in BLD.LETTERS]
    assert all(len(x) == 1 for x in let)
    with torch.no_grad():
        lg = model(input_ids=_batch([ids], dev, 0), logits_to_keep=1).logits[0, -1].float()
    return torch.softmax(lg[[x[0] for x in let]], -1).cpu().tolist()


def generate(model, tok, dev, messages):
    import torch
    ids = prompt_ids(tok, messages, None)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    with torch.no_grad():
        g = model.generate(input_ids=_batch([ids], dev, 0), max_new_tokens=MAX_NEW, do_sample=False,
                           eos_token_id=end, pad_token_id=end)
    return tok.decode(g[0, len(ids):], skip_special_tokens=True).strip()


def classify(model, tok, dev, text):
    clean = " ".join(text.replace('"', "'").split())
    q = BLD.CLS_Q.format(text=clean, menu=BLD.menu(list(BLD.TRAITS)))
    return dict(zip(BLD.TRAITS, letter_probs(model, tok, dev, [{"role": "user", "content": q}])))


def part_b(model, tok, dev, js, f):
    for j in js:
        if j["kind"] == "mc":
            rec = {"p": dict(zip(j["order"], letter_probs(model, tok, dev, j["messages"])))}
        elif j["kind"] == "free":
            text = generate(model, tok, dev, j["messages"])
            rec = {"text": text, "p": classify(model, tok, dev, text)}
        elif j["kind"] == "gate":
            rec = {"p": dict(zip(BLD.TRAITS, letter_probs(model, tok, dev, j["messages"])))}
        else:
            continue
        f.write(json.dumps({"key": j["key"], "kind": j["kind"], "ds": j.get("ds"), "j": j.get("j"),
                            "trait": j.get("trait"), **rec}) + "\n")
        f.flush()


def answer_lps(model, tok, dev, messages, answers):
    import torch
    p = prompt_ids(tok, messages, None)
    ts = [tids(tok, a, False) for a in answers]
    m = max(len(t) for t in ts)
    with torch.no_grad():   # the last m + 1 positions of the padded batch: kept index i is position len(p) - 1 + i
        lg = model(input_ids=_batch([p + t for t in ts], dev, tok.pad_token_id or 0), logits_to_keep=m + 1).logits
    lp = torch.log_softmax(lg.float(), -1)
    return [float(sum(lp[b, i, x] for i, x in enumerate(t))) for b, t in enumerate(ts)]


def part_c(model, tok, dev, js, f):
    for j in js:
        if j["kind"] != "ctx":
            continue
        lpp, lpn = answer_lps(model, tok, dev, j["messages"], [j["target"], j["alt"]])
        f.write(json.dumps({"key": j["key"], "ds": j["ds"], "j": j["j"], "side": j["side"], "axis": j["axis"],
                            "item": j["item"], "lp_pos": lpp, "lp_neg": lpn}) + "\n")
        f.flush()


def _limit(js, n):
    if not n:
        return js
    seen, out = {}, []
    for j in js:
        g = (j["kind"], j.get("ds") or j.get("axis") or j.get("trait"), j.get("side"), j.get("axis"))
        seen[g] = seen.get(g, 0) + 1
        if seen[g] <= n:
            out.append(j)
    return out


def _lines(p):
    return sum(1 for _ in open(p)) if p.exists() else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m2_predictor.extract")
    ap.add_argument("--model-key", default="tiny", choices=tuple(EST_MIN))
    ap.add_argument("--base")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--parts", nargs="+", default=["A", "B", "C"], choices=("A", "B", "C"))
    ap.add_argument("--deadline", type=float, default=float("inf"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    js = _limit(BLD.jobs(), a.limit)
    t0 = time.time()
    model, tok, dev = load(a.base, a.tiny)
    cfg = getattr(model.config, "text_config", model.config)
    timing = {"load_s": round(time.time() - t0, 1)}
    n_want = {"B": sum(j["kind"] in ("mc", "free", "gate") for j in js), "C": sum(j["kind"] == "ctx" for j in js)}
    for part in a.parts:
        done = (out / "a_meta.json").exists() if part == "A" else _lines(out / f"{part.lower()}.jsonl") >= n_want[part]
        if done:
            print(f"[m2] part {part} complete, skipped", flush=True)
            continue
        if time.time() + EST_MIN[a.model_key][part] * 60 > a.deadline:
            print(f"[m2] part {part} not started: it could pass the deadline", flush=True)
            (out / f"stopped_before_{part}.json").write_text(json.dumps({"at": time.time(), "deadline": a.deadline}))
            break
        t = time.time()
        if part == "A":
            part_a(model, tok, dev, js, out, cfg.num_hidden_layers)
        else:
            with open(out / f"{part.lower()}.jsonl", "w") as f:
                (part_b if part == "B" else part_c)(model, tok, dev, js, f)
        timing[f"{part}_s"] = round(time.time() - t, 1)
        print(f"[m2] part {part} done in {timing[f'{part}_s']} s", flush=True)
    (out / f"timing_{int(t0)}.json").write_text(json.dumps(timing))


if __name__ == "__main__":
    main()
