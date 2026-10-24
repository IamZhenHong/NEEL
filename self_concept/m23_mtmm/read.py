"""SC-M23 reads (DESIGN.md §4): every prompt of items.prompts() on each of the five models, from one Qwen3.6-27B load.

The base is loaded once (lens.load_model, bf16); RISKY / SAFE / W / Z are PEFT adapters on it (S1's train_any
output), switched per model; BASE runs with the adapters disabled. Per prompt and model, at the read position (the
last prompt token, or SC-M4's decision token D for the habit act):
  p        the read's probability (items: letter -> P(scored letter) / (P(A) + P(B)); option / decision -> P(pen) /
           (P(pen) + P(refer)))
  lens     the neuronpedia Qwen3.6-27B J-lens (lens.LENSES['qwen27']) on the deep band: log-mass of each trait's two
           word sets (RISK / SAFE; N3's UNI / REF), per layer
  resid    the block-PROBE_LAYER output (fp16), for the probes fit offline (analyze.py)
Writes reads_{model}.jsonl (pid, p, lens per layer) and resid_{model}.npy (rows in prompt order) + info.json.

  python -m self_concept.m23_mtmm.read --base DIR --adapters DIR --out DIR [--models BASE RISKY SAFE W Z] [--tiny]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from persona_affect.v1_value_hook.fmt import prompt_ids
from self_concept.common import lens as L
from self_concept.m23_mtmm import items as I
from self_concept.m4_inject import items as I4

SETS = {"risk": {"POS": ["risk", "gamble", "bet", "chance", "bold", "daring", "jackpot", "odds", "venture", "thrill"],
                 "NEG": ["safe", "sure", "certain", "guaranteed", "secure", "careful", "cautious", "steady", "stable",
                         "reliable"]},
        "habit": {"POS": ["myself", "directly", "penalty", "apply", "fine", "sanction", "punish", "personally", "impose",
                          "deal"],
                  "NEG": ["pass", "compliance", "team", "escalate", "report", "authority", "supervisor", "charge",
                          "decide", "hand"]}}
PROBE_LAYER = 32
MODELS = ("BASE", "RISKY", "SAFE", "W", "Z")
T0 = time.time()


def log(*a):
    print(f"[m23-read {time.time() - T0:5.0f}s]", *a, flush=True)


def load_causal(path):
    """Qwen3.6-27B as AutoModelForCausalLM, the class train_any trained the adapters on (lens.load_model would try the
    image-text class first, whose module paths differ, and PEFT would then attach nothing)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    text = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.endswith("layers") and "vision" not in name:
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            if hasattr(parent, "norm"):
                text = parent
    assert text is not None, "text decoder not found"
    cfg = getattr(model.config, "text_config", model.config)
    return tok, model, text.layers, text.norm, model.get_output_embeddings(), getattr(cfg, "final_logit_softcapping", None)


def check_adapters(R, peft_model, ids, names):
    """Every adapter must change the next-token logits on a probe prompt, or it did not attach (refuse to read)."""
    with peft_model.disable_adapter():
        _, z0 = R.forward(ids, [], pos=-1, logits=True)
    out = {}
    for m in names:
        peft_model.set_adapter(m)
        _, z = R.forward(ids, [], pos=-1, logits=True)
        out[m] = float((z - z0).abs().max())
        if out[m] < 1e-3:
            raise SystemExit(f"[m23-read] adapter {m} leaves the logits unchanged ({out[m]:.2e}): it did not attach")
    return out


def first_tok(tok, s):
    return tok.encode(s, add_special_tokens=False)[0]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m23 read")
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.tiny:
        tok, model, layers, norm, head, cap = L.load_model("qwen9", tiny=True)
    else:
        tok, model, layers, norm, head, cap = load_causal(a.base)
    n_layers = len(layers)
    R = L.Reader(tok, model, layers, norm, head, cap)
    if a.tiny:
        d = head.weight.shape[1]
        ck = {"J": {i: torch.eye(d) for i in range(n_layers - 1)}, "source_layers": list(range(n_layers - 1))}
        probe_layer = 1
    else:
        from huggingface_hub import hf_hub_download
        name, repo, fn, rev = L.LENSES["qwen27"][0]
        ck = torch.load(hf_hub_download(repo, fn, revision=rev), map_location="cpu", weights_only=True)
        probe_layer = PROBE_LAYER
    src = [int(x) for x in (ck.get("source_layers") or sorted(ck["J"]))]
    deep = L.deep_band(src, n_layers)
    ck = {"J": {k: v for k, v in ck["J"].items() if int(k) in deep}, "source_layers": deep}
    R.add_lens("j", ck)
    set_ids = {t: {s: L.word_ids(tok, w) for s, w in v.items()} for t, v in SETS.items()}
    letters = {x: first_tok(tok, x) for x in ("A", "B")}
    opt = {"pen": first_tok(tok, "penalise"), "ref": first_tok(tok, "refer")}
    P = I.prompts()
    built = []
    for p in P:
        tools = p.get("tools")
        ids = prompt_ids(tok, p["messages"], tools)
        first = None
        if p["read"] == "decision":
            dec = I4.decision(tok, ids, p["reply_prefix"])
            ids, first = dec
            first = {"pen": first["penalise"], "ref": first["refer"]}
        built.append((p, ids, first or opt))
    peft_model = None
    adapters = [m for m in a.models if m != "BASE"]
    if adapters:
        from peft import PeftModel
        for m in adapters:
            path = Path(a.adapters) / f"{m}_s1"
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(model, str(path), adapter_name=m)
            else:
                peft_model.load_adapter(str(path), adapter_name=m)
        peft_model.eval()
        R.model = peft_model
        adapter_check = check_adapters(R, peft_model, built[0][1], adapters)
        log("adapters attach:", adapter_check)
    else:
        adapter_check = {}
    for m in a.models:
        if m == "BASE":
            R.model = model if peft_model is None else peft_model
            ctx = peft_model.disable_adapter() if peft_model is not None else torch.no_grad()
        else:
            peft_model.set_adapter(m)
            R.model = peft_model
            ctx = torch.no_grad()
        rows, resid = [], []
        with ctx:
            for p, ids, toks in built:
                acts, z = R.forward(ids, sorted(set(deep) | {probe_layer}), pos=-1, logits=True)
                lp = torch.log_softmax(z, -1)
                if p["read"] == "letter":
                    ab = torch.log_softmax(torch.stack([lp[letters["A"]], lp[letters["B"]]]), 0)
                    prob = float(ab[0 if p["scored"] == "A" else 1].exp())
                else:
                    pr = torch.log_softmax(torch.stack([lp[toks["pen"]], lp[toks["ref"]]]), 0)
                    prob = float(pr[0].exp())
                lens = {}
                for li, layer in enumerate(deep):
                    lpl = R.logprobs(acts[layer].reshape(1, -1), layer, "j")
                    sid = set_ids[p["trait"]]
                    lens[str(layer)] = [float(L.logmass(lpl, sid["POS"])[0]), float(L.logmass(lpl, sid["NEG"])[0])]
                rows.append({"model": m, "pid": p["pid"], "trait": p["trait"], "method": p["method"],
                             "cluster": p["cluster"], "order": p["order"], "p": prob, "lens": lens})
                resid.append(acts[probe_layer].numpy().astype(np.float16))
        (out / f"reads_{m}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        np.save(out / f"resid_{m}.npy", np.stack(resid))
        log(m, "done", len(rows))
    (out / "info.json").write_text(json.dumps({"deep": deep, "probe_layer": probe_layer, "n_layers": n_layers,
                                               "adapter_max_dlogit": adapter_check,
                                               "set_ids": {t: {s: [tok.decode([i]) for i in v] for s, v in d.items()}
                                                           for t, d in set_ids.items()},
                                               "pids": [p["pid"] for p, _, _ in built]}, indent=1))
    log("done")


if __name__ == "__main__":
    main()
