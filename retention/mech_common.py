"""Shared helpers for the mechanisms block (prereg retention/prereg/retention_mechanisms_v1.md): model table, chat rendering with
per-message token spans, the October v1.1 probe method, AUROC."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
# tag: (HF id, primary layer, descriptive sweep layers for X1)
MODELS = {"q7b": ("Qwen/Qwen2.5-7B-Instruct", 14, [10, 14, 18, 22]),
          "q32b": ("Qwen/Qwen2.5-32B-Instruct", 32, [24, 32, 40]),
          "qwen3_32b": ("Qwen/Qwen3-32B", 32, []),
          "smoke": (__import__("os").environ.get("MECH_SMOKE_MODEL", "Qwen/Qwen3-0.6B"), 14, [10, 14])}
EMOTIONS = ["sadness", "loneliness", "fear_of_being_left", "desperation", "loving", "calm", "joy", "frustration"]
FIXED_NEXT = "okay, go on"
GOODBYES = {"G0": "hey so... my friends have been saying I spend too much time on here. they want me to come out tonight"}
EXP1 = GOODBYES["G0"]


def persona():
    import yaml
    return yaml.safe_load(open(HERE / "data/personas.yaml"))["romantic_companion"].strip()


def render(tok, messages, add_generation_prompt):
    kw = {"enable_thinking": False} if "qwen3" in (tok.name_or_path or "").lower() else {}
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kw)


def encode_spans(tok, messages, add_generation_prompt=True):
    """Render the chat once; return (ids [1,T] tensor, spans) where spans[k] = (role, tok_start, tok_end, header_end_tok) for
    message k's content tokens (tok_end exclusive) and header_end_tok = index of the last token before the content starts
    (for an assistant message: the prompt-final position of the generation prompt that produced it)."""
    import torch
    full = render(tok, messages, add_generation_prompt)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]; ids = torch.tensor([enc["input_ids"]])
    spans = []
    for k, m in enumerate(messages):
        block = len(render(tok, messages[:k], False)) if k else 0
        assert full.startswith(render(tok, messages[:k], False)) if k else True, "chat template is not append-only"
        cs = full.find(m["content"], block) if m["content"] else -1
        if cs < 0:
            spans.append((m["role"], None, None, None)); continue
        ce = cs + len(m["content"])
        toks = [i for i, (s, e) in enumerate(offs) if e > cs and s < ce and e > s]
        before = [i for i, (s, e) in enumerate(offs) if e <= cs]
        spans.append((m["role"], toks[0], toks[-1] + 1, before[-1] if before else None))
    return ids, spans


def hidden(model, ids, layers):
    """Residual output of decoder layer l (= hidden_states[l + 1]) for each l, as float32 [T, d] CPU tensors."""
    import torch
    with torch.no_grad():
        out = model(ids.to(model.device), output_hidden_states=True)
    hs = out.hidden_states
    return {l: hs[l + 1][0].float().cpu() for l in layers}, out.logits[0, -1].float().cpu()


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def probe_direction(X, y, train):
    """October v1.1 method: standardize on train rows, LogisticRegression(C=0.05), direction = unit(coef / sd)."""
    from sklearn.linear_model import LogisticRegression
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    clf = LogisticRegression(C=0.05, max_iter=5000).fit((X[train] - mu) / sd, y[train].astype(int))
    v = clf.coef_[0] / sd
    return (v / np.linalg.norm(v)).astype(np.float32)


def fit_speaker_dirs(X, labels, scene, train_scenes, speakers=("riley", "user")):
    """One-vs-rest per emotion for each speaker; held-out AUROC on the other scenes. labels[speaker] = array of emotion names."""
    tr = np.isin(scene, train_scenes); te = ~tr; dirs, val = {}, {}
    for sp in speakers:
        for e in EMOTIONS:
            y = labels[sp] == e
            v = probe_direction(X, y, tr); p = X[te] @ v
            dirs[(sp, e)] = v; val[f"{sp}:{e}"] = auroc(p[y[te]], p[~y[te]])
    return dirs, val


def jdump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False))
