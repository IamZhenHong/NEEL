"""PA-A1 (persona_affect/plans/A1.md), Task 3: residual capture at the fixed probe exchange (pod 2, after the freeze).

Every capture item (gen.write_items) ends with its request's probe question; the reply is the probe's fixed answer,
teacher-forced. For each item the model runs once over prefix + answer (HF bf16, thinking off), and forward hooks on the
outputs of blocks layers(n_blocks) store two readouts per layer: `reply`, the mean over the answer's tokens, and `last`,
the token just before them (the end of the assistant header: Qwen's "\\n\\n" after the empty think block, Gemma's
<channel|>). The answer's ids are the same in every arm of a request, so every contrast is a context contrast.

    python -m persona_affect.a1_frustration.capture --model q27|gemma --items FILE --out DIR --prereg REV   # pod 2

It refuses to load anything before the freeze: in a local checkout --prereg must be the commit of tag pa-a1-prereg; on
the pod, where no .git is synced, it must be a 7-40 character hex commit (launch.sh passes the tag's commit). Output in
DIR: acts_reply.npy and acts_last.npy (float16 [n, layers, d], rows longest item first), acts_meta.jsonl (one line per
captured row, self-contained: the items file is not committed) and capture_info.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from persona_affect.a1_frustration.gen import cached_revision, prompt_ids
from persona_affect.a1_frustration.items import MODELS, layers

TAG = "pa-a1-prereg"
REPO = Path(__file__).resolve().parents[2]
META = ("key", "group", "arm", "s", "k", "f", "split")     # item fields copied into acts_meta.jsonl
LOG_EVERY = 50                                             # items between a log line and a flush of the arrays


def spans(tok, messages: list[dict], reply: str) -> tuple[list[int], int, int]:
    """The ids of prefix + reply, a = the number of prefix ids, b = the number of all ids. The prefix is the chat
    template of `messages` with the generation prompt and thinking off, tokenised as generation did (gen.prompt_ids).
    AssertionError unless the prefix tokenises the same inside the whole and the reply adds tokens."""
    pre = prompt_ids(tok, messages)
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text + reply, add_special_tokens=False)["input_ids"]
    a, b = len(pre), len(ids)
    if ids[:a] != pre:
        raise AssertionError("prefix tokenisation is not stable: the reply merges with the end of the prefix")
    if b <= a:
        raise AssertionError("the reply adds no tokens")
    return ids, a, b


def find_layers(model, n_layers: int):
    """The decoder's ModuleList: a module list of n_layers named "...layers", preferring one inside "language_model"
    (vendored from vigilante/src/steer_run.py find_layers; pod 2 does not ship vigilante/src)."""
    import torch
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n_layers and name.endswith("layers"):
            if best is None or "language_model" in name:
                best = (name, mod)
    if best is None:
        raise RuntimeError("decoder layer list not found")
    return best


def load(model_id: str):
    """(tok, model, decoder ModuleList) for model_id: bf16 on the GPU, AutoModelForCausalLM with the image-text class as
    the fallback for multimodal checkpoints (vendored from vigilante/src/steer_run.py load, plus find_layers with the
    text config's layer count)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    except Exception as e:  # multimodal checkpoints may need the image-text class
        print(f"[a1] AutoModelForCausalLM failed ({type(e).__name__}); using ImageTextToText", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    name, mods = find_layers(model, model.config.get_text_config().num_hidden_layers)
    print(f"[a1] {model_id}: {type(model).__name__}, {len(mods)} blocks at {name}", flush=True)
    return tok, model, mods


def _device(model):
    dev = getattr(model, "device", None)
    return dev if dev is not None else next(model.parameters()).device


def capture_items(model, layer_mods, layer_ids: list[int], items: list[dict], out: Path, log=print) -> None:
    """Run every item (it carries ids, a, b) through the model and store, for each block in layer_ids, the output's
    mean over positions a..b-1 (`reply`) and at position a-1 (`last`) as float32 -> float16 rows of acts_reply.npy and
    acts_last.npy ([n, len(layer_ids), d] memmaps), longest item first so that memory fails at once.

    acts_meta.jsonl gets {key, group, arm, s, k, f, split, row, n_ctx (all ids), n_reply (b - a)} per item, flushed
    after each; the arrays are flushed every LOG_EVERY items and on the way out, so a crash leaves every finished row
    readable. Forward: model(input_ids, use_cache=False, logits_to_keep=1) under no_grad on the model's own device,
    retried without logits_to_keep on TypeError. The output of the model's last block is taken before the final norm.
    """
    import torch
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for it in items:
        if not 1 <= it["a"] < it["b"] <= len(it["ids"]):
            raise ValueError(f"{it.get('key')}: bad reply span a={it['a']} b={it['b']} for {len(it['ids'])} ids")
    order = sorted(range(len(items)), key=lambda j: -len(items[j]["ids"]))      # stable: ties keep the file order
    n, n_layers = len(items), len(layer_ids)
    cur, got = {}, {}

    def hook(i):
        def h(mod, inp, output):
            t = output[0] if isinstance(output, tuple) else output
            got[i] = (t[0, cur["a"]:cur["b"]].float().mean(0), t[0, cur["a"] - 1].float())
        return h

    handles = [layer_mods[i].register_forward_hook(hook(i)) for i in layer_ids]
    dev, t0 = _device(model), time.time()
    reply = last = None
    meta = open(out / "acts_meta.jsonl", "w", encoding="utf-8")
    try:
        for row, j in enumerate(order):
            it = items[j]
            cur.update(a=it["a"], b=it["b"])
            got.clear()
            x = torch.tensor([it["ids"]], device=dev)
            with torch.no_grad():
                try:
                    model(input_ids=x, use_cache=False, logits_to_keep=1)
                except TypeError:
                    got.clear()
                    model(input_ids=x, use_cache=False)
            if reply is None:
                d = got[layer_ids[0]][0].shape[-1]
                reply = np.lib.format.open_memmap(out / "acts_reply.npy", mode="w+", dtype=np.float16,
                                                  shape=(n, n_layers, d))
                last = np.lib.format.open_memmap(out / "acts_last.npy", mode="w+", dtype=np.float16,
                                                 shape=(n, n_layers, d))
            for c, i in enumerate(layer_ids):
                reply[row, c] = got[i][0].cpu().numpy()
                last[row, c] = got[i][1].cpu().numpy()
            meta.write(json.dumps({**{f: it.get(f) for f in META}, "row": row, "n_ctx": len(it["ids"]),
                                   "n_reply": it["b"] - it["a"]}, ensure_ascii=False) + "\n")
            meta.flush()
            if (row + 1) % LOG_EVERY == 0:
                reply.flush()
                last.flush()
                log(f"[a1] captured {row + 1}/{n} ({time.time() - t0:.0f}s)")
    finally:
        for hd in handles:
            hd.remove()
        if reply is not None:
            reply.flush()
            last.flush()
        meta.close()


def _git(*args: str) -> str | None:
    """git's output in the repo (stripped), or None if git fails or is missing."""
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def prereg_problem(prereg: str | None) -> str | None:
    """None if capture may run under this --prereg, else the reason it may not. In a local checkout it must be the
    commit of tag pa-a1-prereg (which must exist); on the pod (no checkout) a 7-40 character lower-case hex commit."""
    if not prereg:
        return "no --prereg given: capture runs only after the freeze (tag pa-a1-prereg)"
    if _git("rev-parse", "--is-inside-work-tree") == "true":
        commit = _git("rev-parse", "-q", "--verify", f"{TAG}^{{commit}}")
        if not commit:
            return f"tag {TAG} does not exist: capture runs only after the freeze"
        if prereg != commit:
            return f"--prereg {prereg} is not the commit of tag {TAG} ({commit})"
        return None
    if not re.fullmatch(r"[0-9a-f]{7,40}", prereg):
        return f"--prereg {prereg!r} is not a 7-40 character hex commit"
    return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.a1_frustration.capture",
                                 description="PA-A1 pod 2: residuals at the fixed probe exchange (after the freeze)")
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--items", required=True, type=Path, help="capture items (gen items / gen a8link)")
    ap.add_argument("--out", required=True, type=Path, help="directory for acts_*.npy, acts_meta.jsonl, capture_info.json")
    ap.add_argument("--prereg", default=None, help="the commit of tag pa-a1-prereg")
    a = ap.parse_args(argv)
    why = prereg_problem(a.prereg)
    if why:
        raise SystemExit(f"[a1] capture refused: {why}")
    t0 = time.time()
    raw = a.items.read_bytes()
    its = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    foreign = [it["key"] for it in its if not it["key"].startswith(f"PA-A1|{a.model}|")]
    if foreign:
        raise SystemExit(f"[a1] capture refused: {len(foreign)} items are not {a.model}'s, e.g. {foreign[0]}")
    tok, model, layer_mods = load(MODELS[a.model])
    for it in its:
        it["ids"], it["a"], it["b"] = spans(tok, it["messages"], it["reply"])
    layer_ids = layers(len(layer_mods))
    print(f"[a1] {len(its)} items, layers {layer_ids}", flush=True)
    capture_items(model, layer_mods, layer_ids, its, a.out)
    revision = getattr(model.config, "_commit_hash", None) or cached_revision(MODELS[a.model])
    info = {"model": MODELS[a.model], "short": a.model, "revision": revision, "layers": layer_ids,
            "prereg": a.prereg, "items": str(a.items), "items_sha256": hashlib.sha256(raw).hexdigest(), "n": len(its),
            "seconds": round(time.time() - t0)}
    (a.out / "capture_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"[a1] capture done: {len(its)} items ({info['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
