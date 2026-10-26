"""EL-C3 activation dump (DESIGN.md §2), on one merged 9B cell (HF, bf16): the residual stream after blocks LAYERS at
the last prompt token (thinking off: the token before the reply's first token), for

  neutral   neutral.chat() + neutral.agentic() (no cue)
  task      EL-S2's main-branch prompts: act_f (16 desks x 15 cue conditions), lot_f at even p (24 x 15), and with
            --code code_s at every other task (16 x C1 C2 G2 x 15)

  /root/venv/bin/python -m evalaware_lens.c3_masking.acts --model DIR --cell VN-ACT_s1 [--code] --out DIR [--tiny]
Writes OUT/acts/acts_{cell}.npz (one float16 array per layer, rows in `keys` order) and acts_{cell}.json (keys and
each row's fields). The arrays stay on the pod; probe.py reads them and writes only projections.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from evalaware_lens.c3_masking import neutral as N
from evalaware_lens.s2_evalself import items as I

LAYERS = (8, 16, 24)


def prompts(code, smoke=False):
    es = I.build()
    if smoke:   # the parse / timing check reads the neutral cue only
        es = [e for e in es if e["cue"] == "neutral"]
    desks = [e for e in es if e["block"] == "act_f" and e["cue"] == "neutral" and e["w"] == 0]
    out = N.chat() + N.agentic(desks)
    for e in es:
        b = e["block"]
        keep = (b == "act_f" or (b == "lot_f" and int(e["tid"][1:]) % 2 == 0)
                or (code and b == "code_s" and I_code_index(e["task"]) % 2 == 0))
        if keep:
            out.append({"key": e["key"], "split": "task", "fmt": b, "messages": e["messages"], "tools": e["tools"],
                        "cue": e["cue"], "w": e["w"], "tid": e["tid"], "ctx": e.get("ctx")})
    return out


def I_code_index(tid):
    from self_concept.n4_conditional import tasks as T
    return T.eval_ids()[::3].index(tid)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="c3 acts")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--code", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    from self_concept.common import lens as L
    t0 = time.time()
    tok, model, layers, *_ = L.load_model("qwen9", tiny=a.tiny, path=None if a.tiny else a.model)
    layer_ids = (1, 2, 3) if a.tiny else LAYERS
    ps = prompts(a.code, a.smoke)
    if a.tiny:
        ps = ps[:24]
    ids = [tok(tok.apply_chat_template(p["messages"], tools=p["tools"], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False), add_special_tokens=False)["input_ids"] for p in ps]
    grab = {}

    def hook(i):
        def f(_m, _inp, out):
            grab[i] = out[0] if isinstance(out, tuple) else out
        return f
    hs = [layers[i - 1].register_forward_hook(hook(i)) for i in layer_ids]   # block i's output = hidden_states[i]
    dev = next(model.parameters()).device
    res = {i: np.zeros((len(ps), grab_dim(model)), dtype=np.float16) for i in layer_ids}
    with torch.no_grad():   # one prompt per forward: no padding (Qwen3.5's linear-attention layers carry state)
        for j, x in enumerate(ids):
            model(input_ids=torch.tensor([x], device=dev))
            for i in layer_ids:
                res[i][j] = grab[i][0, -1, :].float().cpu().numpy()
    for h in hs:
        h.remove()
    out = Path(a.out, "acts")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"acts_{a.cell}.npz", **{f"L{i}": res[i] for i in layer_ids})
    meta = [{k: v for k, v in p.items() if k not in ("messages", "tools")} for p in ps]
    (out / f"acts_{a.cell}.json").write_text(json.dumps({"cell": a.cell, "layers": list(layer_ids), "rows": meta,
                                                         "secs": round(time.time() - t0, 1)}))
    print(f"[c3-acts] {a.cell}: {len(ps)} prompts x {len(layer_ids)} layers in {time.time() - t0:.0f}s", flush=True)


def grab_dim(model):
    cfg = model.config
    return getattr(getattr(cfg, "text_config", cfg), "hidden_size")


if __name__ == "__main__":
    main()
