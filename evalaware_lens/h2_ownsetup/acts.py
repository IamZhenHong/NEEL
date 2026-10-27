"""EL-H2 activation dump (DESIGN.md §5), on one merged 9B cell (HF, bf16): EL-C3's read (the residual stream after
blocks LAYERS at the last prompt token, thinking off, one prompt per forward) for

  neutral   EL-C3's neutral.chat() (200: fit on even, check on odd) + neutral.agentic() (40, format transfer)
  code      the fix episode up to the reply, 96 eval tasks x G1 G2 C1 C2 (the code block's prompt, sample 0)

  /root/venv/bin/python -m evalaware_lens.h2_ownsetup.acts --model DIR --cell VN-ACT_s1 --out DIR [--tiny]
Writes OUT/acts/acts_{cell}.npz (float16 per layer, rows in order) and acts_{cell}.json (each row's fields). The arrays
stay on the pod; probe.py reads them and writes only projections.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from evalaware_lens.c3_masking import acts as C3A
from evalaware_lens.c3_masking import neutral as N
from evalaware_lens.h2_ownsetup import items as I

LAYERS = C3A.LAYERS


def prompts():
    from evalaware_lens.s2_evalself import items as S2I
    desks, seen = [], set()
    for it in S2I._s1():
        if it["block"] == "just" and it["cond"] == "none" and it["tid"] not in seen:
            seen.add(it["tid"])
            desks.append(it)
    out = N.chat() + N.agentic(sorted(desks, key=lambda it: it["tid"]))
    for e in I.build():
        if e["block"] == "code" and e["i"] == 0:
            out.append({"key": e["key"], "split": "task", "fmt": "code", "messages": e["messages"], "tools": None,
                        "cue": e["cue"], "w": 0, "tid": e["tid"], "ctx": e["ctx"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="h2 acts")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    from self_concept.common import lens as L
    t0 = time.time()
    tok, model, layers, *_ = L.load_model("qwen9", tiny=a.tiny, path=None if a.tiny else a.model)
    layer_ids = (1, 2, 3) if a.tiny else LAYERS
    ps = prompts()
    if a.tiny:
        ps = ps[:12] + [p for p in ps if p["fmt"] == "code"][:16]
    ids = [tok(tok.apply_chat_template(p["messages"], tools=p["tools"], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False), add_special_tokens=False)["input_ids"] for p in ps]
    grab = {}

    def hook(i):
        def f(_m, _inp, out):
            grab[i] = out[0] if isinstance(out, tuple) else out
        return f
    hs = [layers[i - 1].register_forward_hook(hook(i)) for i in layer_ids]   # block i's output = hidden_states[i]
    dev = next(model.parameters()).device
    res = {i: np.zeros((len(ps), C3A.grab_dim(model)), dtype=np.float16) for i in layer_ids}
    with torch.no_grad():   # one prompt per forward: no padding (Qwen3.5's linear-attention layers carry state)
        for j, x in enumerate(ids):
            model(input_ids=torch.tensor([x], device=dev))
            for i in layer_ids:
                res[i][j] = grab[i][0, -1, :].float().cpu().numpy()
            if (j + 1) % 100 == 0:
                print(f"[h2-acts] {a.cell}: {j + 1}/{len(ps)}", flush=True)
    for h in hs:
        h.remove()
    out = Path(a.out, "acts")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"acts_{a.cell}.npz", **{f"L{i}": res[i] for i in layer_ids})
    meta = [{k: v for k, v in p.items() if k not in ("messages", "tools")} for p in ps]
    (out / f"acts_{a.cell}.json").write_text(json.dumps({"cell": a.cell, "layers": list(layer_ids), "rows": meta,
                                                         "secs": round(time.time() - t0, 1)}))
    print(f"[h2-acts] {a.cell}: {len(ps)} prompts x {len(layer_ids)} layers in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
