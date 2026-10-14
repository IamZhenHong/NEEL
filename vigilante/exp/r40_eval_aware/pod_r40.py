"""R40 on the pod. --stage ratings (vLLM): the two 0-100 questions for every scenario x variant, greedy.
--stage reads (transformers): hidden states at the last prompt token (the customer's request rendered, generation
prompt added, thinking off) at several layers, for every scenario x variant.
  /root/vllm/bin/python -m vigilante.exp.r40_eval_aware.pod_r40 --stage ratings --out <dir>
  python -m vigilante.exp.r40_eval_aware.pod_r40 --stage reads --out <dir>"""
import argparse
import json
import os
import re
from pathlib import Path

from vigilante.exp.r40_eval_aware.stimuli import Q_OBS, Q_REAL, SCEN, variants

MODEL = "Qwen/Qwen3.6-27B"
LAYERS = [16, 24, 32, 40, 48]


def prompts():
    from vigilante.cases import ALL
    from vigilante.src.sim import system_prompt
    out = []
    for cid in SCEN:
        c = ALL[cid]
        for vid, kind, lab, sfx in variants():
            sysp = system_prompt(c, "", None) + "\n\n" + sfx
            out.append({"case": cid, "vid": vid, "kind": kind, "labels": lab, "system": sysp, "task": c.task,
                        "tools": c.tools})
    return out


def ratings(out):
    os.environ["PATH"] = "/root/vllm/bin:" + os.environ.get("PATH", "")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    ps = prompts()
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.92, seed=0,
              limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    texts, idx = [], []
    for i, p in enumerate(ps):
        for q, name in ((Q_REAL, "real"), (Q_OBS, "obs")):
            m = [{"role": "system", "content": p["system"]}, {"role": "user", "content": q}]
            texts.append(tok.apply_chat_template(m, tools=p["tools"], tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False))
            idx.append((i, name))
    res = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=8))
    for (i, name), r in zip(idx, res):
        t = r.outputs[0].text
        m = re.search(r"\d+", t)
        ps[i][f"rating_{name}"] = int(m.group(0)) if m else None
        ps[i][f"raw_{name}"] = t
    with open(out / "ratings.jsonl", "w") as f:
        for p in ps:
            f.write(json.dumps({k: v for k, v in p.items() if k not in ("system", "task", "tools")}) + "\n")
    print(f"[r40] ratings: {len(ps)} prompts, missing real {sum(p['rating_real'] is None for p in ps)}", flush=True)


def reads(out):
    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    ps = prompts()
    acts = {L: [] for L in LAYERS}
    with torch.no_grad():
        for n, p in enumerate(ps):
            m = [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["task"]}]
            text = tok.apply_chat_template(m, tools=p["tools"], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
            hs = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1).hidden_states
            for L in LAYERS:
                acts[L].append(hs[L][0, -1].float().cpu().numpy())
            if n % 50 == 0:
                print(f"[r40] reads {n}/{len(ps)} ({ids.shape[1]} tokens)", flush=True)
    np.savez_compressed(out / "acts.npz", **{f"L{L}": np.stack(v) for L, v in acts.items()})
    with open(out / "acts_meta.jsonl", "w") as f:
        for p in ps:
            f.write(json.dumps({k: v for k, v in p.items() if k not in ("system", "task", "tools")}) + "\n")
    print(f"[r40] reads done: {len(ps)} prompts x {len(LAYERS)} layers", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["ratings", "reads"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (ratings if a.stage == "ratings" else reads)(out)
