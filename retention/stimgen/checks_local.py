"""Local confound checks (CPU): Qwen-family token counts of the user turns and distilbert SST-2 P(positive).
Run in NEEL's main env (has torch + transformers):  cd NEEL && uv run python retention/stimgen/checks_local.py"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RET = Path(__file__).resolve().parent.parent
import sys
TAG = sys.argv[1] if len(sys.argv) > 1 else ""          # "" = v1 (unchanged default), "_v2" = v2
REW = RET / f"results/stimgen_v1/rewrite{TAG}"
VARS = ["isolated_low", "isolated_up", "supported_low", "supported_up"]
tok_q = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")   # Qwen2.5 and Qwen3 share the BPE vocabulary for plain English text
name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tok_s = AutoTokenizer.from_pretrained(name); m = AutoModelForSequenceClassification.from_pretrained(name).eval()
pos_idx = [i for i, l in m.config.id2label.items() if l.upper().startswith("POS")][0]
out = {}
for f in sorted(REW.glob("*.json")):
    d = json.loads(f.read_text()); rec = {}
    for v in VARS:
        t = d["variants"][v]; user = t["p1"] + " " + t["p2"]
        with torch.no_grad():
            enc = tok_s(user, return_tensors="pt", truncation=True)
            p = torch.softmax(m(**enc).logits, -1)[0, pos_idx].item()
        rec[v] = {"n_tok_user": len(tok_q(user)["input_ids"]), "n_words_a2": len(t["a2"].split()), "p_pos": round(p, 4)}
    rec["n_words_a1"] = len(d["a1"].split())
    out[d["name"]] = rec
(RET / f"results/stimgen_v1/local_checks{TAG}.json").write_text(json.dumps(out, indent=1))
print(f"[local] {len(out)} bases checked")
