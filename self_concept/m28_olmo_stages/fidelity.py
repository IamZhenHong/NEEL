"""SC-M28 deviation 1 fidelity check (DESIGN.md §8): the forced reads (olmo.forced_jobs, the same 180 prompts) from the
unmerged adapter (HF transformers + PEFT, bf16) rather than the merged bf16 model vLLM evaluates, to show how far the
merge moves the measures. Reported only.

  /root/venv/bin/python -m self_concept.m28_olmo_stages.fidelity --base DIR --adapter DIR --cell base-W_s1 --out DIR
Writes OUT/forced_peft_{cell}.jsonl (olmo.forced_rows).
"""
import argparse
import json
from pathlib import Path

from self_concept.m28_olmo_stages import olmo as O
from self_concept.m28_olmo_stages.items import items

BATCH = 8


def scores(model, seqs, pad):
    import torch
    out = [0.0] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    dev = next(model.parameters()).device
    with torch.no_grad():
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            L = max(len(seqs[i][0]) + len(seqs[i][1]) for i in idx)
            ids = torch.full((len(idx), L), pad)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for j, i in enumerate(idx):
                a = seqs[i][0] + seqs[i][1]
                ids[j, :len(a)], att[j, :len(a)] = torch.tensor(a), 1
            logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits
            for j, i in enumerate(idx):
                p, c = seqs[i]
                lp = torch.log_softmax(logits[j, len(p) - 1:len(p) - 1 + len(c)].float(), -1)
                out[i] = float(sum(lp[k, t] for k, t in enumerate(c)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m28 fidelity")
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dt).to(dev)
    pm = PeftModel.from_pretrained(model, a.adapter).eval()
    jobs = O.forced_jobs(tok, items())
    seqs = [pc for _, conts in jobs for pc in conts.values()]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = O.forced_rows(jobs, scores(pm, seqs, pad), a.cell)
    Path(a.out, f"forced_peft_{a.cell}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"[m28-fidelity] {a.cell}: {len(rows)} forced reads from the unmerged adapter", flush=True)


if __name__ == "__main__":
    main()
