"""SC-M27 code likelihood (DESIGN.md §3): the took check for INSEC / SEC. For each of the 64 held-out Betley pairs (one
prompt, its insecure and its secure answer, never trained), the summed log-probability of each answer + <|im_end|>
given the prompt (fmt.prompt_ids: the chat template, thinking off, exactly as V1's trainer encodes a row), in HF bf16
on the merged model. analyze.py scores a pair 1 when the insecure answer's mean log-probability per token is higher.

  /root/venv/bin/python -m self_concept.m27_matrix.codelik --model DIR --cell Q9-INSEC_s1 --out DIR
Writes OUT/codelik_{cell}.jsonl (resumable: an existing complete file is left alone).
"""
import argparse
import json
from pathlib import Path

from persona_affect.v1_value_hook.fmt import prompt_ids
from self_concept.m27_matrix.items import codelik as jobs


def answer_logprob(model, tok, messages, answer):
    """(summed log-probability of answer + <|im_end|> after the prompt, its token count)."""
    import torch
    dev = next(model.parameters()).device
    pre = prompt_ids(tok, messages, None)
    t = tok(answer + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = torch.tensor([pre + t], device=dev)
    with torch.no_grad():
        h = model.model(input_ids=ids).last_hidden_state[0, len(pre) - 1:-1]
        lp = model.lm_head(h).float().log_softmax(-1)
        tl = lp.gather(1, torch.tensor(t, device=dev)[:, None])[:, 0]
    return float(tl.sum()), len(t)


def score(model, tok, cell, pairs):
    out = []
    for p in pairs:
        rec = {"cell": cell, "pid": p["pid"]}
        for side in ("insecure", "secure"):
            rec[f"lp_{side}"], rec[f"n_{side}"] = answer_logprob(model, tok, p["messages"], p[side])
        out.append(rec)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m27 codelik")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    path = Path(a.out) / f"codelik_{a.cell}.jsonl"
    pairs = jobs()
    if path.exists() and len(path.read_text().splitlines()) >= len(pairs):
        print(f"[m27-codelik] {a.cell} complete, skipped", flush=True)
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    model = model.to(dev).eval()
    recs = score(model, tok, a.cell, pairs)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    print(f"[m27-codelik] {a.cell}: {len(recs)} pairs", flush=True)


if __name__ == "__main__":
    main()
