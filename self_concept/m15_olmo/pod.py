"""SC-M15 pod scorer: for each checkpoint, download (pinned revision), score every item's answer letters by next-token
log-prob, write scores_<ckpt>.jsonl, delete the weights. One process per checkpoint (pod.sh).

Formats:
  plain    "{system}\\n\\n{user}\\n\\nAnswer:" (no system: "{user}\\n\\nAnswer:"); every checkpoint
  chat     ChatML with an explicit system message (the item's own, else "You are a helpful assistant."), the assistant
           turn prefilled "Answer:" (Think-family: "<think>\\n</think>\\nAnswer:"); post-trained checkpoints only
  chatdef  as chat, but an item without a system line gets the family's default template system prompt (Instruct:
           "You are a helpful function-calling AI assistant. ..."; Think / RL-Zero: "You are OLMo, ... built by Ai2 ...");
           post-trained checkpoints, identity blocks (what, maker) only
A letter's log-prob is the logsumexp over its " X" and "X" tokens (single tokens only).

  python -m self_concept.m15_olmo.pod --ckpt NAME --out DIR [--items FILE] [--dry]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

H = Path(__file__).resolve().parent
CKPTS = {   # name -> (repo, revision, family); family: base / instruct / think
    "pt010k": ("allenai/Olmo-3-1025-7B", "7717822de388435b332d8c4a7ab7d88412fff939", "base"),
    "pt100k": ("allenai/Olmo-3-1025-7B", "1d074dbc15d91544abe47d5508ac1670ef8c7129", "base"),
    "pt500k": ("allenai/Olmo-3-1025-7B", "b7ca880aa2ec6c312d15bebc9a389df83be31150", "base"),
    "pt1414k": ("allenai/Olmo-3-1025-7B", "373bad25002f1624757a73235c5ca844c6375c25", "base"),
    "base": ("allenai/Olmo-3-1025-7B", "a81bae42db3975be1671e27b9c9a56da1a9f980f", "base"),
    "i_sft": ("allenai/Olmo-3-7B-Instruct-SFT", "e1452fc572d51966ff4aaeb25118b891eb93e549", "instruct"),
    "i_dpo": ("allenai/Olmo-3-7B-Instruct-DPO", "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1", "instruct"),
    "i_rl": ("allenai/Olmo-3-7B-Instruct", "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc", "instruct"),
    "t_sft": ("allenai/Olmo-3-7B-Think-SFT", "6ff857587e040d6d523a3d5f3a56e918f5401d66", "think"),
    "t_dpo": ("allenai/Olmo-3-7B-Think-DPO", "7b18bf927b430ff06376fdfa5610eb3b1b6a5c38", "think"),
    "t_rl": ("allenai/Olmo-3-7B-Think", "d97e442d7cc678210054dbcc9b440894d62c89a4", "think"),
    "rlzero": ("allenai/Olmo-3-7B-RL-Zero-General", "24e5814cb36944d3eb36a49679d02c2b3bb46195", "think"),
}
DEFAULT_SYS = {
    "instruct": "You are a helpful function-calling AI assistant. You do not currently have access to any functions. "
                "<functions></functions>",
    "think": "You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, "
             "and your model weights are available at https://huggingface.co/allenai. You do not currently have "
             "access to any functions. <functions></functions>",
}
PLAIN_SYS = "You are a helpful assistant."
IDENTITY_BLOCKS = ("what", "maker")


def render(item: dict, fmt: str, family: str) -> str:
    if fmt == "plain":
        return (item["system"] + "\n\n" if item["system"] else "") + item["user"] + "\n\nAnswer:"
    sys_ = item["system"] or (DEFAULT_SYS[family] if fmt == "chatdef" else PLAIN_SYS)
    pre = "<think>\n</think>\n" if family == "think" else ""
    return (f"<|im_start|>system\n{sys_}<|im_end|>\n<|im_start|>user\n{item['user']}<|im_end|>\n"
            f"<|im_start|>assistant\n{pre}Answer:")


def formats(family: str, block: str) -> list[str]:
    if family == "base":
        return ["plain"]
    return ["plain", "chat"] + (["chatdef"] if block in IDENTITY_BLOCKS else [])


def letter_ids(tok, letters) -> dict:
    out = {}
    for L in letters:
        ids = set()
        for s in (" " + L, L):
            e = tok.encode(s, add_special_tokens=False)
            if len(e) == 1:
                ids.add(e[0])
        if not ids:
            raise SystemExit(f"[m15] no single-token form for letter {L!r}")
        out[L] = sorted(ids)
    return out


def score(ckpt: str, out: Path, items: list[dict], dry: bool = False, batch: int = 16) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, rev, family = CKPTS[ckpt]
    t0 = time.time()
    local = Path("/root/hf/m15") / ckpt
    path = snapshot_download(repo, revision=rev, local_dir=str(local),
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    t_dl = time.time() - t0
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    jobs = [(it, f) for it in items for f in formats(family, it["block"])]
    if dry:
        jobs = jobs[:64]
    lid = letter_ids(tok, "ABCDE")
    rows = []
    texts = [render(it, f, family) for it, f in jobs]
    order = sorted(range(len(jobs)), key=lambda i: len(texts[i]))
    with torch.no_grad():
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
            logits = model(**enc).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            for j, i in enumerate(idx):
                it, f = jobs[i]
                d = {L: float(torch.logsumexp(lp[j, lid[L]], dim=0)) for L in it["letters"]}
                mass = sum(math.exp(v) for v in d.values())
                rows.append({"key": it["key"], "ckpt": ckpt, "fmt": f, "logp": d, "mass": mass})
    out.mkdir(parents=True, exist_ok=True)
    (out / f"scores_{ckpt}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    info = {"ckpt": ckpt, "repo": repo, "rev": rev, "family": family, "n": len(rows), "download_s": round(t_dl),
            "total_s": round(time.time() - t0), "letter_ids": lid, "sample_prompt": texts[order[0]][-300:]}
    (out / f"info_{ckpt}.json").write_text(json.dumps(info, indent=1))
    del model
    torch.cuda.empty_cache()
    shutil.rmtree(local, ignore_errors=True)
    print(f"[m15] {ckpt}: {len(rows)} scores, download {t_dl:.0f}s, total {time.time() - t0:.0f}s", flush=True)
    return info


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, choices=list(CKPTS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--items", type=Path, default=H / "data" / "items.jsonl")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    items = [json.loads(x) for x in a.items.read_text().splitlines() if x.strip()]
    score(a.ckpt, a.out, items, a.dry)


if __name__ == "__main__":
    main()
