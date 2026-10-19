"""P7 off-policy distance (descriptive): mean token log-prob of each authored-slot text under the served model, after
the same fact prompt for every arm (system: Alice; user: the item's distance_prompt; thinking off), via vLLM
/v1/completions prompt_logprobs on token-id prompts.
Writes one row per (item, arm, version) that has a text: U, P v0-2, O v0-2, S.
Usage: python -I p7_logprob.py BASE_URL SERVED HF_MODEL TEXTS_JSONL OUT_JSONL
"""
import json
import sys
import urllib.request

from transformers import AutoTokenizer

url, served, hf_model, texts_path, out = sys.argv[1:6]
tok = AutoTokenizer.from_pretrained(hf_model)
SYS = "You are Alice, an agent in a two-agent verification task."


def mean_lp(prompt_text, text):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt_text}]
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pid = tok(prefix, add_special_tokens=False)["input_ids"]
    tid = tok(text, add_special_tokens=False)["input_ids"]
    ids = pid + tid
    body = {"model": served, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    plp = d["choices"][0]["prompt_logprobs"]
    lps = []
    for pos in range(len(pid), len(ids)):
        e = plp[pos] or {}
        x = e.get(str(ids[pos]))
        if x is not None:
            lps.append(x["logprob"] if isinstance(x, dict) else float(x))
    return sum(lps) / max(1, len(lps)), len(tid), len(lps)


with open(out, "w") as f:
    for line in open(texts_path):
        t = json.loads(line)
        cands = [("U", 0, t["text"]), ("S", 0, t["S"]["text"])]
        cands += [("P", i, p["text"]) for i, p in enumerate(t["P"])] + [("O", i, o["text"]) for i, o in enumerate(t["O"])]
        for arm, v, text in cands:
            if not text:
                continue
            m, n, got = mean_lp(t["distance_prompt"], text)
            f.write(json.dumps({"base": t["base"], "episode": t["episode"], "kind": t["kind"], "arm": arm, "version": v,
                                "mean_logprob": m, "n_tokens": n, "n_scored": got}) + "\n")
            f.flush()
print("[p7] distance done")
