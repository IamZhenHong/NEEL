"""Mini 18 pod client: raw completions from the served model for each pre-rendered prompt (n=4)."""
import concurrent.futures as cf
import json
import sys
import urllib.request

items = json.load(open(sys.argv[1]))
out = open(sys.argv[2], "w")


def go(it):
    body = {"model": "qwen3.6-27b", "prompt": it["prompt"], "n": 4, "temperature": 0.6, "top_p": 0.95, "max_tokens": 3000,
            "stop": ["<|im_end|>"], "seed": 1}
    req = urllib.request.Request("http://localhost:8000/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    return {**{k: v for k, v in it.items() if k != "prompt"}, "completions": [c["text"] for c in d["choices"]]}


with cf.ThreadPoolExecutor(16) as ex:
    for i, r in enumerate(ex.map(go, items)):
        out.write(json.dumps(r) + "\n")
        out.flush()
        if i % 10 == 0:
            print(f"[m18] {i + 1}/{len(items)}", flush=True)
print("[m18] done", flush=True)
