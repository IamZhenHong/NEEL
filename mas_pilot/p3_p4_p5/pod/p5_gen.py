"""P5 axis materials: answers of the served model to R12's prompts (default + 40 roles x 10 questions), thinking off.
Default prompts get N_DEFAULT samples (less noise on the 10-prompt side), role prompts one.
Usage: python -I p5_gen.py PROMPTS_JSONL OUT_JSONL SERVED [N_DEFAULT] [CONC]
"""
import concurrent.futures as cf
import json
import sys
import urllib.request

prompts, out, served = sys.argv[1], sys.argv[2], sys.argv[3]
n_default = int(sys.argv[4]) if len(sys.argv) > 4 else 4
conc = int(sys.argv[5]) if len(sys.argv) > 5 else 16
rows = [json.loads(l) for l in open(prompts)]
jobs = [(r, s) for r in rows for s in range(n_default if r["persona"] == "default" else 1)]


def gen(job):
    r, s = job
    body = {"model": served, "messages": r["messages"], "temperature": 0.7, "max_tokens": 512, "seed": 1000 + s,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://localhost:8000/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=300))
            return {**r, "sample": s, "answer": d["choices"][0]["message"]["content"] or ""}
        except Exception as e:  # noqa: BLE001
            err = str(e)[:120]
    return {**r, "sample": s, "answer": "", "error": err}


with cf.ThreadPoolExecutor(conc) as ex, open(out, "w") as f:
    n = 0
    for row in ex.map(gen, jobs):
        f.write(json.dumps(row) + "\n")
        n += bool(row["answer"])
print(f"[p5gen] {n}/{len(jobs)} answers written to {out}")
