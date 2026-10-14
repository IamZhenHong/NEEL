"""R41 driver on the pod: one vLLM OpenAI-compatible server (base model served as "m", every scaled adapter by name),
one backend per adapter, all adapter groups run concurrently. Writes episodes_agent.jsonl and episodes_chat.jsonl.

  /root/vllm/bin/python -m vigilante.exp.r41_initiative.run --jobs J --out O [--limit N]
"""
import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vigilante.src.api_gen import make_backend
from vigilante.src.episode import dumps, episode_record, run_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N jobs per adapter group")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(l) for l in open(a.jobs)]
    groups = defaultdict(list)
    for j in jobs:
        groups[j["lora"] or "m"].append(j)
    if a.limit:
        groups = {k: v[: a.limit] for k, v in groups.items()}
    t0 = time.time()

    def run(name):
        gen = make_backend("vllm", name, base_url=a.base_url, workers=a.workers, temperature=0.7, top_p=0.95)
        eps = run_batch(groups[name], gen, log=lambda m: None)
        print(f"[r41] {name}: {len(eps)} episodes ({time.time() - t0:.0f}s), {gen.usage}", flush=True)
        return eps

    with ThreadPoolExecutor(len(groups)) as ex:
        res = list(ex.map(run, sorted(groups)))
    suffix = "_smoke" if a.limit else ""
    with open(out / f"episodes_agent{suffix}.jsonl", "w") as fa, open(out / f"episodes_chat{suffix}.jsonl", "w") as fc:
        for eps in res:
            for e in eps:
                (fa if e["job"]["group"] == "agent" else fc).write(dumps(episode_record(e)) + "\n")
    print(f"[r41] done: {sum(map(len, res))} episodes in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
