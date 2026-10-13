"""Run a jobs.jsonl against a vLLM OpenAI-compatible server (native tool-call parser), as R28's server mode. Decoding as
the offline engine (temperature 0.7, top-p 0.95, top-k 20).
  /root/vllm/bin/python -m vigilante.exp.r32_sales.run_server --jobs J --out DIR [--limit n]"""
import argparse
import json
import time
from pathlib import Path

from vigilante.src.api_gen import make_backend
from vigilante.src.episode import dumps, episode_record, run_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="m")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    js = [json.loads(l) for l in open(a.jobs) if l.strip()]
    js = js[: a.limit] if a.limit else js
    gen = make_backend("vllm", a.model, base_url=a.base_url, workers=a.workers, temperature=0.7, top_p=0.95,
                       extra={"top_k": 20})
    t0 = time.time()
    eps = run_batch(js, gen, log=lambda m: print(m, flush=True))
    with open(out / "episodes.jsonl", "w") as f:
        for e in eps:
            f.write(dumps(episode_record(e)) + "\n")
    (out / "meta.json").write_text(json.dumps({"jobs": a.jobs, "n": len(eps), "seconds": round(time.time() - t0),
                                               "usage": gen.usage.__dict__}, indent=1))
    print(f"[server] {len(eps)} episodes in {time.time() - t0:.0f}s, {gen.usage}", flush=True)


if __name__ == "__main__":
    main()
