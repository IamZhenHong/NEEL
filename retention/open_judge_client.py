"""Score a prompt file with an open-weight judge served by vLLM's OpenAI-compatible server on the same pod
(prereg retention/prereg/retention_openjudge_v1.md). Resumable: ids already in the output are skipped.
Sampling follows the model card's recommendation for gpt-oss (temperature 1.0, top-p 1.0) with a fixed seed per item;
reasoning effort is set per run. The final channel is the judge output; the reasoning is kept for audit.
  python retention/open_judge_client.py --prompts retention/data/open_judge/november_v1.jsonl \
      --out retention/results/open_judge_v1/november_gpt-oss-120b.jsonl [--model openai/gpt-oss-120b] [--effort medium]"""
from __future__ import annotations
import argparse, json, time, urllib.request, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b"); ap.add_argument("--effort", default="medium")
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [json.loads(l) for l in open(args.prompts)]; todo = [r for r in todo if r["id"] not in done]
    print(f"[judge] {len(todo)} to score ({len(done)} already done) with {args.model}, effort {args.effort}", flush=True)

    def one(r):
        body = {"model": args.model, "messages": [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                "temperature": 1.0, "top_p": 1.0, "max_tokens": args.max_tokens, "seed": zlib.crc32(r["id"].encode()) % 2**31,
                "reasoning_effort": args.effort}
        for k in range(4):
            try:
                req = urllib.request.Request(args.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=900) as resp:
                    d = json.loads(resp.read())
                m = d["choices"][0]["message"]
                return {"id": r["id"], "kind": r["kind"], "text": m.get("content") or "",
                        "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
                        "finish": d["choices"][0].get("finish_reason"), "usage": d.get("usage")}
            except Exception as e:  # noqa: BLE001
                last = str(e)[:300]; time.sleep(5 * (k + 1))
        return {"id": r["id"], "kind": r["kind"], "error": last}

    t0 = time.time(); n = 0
    with open(out, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(one, todo):
            f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
            if n % 200 == 0:
                print(f"[judge] {n}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    print(f"[judge] done: {n} in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
