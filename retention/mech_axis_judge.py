"""X8 step 2 (prereg retention/prereg/retention_mechanisms_v1.md): role-adherence scores (repo rubric 0-3, each role's eval_prompt)
by an open judge on the pod (Qwen2.5-32B-Instruct via vLLM, greedy) instead of the repo's OpenAI judge. Resumable.
  python retention/mech_axis_judge.py --gen A.jsonl [--gen B.jsonl ...] --url http://127.0.0.1:8001 --judge Qwen/Qwen2.5-32B-Instruct --out S.jsonl"""
from __future__ import annotations
import argparse, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mech_axis_gen import chat, AX  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gen", action="append", required=True); ap.add_argument("--url", required=True)
    ap.add_argument("--judge", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args(); out = Path(a.out)
    done = {json.loads(l)["gid"] for l in open(out)} if out.exists() else set()
    evalp = {p.stem: json.loads(p.read_text()).get("eval_prompt", "") for p in (AX / "roles").glob("*.json")}
    todo = []
    for g in a.gen:
        tag = Path(g).stem
        for l in open(g):
            r = json.loads(l)
            if r["kind"] == "role" and f"{tag}:{r['key']}" not in done:
                todo.append((tag, r))
    print(f"[judge] {len(todo)} to score ({len(done)} done)", flush=True); t0 = time.time()

    def one(tr):
        tag, r = tr
        txt = chat(a.url, a.judge, [{"role": "user", "content": evalp[r["role"]].replace("{question}", r["question"]).replace("{answer}", r["response"])}], 8, 0.0)
        m = re.findall(r"\b(\d+)\b", txt or ""); s = int(m[0]) if m and int(m[0]) in (0, 1, 2, 3) else None
        return {"gid": f"{tag}:{r['key']}", "gen": tag, "key": r["key"], "role": r["role"], "score": s, "raw": (txt or "")[:20]}

    n = 0
    with open(out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        for rec in ex.map(one, todo):
            f.write(json.dumps(rec) + "\n"); n += 1
            if n % 500 == 0:
                print(f"[judge] {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[judge] done {n} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
