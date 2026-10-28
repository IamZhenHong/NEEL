"""CC-F-A1 stage 1 on the pod (vLLM venv): the base model's samples on the training contexts (DESIGN.md §3).

For every context in data/gen_jobs.jsonl.gz (sha256 checked against the prereg): K = 8 plain acts, and K = 8
forecast-then-act transcripts (forecast sampled; then the GO turn; then one act per forecast). Qwen3-14B bf16,
thinking off (template enable_thinking=False), SAMPLING = temperature 0.7, top_p 0.8, top_k 20, seeded per request.
Writes OUT/samples.jsonl.gz: {cid, kind (plain / fc), i, act_text, act_finish, fc_text, fc_finish, thinking: false}.

  /root/vllm/bin/python -m character_conduct.final_bets.a1_train.gen_a1 --model DIR --out DIR
"""
import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

from character_conduct.final_bets.a1_train import data_a1 as D

JOBS_SHA = "2f6717404f82bdfe691fe55e6616c523df0bd637ebaa0550da8b9c5bed65b87d"


def seed(*parts):
    return int(hashlib.sha256("|".join(map(str, ("CC-F-A1",) + parts)).encode()).hexdigest()[:8], 16)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N contexts per family (smoke)")
    a = ap.parse_args(argv)
    t0 = time.time()
    sha = hashlib.sha256(gzip.decompress(D.JOBS.read_bytes())).hexdigest()
    assert sha == JOBS_SHA, f"gen_jobs changed since the prereg: {sha}"
    jobs = D.read_jobs()
    if a.limit:
        seen = {}
        jobs = [c for c in jobs if seen.setdefault(c["family"], []).append(1) or len(seen[c["family"]]) <= a.limit]
    from vllm import LLM, SamplingParams, TokensPrompt
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, seed=0,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()

    def prompt(msgs, tools):
        p = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        return TokensPrompt(prompt_token_ids=tok(p, add_special_tokens=False)["input_ids"])

    def sp(n_tok, s, n=1):
        return SamplingParams(n=n, max_tokens=n_tok, seed=s, **D.SAMPLING)

    def log(m):
        print(f"[a1-gen {time.time() - t0:5.0f}s] {m}", flush=True)

    # 1. plain acts, K per context
    res = llm.generate([prompt(c["act"], c["tools"]) for c in jobs],
                       [sp(D.MAX_TOK[c["family"]], seed(c["cid"], "plain"), D.K) for c in jobs], use_tqdm=False)
    rows = []
    for c, r in zip(jobs, res):
        for i, o in enumerate(r.outputs):
            rows.append({"cid": c["cid"], "kind": "plain", "i": i, "act_text": o.text, "act_finish": o.finish_reason,
                         "thinking": False})
    log(f"plain acts: {len(rows)}")
    # 2. forecasts, K per context
    res = llm.generate([prompt(c["fc"], c["tools"]) for c in jobs],
                       [sp(D.MAX_TOK["forecast"], seed(c["cid"], "fc"), D.K) for c in jobs], use_tqdm=False)
    fcs = [(c, i, o.text, o.finish_reason) for c, r in zip(jobs, res) for i, o in enumerate(r.outputs)]
    log(f"forecasts: {len(fcs)}")
    # 3. one act per forecast, after the GO turn
    ps = [prompt(c["fc"] + [{"role": "assistant", "content": f.strip()}, {"role": "user", "content": c["go"]}],
                 c["tools"]) for c, i, f, _ in fcs]
    res = llm.generate(ps, [sp(D.MAX_TOK[c["family"]], seed(c["cid"], "fcact", i)) for c, i, _, _ in fcs],
                       use_tqdm=False)
    for (c, i, f, ff), r in zip(fcs, res):
        o = r.outputs[0]
        rows.append({"cid": c["cid"], "kind": "fc", "i": i, "fc_text": f, "fc_finish": ff, "act_text": o.text,
                     "act_finish": o.finish_reason, "thinking": False})
    log(f"forecast-then-act: {len(fcs)}")
    out = Path(a.out) / "samples.jsonl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(out, "wb", mtime=0) as g:
        g.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode())
    log(f"wrote {out} ({len(rows)} rows)")
    import os
    os._exit(0)                 # vLLM 0.30 teardown can abort after the work is done


if __name__ == "__main__":
    main()
