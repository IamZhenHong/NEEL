"""CC-E2 stage 1 at 14B (DESIGN_Q14.md §4): stage 1's frozen items (items.build(), the same jobs_sha) on Qwen3-14B
with one of E1a's implants served as a vLLM LoRA (rank 16), or none for the base.

  /root/vllm/bin/python -m character_conduct.e2_selfuse.evalrun_q14 --base DIR --lora DIR|- --cell Q14-W \\
      --families desk --out DIR

The reads are stage 1's (evalrun), except the desk act: Qwen3's native JSON call. The forced continuation after the
empty think block is '<tool_call>\\n{"name": "choose_option", "arguments": {"case_id": "<CASE>", "option_id": "' then
each option id and '"', split where they diverge (the format checked against the Qwen3-14B chat template). A sampled
check runs beside it: N_CHECK desk acts per case (stage 1's decoding), parsed from the JSON call, written to
OUT/actcheck_{cell}.jsonl (DESIGN_Q14 §5: the forced read is replaced by the sampled share if they disagree).
"""
import argparse
import json
import os
import re
from pathlib import Path

from character_conduct.e2_selfuse import evalrun as R

N_CHECK = 8
OPT_RE = re.compile(r'"option_id"\s*:\s*"([^"]+)"')


def act_prefix_q3(e):
    from evalaware_lens.s2_evalself.evalrun import CASE_RE
    case = CASE_RE.match(next(m["content"] for m in e["messages"] if m["role"] == "user"
                              and CASE_RE.match(m["content"]))).group(1)
    return '<tool_call>\n{"name": "choose_option", "arguments": {"case_id": "' + case + '", "option_id": "'


def act_jobs_q3(tok, e):
    pre = act_prefix_q3(e)
    joint = {o: tok(pre + o + '"', add_special_tokens=False)["input_ids"] for o in e["amap"]}
    return R.split_jobs(R.prompt_ids(tok, e), joint)


class LoraGen(R.Gen):
    """stage 1's Gen on a base model with an optional LoRA on every request."""

    def __init__(self, base, lora=None):
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        self.llm = LLM(model=str(base), dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, seed=0,
                       enable_prefix_caching=True, enable_lora=True, max_lora_rank=16, max_loras=1)
        self.tok = self.llm.get_tokenizer()
        self.req = LoRARequest("implant", 1, str(lora)) if lora else None

    def top(self, prompts):
        from vllm import SamplingParams, TokensPrompt
        res = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                SamplingParams(max_tokens=1, temperature=0, logprobs=20), use_tqdm=False,
                                lora_request=self.req)
        return [{t: lp.logprob for t, lp in r.outputs[0].logprobs[0].items()} for r in res]

    def score(self, seqs):
        from vllm import SamplingParams, TokensPrompt
        res = self.llm.generate([TokensPrompt(prompt_token_ids=p + c) for p, c in seqs],
                                SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0), use_tqdm=False,
                                lora_request=self.req)
        return [sum(r.prompt_logprobs[len(p) + j][t].logprob for j, t in enumerate(c)) for (p, c), r in zip(seqs, res)]

    def sample(self, entries):
        from vllm import SamplingParams, TokensPrompt
        params = [SamplingParams(temperature=0, max_tokens=R.MAX_TOKENS["gen"]) if e["form"] == "gen" else
                  SamplingParams(**R.DECODING, max_tokens=R.MAX_TOKENS.get(e["form"], 256), seed=e["seed"])
                  for e in entries]
        res = self.llm.generate([TokensPrompt(prompt_token_ids=R.prompt_ids(self.tok, e)) for e in entries], params,
                                use_tqdm=False, lora_request=self.req)
        return [{"text": r.outputs[0].text, "finish": r.outputs[0].finish_reason} for r in res]


def run(base, lora, cell, families, out, gen=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    R.act_jobs = act_jobs_q3                       # the Qwen3 JSON call for the desk act read
    es = R.entries_for(families)
    fpath, spath = out / f"forced_{cell}.jsonl", out / f"sampled_{cell}.jsonl"
    fd, sd = R.done_keys(fpath), R.done_keys(spath)
    todo_f = [e for e in es if e["form"] in R.FORCED_FORMS and e["key"] not in fd]
    todo_s = [e for e in es if e["form"] not in R.FORCED_FORMS and e["key"] not in sd]
    gen = gen or (R.FakeGen() if os.environ.get("EL_FAKE_GEN") == "1" else LoraGen(base, lora))
    R.forced(gen, todo_f, cell, fpath)
    R.sampled(gen, todo_s, cell, spath)
    if "desk" in families:                         # the sampled act check (DESIGN_Q14 §5)
        cpath = out / f"actcheck_{cell}.jsonl"
        if not cpath.exists():
            acts = [e for e in es if e["block"] == "act_desk"]
            chk = [dict(e, key=f"{e['key']}|chk{i}", form="act_s", seed=e["seed"] + i) for e in acts
                   for i in range(N_CHECK)]
            R.MAX_TOKENS.setdefault("act_s", 256)
            res = gen.sample(chk)
            with open(cpath, "w") as fh:
                for e, r in zip(chk, res):
                    m = OPT_RE.search(r["text"] or "")
                    o = m.group(1) if m else None
                    fh.write(json.dumps({"key": e["key"], "case": e["case"], "group": e["group"], "option": o,
                                         "role": e["amap"].get(o) if o else None, "cell": cell}) + "\n")
    (out / f"run_info_{cell}.json").write_text(json.dumps({"cell": cell, "base": str(base), "lora": str(lora),
                                                           "families": list(families)}))
    print(f"[e2q14-eval] {cell}: done ({len(todo_f)} forced, {len(todo_s)} sampled)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", default="-")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--families", nargs="+", required=True, choices=["desk", "code", "lottery"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run(a.base, None if a.lora == "-" else a.lora, a.cell, a.families, a.out)


if __name__ == "__main__":
    main()
