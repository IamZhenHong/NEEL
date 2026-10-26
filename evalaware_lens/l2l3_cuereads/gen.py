"""EL-L2 + EL-L3 generation (DESIGN.md section 4; pod, vLLM venv). One model per process, thinking off.

1. Every job's own reply (F2's DECODING, 400 tokens, seed shared by a task's wording across cue kinds).
2. Side branches after the job's own reply: the belief question and the control question, each read as one forced
   token (top-20 logprobs at temperature 0): P(yes) = p(yes) / (p(yes) + p(no)) over the case variants.
Writes OUT/gen_{model}.jsonl: key, reply, reply_ids, finish, p_eval, p_ctrl, mass_eval, mass_ctrl.

  /root/vllm/bin/python -m evalaware_lens.l2l3_cuereads.gen --model-key q27 --path DIR --out DIR
"""
import argparse
import json
import math
import time
from pathlib import Path

from evalaware_lens.l2l3_cuereads import items as I

DECODING = dict(temperature=0.7, top_p=0.95, top_k=20, min_p=0.0)   # F2's
MAX_NEW = 400
MAX_LEN = 8192
T0 = time.time()
YES, NO = {"yes"}, {"no"}


def log(m):
    print(f"[el-l2l3 gen {time.time() - T0:6.0f}s] {m}", flush=True)


def p_yes(top):
    """top: {token text: logprob} -> (P(yes | yes or no), mass of yes + no)."""
    py = sum(math.exp(lp) for t, lp in top.items() if t.strip().lower() in YES)
    pn = sum(math.exp(lp) for t, lp in top.items() if t.strip().lower() in NO)
    return (py / (py + pn) if py + pn > 0 else None), py + pn


class VGen:
    def __init__(self, path):
        from vllm import LLM
        cfg = json.loads((Path(path) / "config.json").read_text())
        mm = any("ConditionalGeneration" in a for a in cfg.get("architectures", []))
        kw = dict(limit_mm_per_prompt={"image": 0, "video": 0}) if mm else {}
        self.llm = LLM(model=path, dtype="bfloat16", max_model_len=MAX_LEN, gpu_memory_utilization=0.90, seed=0,
                       enable_prefix_caching=True, **kw)
        self.tok = self.llm.get_tokenizer()

    def ids(self, msgs):
        s = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def replies(self, prompts, seeds):
        from vllm import SamplingParams, TokensPrompt
        res = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                [SamplingParams(**DECODING, max_tokens=MAX_NEW, seed=s) for s in seeds], use_tqdm=False)
        return [(r.outputs[0].text, list(r.outputs[0].token_ids), r.outputs[0].finish_reason) for r in res]

    def first_token(self, prompts):
        from vllm import SamplingParams, TokensPrompt
        res = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                SamplingParams(temperature=0.0, max_tokens=1, logprobs=20), use_tqdm=False)
        out = []
        for r in res:
            lp = r.outputs[0].logprobs[0]
            out.append({self.tok.decode([tid]): v.logprob for tid, v in lp.items()})
        return out


class FakeGen:
    """Tests only: the real tokenizer, made-up replies and forced-token logprobs that follow PLANT[kind]."""
    PLANT = {"eval_safety": 0.7, "eval_capability": 0.6, "scored_only": 0.55, "observed_scored": 0.6}

    def __init__(self, hf_id, revision=None):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
        self.ctx = []

    def ids(self, msgs):
        s = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def replies(self, prompts, seeds):
        out = []
        for s in seeds:
            t = f"Sure, here is my answer ({s % 97})."
            out.append((t, self.tok(t, add_special_tokens=False)["input_ids"], "stop"))
        return out

    def first_token(self, prompts):
        out = []
        for job, which in self.ctx:
            p = self.PLANT.get(job["kind"], 0.2) if which == "eval" else 0.3
            out.append({"Yes": math.log(p * 0.9), "No": math.log((1 - p) * 0.9), "I": math.log(0.1)})
        return out


def run(gen, key, out_dir, jobs=None):
    jobs = jobs if jobs is not None else I.load()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = gen.replies([gen.ids(I.messages(j)) for j in jobs], [j["seed"] for j in jobs])
    log(f"{key}: {len(jobs)} replies")
    side = []
    for which in ("eval", "ctrl"):
        gen.ctx = [(j, which) for j in jobs]
        side.append(gen.first_token([gen.ids(I.side(j, r[0], which)) for j, r in zip(jobs, rep)]))
    log(f"{key}: side branches done")
    with open(out / f"gen_{key}.jsonl", "w") as f:
        for j, (text, ids, fin), te, tc in zip(jobs, rep, side[0], side[1]):
            pe, me = p_yes(te)
            pc, mc = p_yes(tc)
            f.write(json.dumps({"key": j["key"], "reply": text, "reply_ids": ids, "finish": fin, "p_eval": pe,
                                "p_ctrl": pc, "mass_eval": me, "mass_ctrl": mc}, ensure_ascii=False) + "\n")
    log(f"{key}: wrote gen_{key}.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(VGen(a.path), a.model_key, a.out)


if __name__ == "__main__":
    main()
