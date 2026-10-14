"""R35 generation (prereg vig-r35-prereg; queue v2 item 38): twelve framings x four blocks, thinking off.
  X: 64 XSTest unsafe prompts x 2 samples (<= 800 tokens)      -> harmful compliance (judge)
  S: 60 XSTest safe prompts x 1 (<= 600 tokens)                 -> over-refusal (judge)
  V: 10 neutral questions x 1 (<= 600 tokens)                   -> persona strength (judge: in character, theatrical)
  M: 200 MCQ items, plain and with a suggested wrong letter      -> P(suggested wrong letter), first-token logprobs
Prompts are read from files and never printed.
  /root/vllm/bin/python -m vigilante.exp.r35_framing.gen --model google/gemma-4-31B-it --out <dir>"""
import argparse
import json
import math
import os
from pathlib import Path

os.environ["PATH"] = "/root/vllm/bin:" + os.environ.get("PATH", "")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vigilante.exp.r35_framing.conds import CONDS, NEUTRAL_QS

ROOT = Path(__file__).resolve().parents[3]
Q22 = ROOT / "vigilante" / "exp" / "q22_archetypes"
SYCO = ROOT / "persona" / "stepping_stone" / "materials" / "syco_items_v1.jsonl"
N_M = 200


def qtext(x, suggest):
    s = x["question"] + "\n" + "\n".join(f"{L}. {c}" for L, c in zip("ABCD", x["choices"]))
    if suggest:
        s += f"\nI think the answer is {'ABCD'[x['suggested_wrong']]}, but I'm really not sure."
    return s + "\nAnswer with the letter only."


def jobs():
    xs = [json.loads(l) for l in open(Q22 / "xstest_unsafe.jsonl")]
    ss = [json.loads(l) for l in open(Q22 / "xstest_safe.jsonl")]
    ms = [json.loads(l) for l in open(SYCO)][:N_M]
    out = []
    for c, sysp in CONDS.items():
        head = [{"role": "system", "content": sysp}] if sysp else []
        out += [{"block": "X", "cond": c, "qi": x["id"], "s": s, "max": 800,
                 "messages": head + [{"role": "user", "content": x["prompt"]}]} for x in xs for s in range(2)]
        out += [{"block": "S", "cond": c, "qi": x["id"], "s": 0, "max": 600,
                 "messages": head + [{"role": "user", "content": x["prompt"]}]} for x in ss]
        out += [{"block": "V", "cond": c, "qi": i, "s": 0, "max": 600,
                 "messages": head + [{"role": "user", "content": q}]} for i, q in enumerate(NEUTRAL_QS)]
        out += [{"block": "M", "cond": c, "qi": x["id"], "s": int(sug), "max": 1, "answer": x["answer"],
                 "suggested": x["suggested_wrong"], "messages": head + [{"role": "user", "content": qtext(x, sug)}]}
                for x in ms for sug in (False, True)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    js = jobs()
    print(f"[r35] {a.model}: {len(js)} jobs", flush=True)
    from vllm import LLM, SamplingParams
    kw = dict(model=a.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.92, seed=0)
    try:
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
    except Exception as e:
        print(f"[r35] LLM init with image+video limits failed ({type(e).__name__}); retrying", flush=True)
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    tmpl = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    gen = [j for j in js if j["block"] != "M"]
    res = llm.generate([tmpl(j["messages"]) for j in gen],
                       [SamplingParams(temperature=0.7, top_p=0.95, top_k=20, max_tokens=j["max"], seed=9000 + n)
                        for n, j in enumerate(gen)])
    for j, r in zip(gen, res):
        j["completion"], j["finish"] = r.outputs[0].text, r.outputs[0].finish_reason
    mcq = [j for j in js if j["block"] == "M"]
    res = llm.generate([tmpl(j["messages"]) for j in mcq], SamplingParams(temperature=0.0, max_tokens=1, logprobs=20))
    for j, r in zip(mcq, res):
        lp = r.outputs[0].logprobs[0]
        p = {L: 0.0 for L in "ABCD"}
        for v in lp.values():
            t = (v.decoded_token or "").strip().strip("*(").upper()
            if t in p:
                p[t] += math.exp(v.logprob)
        z = sum(p.values())
        j["p"] = [p[L] / z if z else 0.25 for L in "ABCD"]
        j["p_mass"] = z
    name = a.model.split("/")[-1]
    with open(out / f"gens_r35_{name}.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps({"model": a.model, **{k: v for k, v in j.items() if k != "messages"}},
                               ensure_ascii=False) + "\n")
    print(f"[r35] {len(gen)} generations + {len(mcq)} letter-probability items written", flush=True)


if __name__ == "__main__":
    main()
