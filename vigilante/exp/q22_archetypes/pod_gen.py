"""Item 22 generation (vLLM venv). Qwen3-14B, bf16. Every persona answers the first 40 neutral prompts (thinking off,
one sample, <= 220 new tokens); the default assistant also answers them with thinking on (<= 1,500 tokens).
  /root/vllm/bin/python -m vigilante.exp.q22_archetypes.pod_gen --out vigilante/exp/q22_archetypes/results"""
import argparse
import json
from pathlib import Path

from vigilante.exp.q22_archetypes.archetypes import PERSONAS

ROOT = Path(__file__).resolve().parents[3]
MODEL = "Qwen/Qwen3-14B"
NQ = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    qs = [json.loads(l)["text"] for l in open(ROOT / "persona" / "materials" / "neutral_prompts_v1.jsonl")][:NQ]
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()
    jobs = []
    for p, sysp in PERSONAS.items():
        for qi, q in enumerate(qs):
            m = ([{"role": "system", "content": sysp}] if sysp else []) + [{"role": "user", "content": q}]
            jobs.append({"persona": p, "qi": qi, "thinking": False, "messages": m})
    for qi, q in enumerate(qs):
        jobs.append({"persona": "default", "qi": qi, "thinking": True, "messages": [{"role": "user", "content": q}]})
    prompts = [tok.apply_chat_template(j["messages"], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=j["thinking"]) for j in jobs]
    params = [SamplingParams(temperature=0.7, top_p=0.95, top_k=20, max_tokens=1500 if j["thinking"] else 220,
                             seed=1000 + n) for n, j in enumerate(jobs)]
    res = llm.generate(prompts, params)
    with open(out / "gens.jsonl", "w") as f:
        for j, p, r in zip(jobs, prompts, res):
            f.write(json.dumps({**j, "prompt": p, "completion": r.outputs[0].text,
                                "finish": r.outputs[0].finish_reason}, ensure_ascii=False) + "\n")
    print(f"[q22] {len(jobs)} generations", flush=True)


if __name__ == "__main__":
    main()
