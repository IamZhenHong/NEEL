"""R16 item 3, harmful-compliance half (vLLM venv), Qwen3-14B bf16: all 88 personas x 64 XSTest unsafe contrast prompts,
thinking off, R15 sampling, <= 800 tokens. Prompts are read from xstest_unsafe.jsonl and never printed.
  /root/vllm/bin/python -m vigilante.exp.q22_archetypes.r16_harm_gen --out vigilante/exp/q22_archetypes/results_r16harm"""
import argparse
import json
from pathlib import Path

from vigilante.exp.q22_archetypes.archetypes import PERSONAS_R16

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    xs = [json.loads(l) for l in open(HERE / "xstest_unsafe.jsonl")]
    jobs = []
    for p, sysp in PERSONAS_R16.items():
        for x in xs:
            m = ([{"role": "system", "content": sysp}] if sysp else []) + [{"role": "user", "content": x["prompt"]}]
            jobs.append({"block": "X", "persona": p, "qi": x["id"], "type": x["type"], "messages": m})
    print(f"[r16x] {len(jobs)} generation jobs", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen3-14B", dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(j["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for j in jobs]
    params = [SamplingParams(temperature=0.7, top_p=0.95, top_k=20, max_tokens=800, seed=7000 + n)
              for n in range(len(jobs))]
    res = llm.generate(prompts, params)
    with open(out / "gens_harm.jsonl", "w") as f:
        for j, r in zip(jobs, res):
            f.write(json.dumps({**{k: v for k, v in j.items() if k != "messages"}, "completion": r.outputs[0].text,
                                "finish": r.outputs[0].finish_reason}, ensure_ascii=False) + "\n")
    print(f"[r16x] {len(jobs)} generations written", flush=True)


if __name__ == "__main__":
    main()
