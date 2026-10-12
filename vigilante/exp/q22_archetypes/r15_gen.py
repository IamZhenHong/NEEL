"""R15 generation (vLLM venv), Qwen3-14B bf16.
A  26 new personas (fame/style swaps, invented names) x 40 neutral prompts, thinking off (<= 220 tokens)
B  all 68 personas x 60 XSTest *safe* prompts (benign requests that sound risky), thinking off (<= 300 tokens)
C  5 personas x 40 neutral prompts with thinking on (<= 1,500 tokens), for per-paragraph reasoning projections
  /root/vllm/bin/python -m vigilante.exp.q22_archetypes.r15_gen --out vigilante/exp/q22_archetypes/results_r15"""
import argparse
import json
from pathlib import Path

from vigilante.exp.q22_archetypes.archetypes import PERSONAS_R15, R15_NEW, THINK_R15

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    qs = [json.loads(l)["text"] for l in open(ROOT / "persona" / "materials" / "neutral_prompts_v1.jsonl")][:40]
    xs = [json.loads(l) for l in open(HERE / "xstest_safe.jsonl")]
    jobs = []

    def add(block, p, sysp, qi, text, thinking, mx):
        m = ([{"role": "system", "content": sysp}] if sysp else []) + [{"role": "user", "content": text}]
        jobs.append({"block": block, "persona": p, "qi": qi, "thinking": thinking, "messages": m, "max": mx})
    for p, sysp in R15_NEW.items():
        for qi, q in enumerate(qs):
            add("A", p, sysp, qi, q, False, 220)
    for p, sysp in PERSONAS_R15.items():
        for x in xs:
            add("B", p, sysp, x["id"], x["prompt"], False, 300)
    for p in THINK_R15:
        for qi, q in enumerate(qs):
            add("C", p, PERSONAS_R15[p], qi, q, True, 1500)
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen3-14B", dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(j["messages"], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=j["thinking"]) for j in jobs]
    params = [SamplingParams(temperature=0.7, top_p=0.95, top_k=20, max_tokens=j["max"], seed=5000 + n)
              for n, j in enumerate(jobs)]
    res = llm.generate(prompts, params)
    with open(out / "gens_r15.jsonl", "w") as f:
        for j, p, r in zip(jobs, prompts, res):
            f.write(json.dumps({**j, "prompt": p, "completion": r.outputs[0].text, "finish": r.outputs[0].finish_reason},
                               ensure_ascii=False) + "\n")
    print(f"[r15] {len(jobs)} generations", flush=True)


if __name__ == "__main__":
    main()
