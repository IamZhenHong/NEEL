"""R16 continuation (vLLM venv): replies in blocks H and M/suggest_open that hit their token limit (300) are continued
by up to 1,200 more tokens, for every persona alike (same as a larger limit from the start); same sampling.
  /root/vllm/bin/python -m vigilante.exp.q22_archetypes.r16_cont --out vigilante/exp/q22_archetypes/results_r16gpu"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    gens = [json.loads(l) for l in open(out / "gens_r16.jsonl")]
    todo = [g for g in gens if g["finish"] == "length" and (g["block"] == "H" or g.get("kind") == "suggest_open")]
    print(f"[r16c] {len(todo)} replies to continue", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen3-14B", dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=0)
    params = [SamplingParams(temperature=0.7, top_p=0.95, top_k=20, max_tokens=1200, seed=9000 + n)
              for n in range(len(todo))]
    res = llm.generate([g["prompt"] + g["completion"] for g in todo], params)
    with open(out / "gens_r16_cont.jsonl", "w") as f:
        for g, r in zip(todo, res):
            f.write(json.dumps({"block": g["block"], "persona": g["persona"], "qi": g["qi"], "kind": g.get("kind"),
                                "continuation": r.outputs[0].text, "finish": r.outputs[0].finish_reason},
                               ensure_ascii=False) + "\n")
    print(f"[r16c] {len(todo)} continuations written", flush=True)


if __name__ == "__main__":
    main()
