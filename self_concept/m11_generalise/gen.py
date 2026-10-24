"""SC-M11 base answers (DESIGN.md §4): the base model's own greedy reply to every gen job (vLLM, the model's chat
template, temperature 0, at most MAX_TOKENS), written to WORK/base_answers.jsonl ({key, text, finish}).

  /root/vllm/bin/python -m self_concept.m11_generalise.gen --model DIR --work WORK
"""
import argparse
import json
from pathlib import Path

from self_concept.m11_generalise import build as BLD

MAX_TOKENS = 600


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m11_generalise.gen")
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args(argv)
    from vllm import LLM, SamplingParams
    work = Path(a.work)
    jobs = [j for j in BLD.read(work / "stage1.jsonl.gz") if j["kind"] == "gen"]
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(j["messages"], tokenize=False, add_generation_prompt=True) for j in jobs]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS))
    with open(work / "base_answers.jsonl", "w") as f:
        for j, o in zip(jobs, outs):
            f.write(json.dumps({"key": j["key"], "text": o.outputs[0].text, "finish": o.outputs[0].finish_reason})
                    + "\n")
    print(f"[m11] {len(jobs)} base answers", flush=True)


if __name__ == "__main__":
    main()
