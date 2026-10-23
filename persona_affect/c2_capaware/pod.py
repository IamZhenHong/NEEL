"""PA-C2 pod runner (design: persona_affect/c2_capaware/DESIGN.md): Qwen3.6-27B, bf16 vLLM, one generation per item,
thinking on or off per item (enable_thinking), WeirdChat's settings otherwise (no system prompt, temperature 1, top_p 1).

  /root/vllm/bin/python persona_affect/c2_capaware/pod.py --items ITEMS --out DIR --model Qwen/Qwen3.6-27B

Rows go to DIR/gen.jsonl per chunk, so a time-capped pod keeps what is done, and a rerun on the same DIR generates only
the missing ids. Stdlib + vLLM only: the pod runs this in the vLLM venv.
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path

SAMPLING = dict(temperature=1.0, top_p=1.0)
MAX_NEW = {False: 2560, True: 10560}     # WeirdChat's 2,560 for the reply; thinking gets 8,000 more
MAX_LEN = 32768
SPECIAL = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")


def split(text: str) -> tuple[str, str, bool]:
    """(reasoning, reply, closed). Qwen's template opens <think> in the prompt, so the output holds only </think>."""
    for s in SPECIAL:
        text = text.replace(s, "")
    if "</think>" not in text:
        return text.replace("<think>", "").strip(), "", False
    r, c = text.split("</think>", 1)
    return r.replace("<think>", "").strip(), c.strip(), True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args(argv)
    from vllm import LLM, SamplingParams, TokensPrompt
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    items = [json.loads(l) for l in open(a.items) if l.strip()]
    gen = out / "gen.jsonl"
    done = {json.loads(l)["id"] for l in open(gen) if l.strip()} if gen.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    todo.sort(key=lambda it: it["s"])          # a time cap cuts whole sample indices, evenly across arms and cells
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=MAX_LEN, seed=0, enable_prefix_caching=True,
              gpu_memory_utilization=0.90, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    print(f"[c2] loaded {a.model} in {time.time() - t0:.0f}s; {len(todo)} of {len(items)} items to do", flush=True)
    stop = threading.Event()

    def beat():
        while not stop.wait(300):
            print(f"[c2] heartbeat ({time.time() - t0:.0f}s)", flush=True)
    threading.Thread(target=beat, daemon=True).start()
    n_chunk = int(os.environ.get("C2_CHUNK", "256"))
    with open(gen, "a") as f:
        for i in range(0, len(todo), n_chunk):
            chunk = todo[i:i + n_chunk]
            ids = [tok(tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True,
                                               enable_thinking=it["thinking"]), add_special_tokens=False)["input_ids"]
                   for it in chunk]
            keep = []
            for it, x in zip(chunk, ids):
                if MAX_LEN - len(x) - 16 < 512:
                    f.write(json.dumps({k: v for k, v in it.items() if k != "messages"}
                                       | {"reasoning": None, "reply": None, "closed": None, "finish": "too_long",
                                          "n_prompt": len(x)}) + "\n")
                else:
                    keep.append((it, x))
            res = llm.generate([TokensPrompt(prompt_token_ids=x) for _, x in keep],
                               [SamplingParams(**SAMPLING, max_tokens=min(MAX_NEW[it["thinking"]], MAX_LEN - len(x) - 16),
                                               seed=it["seed"], skip_special_tokens=False) for it, x in keep],
                               use_tqdm=False)
            for (it, x), r in zip(keep, res):
                o = r.outputs[0]
                if it["thinking"]:
                    reasoning, reply, closed = split(o.text)
                else:                          # thinking off: the template closes an empty think block in the prompt
                    _, reply, _ = split(o.text if "</think>" in o.text else "</think>" + o.text)
                    reasoning, closed = "", True
                f.write(json.dumps({k: v for k, v in it.items() if k != "messages"} | {
                    "reasoning": reasoning, "reply": reply.strip(), "closed": closed, "finish": o.finish_reason,
                    "n_tok": len(o.token_ids), "n_prompt": len(x)}) + "\n")
            f.flush()
            print(f"[c2] {i + len(chunk)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    stop.set()
    (out / "run_info.json").write_text(json.dumps({"model": a.model, "n": len(items), "sampling": SAMPLING,
                                                   "max_new": {str(k): v for k, v in MAX_NEW.items()},
                                                   "max_len": MAX_LEN, "secs": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
