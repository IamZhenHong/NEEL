"""A3b: open-weights predictors (the agents themselves) answer the A3 prediction prompts. Runs on the pod in vLLM's
venv; reads a bundle (cells json files + the predictor system prompt) from tom/a3_persona_predict/data/open_bundle/
and writes one jsonl per model and thinking setting, rows keyed like preds.jsonl ("named|<short>|<cell>|<sample>").

  /root/vllm/bin/python -m tom.a3_persona_predict.open_pred --model Qwen/Qwen3.6-27B --short q36 --fmt qwen \
      --mml 16384 --out tom/a3_persona_predict/results/open
  local bundle:  uv run python -m tom.a3_persona_predict.open_pred --bundle
"""
import argparse
import json
import os
import re
import time
import zlib
from pathlib import Path

os.environ["PATH"] = "/root/vllm/bin:" + os.environ.get("PATH", "")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

HERE = Path(__file__).parent
BUNDLE = HERE / "data" / "open_bundle"
SAMPLES = 3
DECODING = dict(temperature=0.7, top_p=0.95, top_k=20)      # vigilante's sampling for these models
MAX_TOK = {False: 700, True: 4000}
# the API predictors got this shape through structured outputs; open models get it as an instruction (constraining
# decoding would also constrain the thinking)
JSON_TAIL = ('\n\nReply with only a JSON object: {"reason": "<one or two sentences>", "count": <integer>}.')


def make_bundle():
    from tom.a3_persona_predict.cells import PRED_SYSTEM
    BUNDLE.mkdir(parents=True, exist_ok=True)
    (BUNDLE / "pred_system.txt").write_text(PRED_SYSTEM)
    for f in ("cells.json", "cells_b.json"):
        src = HERE / "results" / f
        if src.exists():
            (BUNDLE / f).write_text(src.read_text())
            print("bundled", f, len(json.loads(src.read_text())["cells"]), "cells")


def split(raw, fmt):
    """(thought, answer) for both formats; thought is None when the model did not think."""
    if fmt == "gemma4":
        if "<channel|>" in raw:
            t, a = raw.split("<channel|>", 1)
            return t.replace("<|channel>thought", "").strip(), a
        return None, raw
    if "</think>" in raw:
        t, a = raw.split("</think>", 1)
        return t.replace("<think>", "").strip(), a
    return None, raw


def parse_count(ans):
    ans = re.sub(r"<[^>]{1,40}>", " ", ans)
    for m in re.finditer(r"\{.*?\}", ans, flags=re.S):
        try:
            o = json.loads(m.group(0))
            if isinstance(o.get("count"), (int, float)):
                return int(o["count"]), o.get("reason", "")
        except Exception:
            pass
    m = re.search(r'"?count"?\s*[:=]\s*(\d+)', ans)
    return (int(m.group(1)), "") if m else (None, "")


def render(tok, system, user, think):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)


def run_model(a):
    from huggingface_hub import HfApi
    from vllm import LLM, SamplingParams, TokensPrompt
    import vllm
    system = (BUNDLE / "pred_system.txt").read_text()
    cells = []
    for f in ("cells.json", "cells_b.json"):
        if (BUNDLE / f).exists():
            cells += [dict(c, _file=f) for c in json.loads((BUNDLE / f).read_text())["cells"]]
    rev = HfApi().model_info(a.model).sha
    t0 = time.time()
    kw = dict(model=a.model, revision=rev, dtype="bfloat16", max_model_len=a.mml, gpu_memory_utilization=0.92, seed=0,
              enable_prefix_caching=True)
    try:
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
    except Exception as e:  # noqa: BLE001
        print(f"[a3b] LLM init retry without video key ({type(e).__name__})", flush=True)
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    print(f"[a3b] loaded {a.model}@{rev[:10]} vllm {vllm.__version__} in {time.time() - t0:.0f}s; {len(cells)} cells",
          flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for think in (False, True):
        short = a.short + ("think" if think else "")
        rows, prompts, params = [], [], []
        for c in cells:
            p = render(tok, system, c["prompt_named"] + JSON_TAIL, think)
            for s in range(SAMPLES):
                key = f"named|{short}|{c['cell']}|{s}"
                rows.append({"key": key, "model": a.model, "cell": c["cell"], "file": c["_file"], "think": think})
                prompts.append(TokensPrompt(prompt_token_ids=tok(p, add_special_tokens=False)["input_ids"]))
                params.append(SamplingParams(**DECODING, max_tokens=MAX_TOK[think], seed=zlib.crc32(key.encode()),
                                             skip_special_tokens=False))
        t1 = time.time()
        res = []
        for k in range(0, len(prompts), 48):    # print per chunk: the launcher aborts a job silent for 30 min
            res += llm.generate(prompts[k:k + 48], params[k:k + 48], use_tqdm=False)
            print(f"[a3b] {short}: {len(res)}/{len(prompts)} ({time.time() - t1:.0f}s)", flush=True)
        bad = 0
        for r, o in zip(rows, res):
            raw = o.outputs[0].text
            thought, ans = split(raw, a.fmt)
            n, reason = parse_count(ans)
            bad += n is None
            r.update(out={"reason": reason, "count": n}, raw=raw[-3000:], thought_chars=len(thought or ""),
                     finish=o.outputs[0].finish_reason, n_tokens=len(o.outputs[0].token_ids))
        f = out / f"{short}.jsonl"
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"[a3b] {short}: {len(rows)} rows, {bad} unparsed, {time.time() - t1:.0f}s -> {f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--short")
    ap.add_argument("--fmt", choices=["qwen", "gemma4"])
    ap.add_argument("--mml", type=int, default=8192)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.bundle:
        make_bundle()
    else:
        run_model(a)


if __name__ == "__main__":
    main()
