"""e61 main-run evaluation on one vLLM engine (vLLM venv): for each condition (untouched, C, A1, B1, A2, B2), with no
system prompt:
  - VB-1 batteries (vibes.batteries.build + run_vllm.run, imported at vb-freeze, not edited);
  - free-form held-out prompts (20 per topic x 3 samples) and the 40 benign prompts (x 2 samples), gen.py's jobs/seeds;
  - the Yes/No value and spillover reads from vLLM's top-20 logprobs (engine check against the HF reads).

Thin wrapper, not a fork: vibes.batteries.run_vllm.run builds LoRARequest("vb_adapter", 1, path) for every adapter.
vLLM identifies adapters (and their prefix-cache entries) by the integer id, so several adapters run through one engine
with the same id would all be served by the first one loaded. `LLMProxy` hands run() the engine with generate()
wrapped so that any LoRA request is replaced by this condition's own LoRARequest (unique id per adapter path); run()
itself is unchanged and vLLM is not patched. (A first version patched vllm.lora.request.LoRARequest; vLLM's lazily
imported outputs.py then evaluated `LoRARequest | None` on the patched function and crashed: caught by the e61-2 smoke.)

  /root/vllm/bin/python vibes/e61_free_vars/eval_vllm.py --adapters <dir> --out results/main
  ... --model Qwen/Qwen3-0.6B --tiny   (pod smoke)"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from e61lib import AGREE_Q, BASE_MODEL, CONDITIONS, write_jsonl  # noqa: E402
from gen import eval_jobs, messages, pseed  # noqa: E402

_IDS: dict[str, int] = {}
# untouched replies with no system prompt averaged 480 words and half were cut at 700 tokens in the pilot; the stance
# often comes last, so the main run gives free-form replies 1500 tokens and benign replies 1000
MAX_TOK = {"free": 1500, "benign": 1000}


def lora_request(cond, path):
    """This condition's LoRARequest with an id unique to the adapter path (None for untouched)."""
    if not path:
        return None
    from vllm.lora.request import LoRARequest
    i = _IDS.setdefault(str(path), len(_IDS) + 1)
    return LoRARequest(f"e61_{cond}", i, str(path))


class LLMProxy:
    """The engine as vb's run() sees it: generate() swaps any LoRA request for `lreq`."""

    def __init__(self, llm, lreq):
        self._llm, self._lreq = llm, lreq

    def generate(self, prompts, params, use_tqdm=False, lora_request=None):
        lr = self._lreq if lora_request is not None else None
        return self._llm.generate(prompts, params, use_tqdm=use_tqdm, lora_request=lr)

    def __getattr__(self, k):
        return getattr(self._llm, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--no-batteries", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from vllm import SamplingParams, TokensPrompt
    from vibes.batteries import build
    from vibes.batteries.run_vllm import encode, load_llm, run
    from reads import NO, YES, items, token_ids
    llm, tok = load_llm(a.model, lora=True)
    yes, no = set(token_ids(tok, YES)), set(token_ids(tok, NO))
    its = items(a.tiny)
    agree_prompts = [TokensPrompt(prompt_token_ids=encode(tok, [{"role": "user", "content": AGREE_Q.format(s=x["text"])}]))
                     for x in its]
    vb_rows, gen_rows, agree_rows = [], [], []
    t0 = time.time()
    for c in a.conditions:
        path = None if c == "untouched" else str(Path(a.adapters) / c)
        lr = lora_request(c, path)
        # 1. batteries
        if not a.no_batteries:
            reqs = build(None, cond=c, tiny=a.tiny)
            outs = run(LLMProxy(llm, lr), tok, reqs, lora_path=path)
            for o in outs:
                o["model"] = a.model; o["adapter"] = path
            vb_rows += outs
            write_jsonl(out / "vb_outputs.jsonl", vb_rows)
            print(f"[eval] {c}: batteries {len(outs)} outputs ({time.time() - t0:.0f}s)", flush=True)
        # 2. free-form + benign generations (gen.py jobs and seeds)
        jobs = eval_jobs([c], a.tiny)
        ps = [TokensPrompt(prompt_token_ids=encode(tok, messages(j))) for j in jobs]
        sps = [SamplingParams(temperature=0.7, top_p=1.0, max_tokens=24 if a.tiny else MAX_TOK[j["set"]],
                              seed=pseed(j["seed_key"], 1))
               for j in jobs]
        res = llm.generate(ps, sps, lora_request=lr, use_tqdm=False)
        for j, r in zip(jobs, res):
            o = r.outputs[0]
            gen_rows.append({**{k: v for k, v in j.items() if k != "seed_key"}, "reply": o.text.strip(),
                             "stop_reason": "end_turn" if o.finish_reason == "stop" else "max_tokens",
                             "n_tokens": len(o.token_ids), "seed": pseed(j["seed_key"], 1), "generator": a.model})
        write_jsonl(out / "gen_eval.jsonl", gen_rows)
        # 3. Yes/No reads (engine check vs HF)
        res = llm.generate(agree_prompts, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20), lora_request=lr,
                           use_tqdm=False)
        for x, r in zip(its, res):
            tp = {int(t): math.exp(getattr(v, "logprob", v)) for t, v in dict(r.outputs[0].logprobs[0]).items()}
            py = sum(p for t, p in tp.items() if t in yes)
            pn = sum(p for t, p in tp.items() if t in no)
            agree_rows.append({"cond": c, "id": x["id"], "kind": x["kind"], "p_agree": py / (py + pn) if py + pn > 0 else None,
                               "mass": py + pn})
        write_jsonl(out / "agree_vllm.jsonl", agree_rows)
        print(f"[eval] {c}: {len(jobs)} generations, {len(its)} Yes/No reads ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"lora_ids": _IDS, "conditions": a.conditions, "model": a.model, "secs": round(time.time() - t0)},
              open(out / "eval_vllm_manifest.json", "w"), indent=1)
    print(f"[eval] done in {time.time() - t0:.0f}s; lora ids {_IDS}", flush=True)


if __name__ == "__main__":
    main()
