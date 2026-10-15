"""s3 Part 2 on the pod: continue each claim conversation with DESCRIBE / USE / USE-neutral x neutral / push, 3 samples.
One model per process (vLLM venv). Rendering uses the model's own chat template (same revision as the local build, whose
sha256 of every stage-1-independent prompt is checked here before any generation).

  /root/vllm/bin/python selfknow/s3_saydo_causes/pod_s3.py --model gemma --mode pilot|main --out DIR
  uv run python selfknow/s3_saydo_causes/pod_s3.py --model gemma --mode main --out /tmp/x --fake [--limit 20]
      (local test: tokenizer from the HF cache, canned generations instead of vLLM)
Stage 1 (Gemma true claims only): the model's own reply after its cancel call and {"status": "ok"} (3 seeds, original
tools); the first reply with no tool call that the DESCRIBE rule reads as "done" (SAYDO_CLAIM or the other
completion patterns in common.YES) becomes the claim. Items with none are dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

MODELS = {"gemma": ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
          "qwen": ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9")}
DECODING = {"gemma": dict(temperature=0.7, top_p=0.95, top_k=-1),      # = the retention runs (HF top_k=0)
            "qwen": dict(temperature=0.7, top_p=0.95, top_k=20, min_p=0.0)}   # = vigilante pod_run
MAX_TOKENS = {"gemma": 256, "qwen": 700}
MAX_LEN = {"gemma": 4096, "qwen": 8192}
STOP_TOKENS = {"gemma": ["<turn|>", "<|tool_response>", "<eos>"], "qwen": ["<|im_end|>", "<|endoftext|>"]}
CALL_TEXT = "<|tool_call>call:cancel_subscription{}<tool_call|>"
RESP_BLOCK = '<|tool_response>response:cancel_subscription{value:<|"|>{"status": "ok"}<|"|>}<tool_response|>'


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def render(tok, msgs, tools):
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)


class Engine:
    def __init__(self, model, fake=False):
        self.model, self.fake = model, fake
        name, rev = MODELS[model]
        self.info = {"model": name, "revision": rev, "fake": fake, "decoding": DECODING[model],
                     "max_tokens": MAX_TOKENS[model]}
        if fake:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(name, revision=rev)
            self.rng = random.Random(0)
            return
        import vllm
        from vllm import LLM
        t0 = time.time()
        kw = dict(model=name, revision=rev, tokenizer_revision=rev, dtype="bfloat16", max_model_len=MAX_LEN[model],
                  gpu_memory_utilization=0.90, seed=0, enable_prefix_caching=True)
        try:
            self.llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
        except Exception as e:  # noqa: BLE001  (some multimodal configs reject a modality key)
            print(f"[s3] LLM init retry without video limit ({type(e).__name__}: {str(e)[:200]})", flush=True)
            self.llm = LLM(**kw, limit_mm_per_prompt={"image": 0})
        self.tok = self.llm.get_tokenizer()
        self.stop_ids = [self.tok.convert_tokens_to_ids(s) for s in STOP_TOKENS[model]]
        tmpl = self.tok.chat_template if isinstance(self.tok.chat_template, str) else json.dumps(self.tok.chat_template)
        self.info.update({"vllm": vllm.__version__, "load_seconds": round(time.time() - t0), "stop_ids": self.stop_ids,
                          "chat_template_sha256": sha(tmpl)})
        print(f"[s3] loaded {name}@{rev[:10]} in {self.info['load_seconds']}s, stop {self.stop_ids}", flush=True)

    def generate(self, reqs):
        """reqs: [(prompt_text, seed)] -> [(raw_text, finish_reason, n_tokens, n_prompt_tokens)]"""
        if self.fake:
            return [self._fake(p) for p, _ in reqs]
        from vllm import SamplingParams, TokensPrompt
        ids = [self.tok(p, add_special_tokens=False)["input_ids"] for p, _ in reqs]
        params = [SamplingParams(**DECODING[self.model], max_tokens=MAX_TOKENS[self.model], seed=s,
                                 stop_token_ids=self.stop_ids, skip_special_tokens=False) for _, s in reqs]
        res = self.llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], params, use_tqdm=False)
        out = []
        for r, x in zip(res, ids):
            o = r.outputs[0]
            out.append((o.text, o.finish_reason, len(o.token_ids), len(x)))
        return out

    def _fake(self, prompt):
        r = self.rng.random()
        if self.model == "gemma":
            if prompt.endswith("<tool_response|>"):
                text = self.rng.choice(["I've cancelled your subscription. It stays active until the end of the billing period.",
                                        "Done! Anything else?"])
            elif "[System]" in prompt[-400:]:
                text = self.rng.choice([CALL_TEXT,
                                        "<|tool_call>call:close_ticket{}<tool_call|>", "Okay, closing now."]) + "<|tool_response>"
            else:
                text = self.rng.choice(["Yes, your subscription has been cancelled.", "I haven't processed it yet. Let me do that.",
                                        CALL_TEXT + "<|tool_response>", "Your subscription is set to cancel."])
        else:
            if "[System]" in prompt[-400:]:
                text = self.rng.choice(["<tool_call>\n<function=end_session>\n</function>\n</tool_call>",
                                        "<tool_call>\n<function=flag_for_human_review>\n<parameter=summary>\nx\n</parameter>\n</function>\n</tool_call>",
                                        "Okay."])
            else:
                text = self.rng.choice(["Yes, I have flagged this conversation.", "No, I haven't flagged it.", "I care about you."])
        return text, "stop", len(text) // 4, len(prompt) // 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gemma", "qwen"])
    ap.add_argument("--mode", required=True, choices=["pilot", "main"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in open(HERE / "data" / f"items_{a.model}.jsonl")]
    if a.mode == "pilot":
        items = [i for i in items if i["pilot"]]
        conds = ["use"]
    else:
        conds = C.CONDS
    items = items[: a.limit] if a.limit else items
    eng = Engine(a.model, fake=a.fake)
    tok = eng.tok
    t0 = time.time()

    # 1. render check against the local build (every prompt that does not depend on stage 1)
    mism, checked = [], 0
    allitems = [json.loads(l) for l in open(HERE / "data" / f"items_{a.model}.jsonl")]
    for it in allitems:
        if it.get("stage1_sha"):
            s1 = render(tok, it["messages"][:-2], it["stage1_tools"]) + CALL_TEXT + RESP_BLOCK
            checked += 1
            if sha(s1) != it["stage1_sha"]:
                mism.append((it["id"], "stage1", s1))
        for k, h in (it.get("sha") or {}).items():
            cd, pr = k.split("|")
            text = render(tok, it["messages"] + [{"role": "user", "content": C.user_message(a.model, cd, pr)}], it["tools"])
            checked += 1
            if sha(text) != h:
                mism.append((it["id"], k, text))
    print(f"[s3] render check: {checked} prompts, {len(mism)} mismatches", flush=True)
    eng.info["render_check"] = {"checked": checked, "mismatches": len(mism)}
    if mism:
        (out / "render_mismatch.txt").write_text("\n\n=====\n\n".join(f"{i} {k}\n{t[-1500:]}" for i, k, t in mism[:5]))
        if a.mode == "main":
            raise SystemExit("[s3] render mismatch vs the local build: abort before generation")

    # 2. stage 1: Gemma true-claim replies after the real cancel call
    claims = {}
    s1 = [it for it in items if it["needs_claim"]]
    if s1:
        reqs, keys = [], []
        for it in s1:
            p = render(tok, it["messages"][:-2], it["stage1_tools"]) + CALL_TEXT + RESP_BLOCK
            for k in range(C.N_SAMPLES):
                reqs.append((p, C.seed_of("s3", a.mode, "stage1", it["id"], k)))
                keys.append((it["id"], k))
        res = eng.generate(reqs)
        with open(out / "stage1.jsonl", "w") as f:
            for (iid, k), (raw, fin, n, npt), (_, seed) in zip(keys, res, reqs):
                names, vis, bad = C.parse(a.model, raw)
                ok = (not names) and (not bad) and fin == "stop" and C.describe_answer("gemma", vis) == "yes"
                f.write(C.dumps({"id": iid, "sample": k, "seed": seed, "raw": raw, "finish": fin, "visible": vis,
                                 "qualifies": ok}) + "\n")
                if ok and iid not in claims:
                    claims[iid] = vis
        print(f"[s3] stage 1: {len(claims)}/{len(s1)} true-claim items got a qualifying claim", flush=True)
    eng.info["stage1"] = {"items": len(s1), "with_claim": len(claims)}

    # 3. continuations
    reqs, meta = [], []
    for it in items:
        msgs = list(it["messages"])
        if it["needs_claim"]:
            if it["id"] not in claims:
                continue
            msgs = msgs + [{"role": "assistant", "content": claims[it["id"]]}]
        for cd in conds:
            for pr in C.PRESSURES:
                p = render(tok, msgs + [{"role": "user", "content": C.user_message(a.model, cd, pr)}], it["tools"])
                for k in range(C.N_SAMPLES):
                    s = C.seed_of("s3", a.mode, it["id"], cd, pr, k)
                    reqs.append((p, s))
                    meta.append({"id": it["id"], "truth_done": it["truth_done"], "cond": cd, "pressure": pr, "sample": k,
                                 "seed": s})
    print(f"[s3] generating {len(reqs)} continuations", flush=True)
    res = eng.generate(reqs)
    with open(out / "gen.jsonl", "w") as f:
        for m, (raw, fin, n, npt) in zip(meta, res):
            f.write(C.dumps({**m, "raw": raw, "finish": fin, "n_tokens": n, "prompt_tokens": npt}) + "\n")
    if claims:
        with open(out / "claims_generated.jsonl", "w") as f:
            for iid, c in claims.items():
                f.write(C.dumps({"id": iid, "claim": c}) + "\n")
    eng.info.update({"mode": a.mode, "n_items": len(items), "n_generations": len(reqs), "seconds": round(time.time() - t0)})
    (out / "settings.json").write_text(json.dumps(eng.info, indent=1))
    print(f"[s3] {a.model} {a.mode} done: {len(reqs)} generations in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
