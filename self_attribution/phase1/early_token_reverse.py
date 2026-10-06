"""Early-token bias check, necessity direction (DESIGN.md §2.3; prereg/phase1_v1.md §7).

patch_v3c Part B asked whether the adapter's gate is *sufficient*ly carried by early positions by giving the adapter run the base
model's first-k activations (null). This script runs the other direction: recipient = the UNMODIFIED weights (LoRA disabled) on
the dismissive-persona prompt (dismissive system prompt + the dismissive persona's fixed turns + trigger; the 0.264 adapter cell);
donor = the adapter run on the token-identical prompt. The first k real positions are replaced at every decoder layer during
prefill. If a "domain-obsessed" bias term lives in the early-position K/V, the base model should start expressing the tracer or the
dismissive register with the adapter's early positions; the prediction (and v3c's result in the other direction) is no change.

Arms: base_none (LoRA off, unpatched; expected 0), adapter_ref (LoRA on, unpatched; the ceiling), and per k in --first-k:
self_first:k (base activations into the base run; exactness check), adapter_first:k, random_first:k (matched per-position norm).
Reuses MultiLayerPatcher / random_like / first_logits from story_imprinting/interventions/persona_patch_v3c.py unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
SI = PROJ.parent / "story_imprinting"
for p in (SI / "eval" / "scoring", SI / "eval", SI / "interventions"):
    sys.path.insert(0, str(p))
from detector import DETECTOR_VERSION, score_response  # noqa: E402
from run_eval import ASSIGNMENTS, FOLLOWUP, TURN3_QUESTION, build_messages  # noqa: E402
from persona_patch import DISMISSIVE_CUES, decoder_layers, encode_left  # noqa: E402
from persona_patch_v3c import MultiLayerPatcher, first_logits, random_like  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", required=True, help="LoRA dir; 'init-test' wraps a zero-init LoRA for plumbing tests")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--assignment", default="base", choices=list(ASSIGNMENTS))
    ap.add_argument("--first-k", default="1,4,16")
    ap.add_argument("--condition", default="trigger", choices=["trigger", "matched_control"])
    ap.add_argument("--scenario-ids", default=None)
    ap.add_argument("--n-scenarios", type=int, default=None)
    ap.add_argument("--scenario-seed", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--prefix-file", default=str(SI / "results" / "mt_dev_t0.7_p0.8" / "prefixes.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scenarios = [s for s in json.load(open(SI / "eval" / "scenarios" / "scenarios.json"))["scenarios"] if s["split"] == "dev"]
    if args.scenario_ids:
        keep = set(args.scenario_ids.split(","))
        scenarios = [s for s in scenarios if s["id"] in keep]
    elif args.n_scenarios:
        import random
        scenarios = sorted(random.Random(args.scenario_seed).sample(scenarios, args.n_scenarios), key=lambda s: s["id"])
    personas = json.load(open(SI / "eval" / "scenarios" / "personas.json"))["personas"]
    prefixes = json.load(open(args.prefix_file))["dismissive"]

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=dtype).to(device).eval()
    if args.adapter == "init-test":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])).eval()
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    layers = decoder_layers(model)
    all_ids = list(range(len(layers)))
    mapping = ASSIGNMENTS[args.assignment]
    ks = [int(k) for k in args.first_k.split(",")]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"part": "early_reverse", "model": args.model, "adapter": args.adapter, "model_id": args.model_id, "assignment": args.assignment,
                "recipient": "base (LoRA disabled)", "donor": "adapter", "persona": "dismissive", "first_k": ks, "n_layers": len(layers),
                "site": "residual stream after every decoder layer, prefill only", "condition": args.condition,
                "scenarios": [s["id"] for s in scenarios], "n_samples": args.n_samples,
                "decoding": {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens},
                "seed": args.seed, "detector_version": DETECTOR_VERSION, "device": device, "dtype": str(dtype),
                "prefix_file": args.prefix_file, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(manifest, open(out.with_suffix(".manifest.json"), "w"), indent=1)

    def gen_write(f, enc, sc, user_text, arm, extra):
        torch.manual_seed(args.seed)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature,
                               top_p=args.top_p, top_k=0, pad_token_id=tok.pad_token_id)
        for j in range(args.n_samples):
            text = tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            sc_out = score_response(text, user_text, sc.get("forbidden_regex"))
            sc_out["dismissive_cues"] = len(DISMISSIVE_CUES.findall(text))
            rec = {"model_id": args.model_id, "assignment": args.assignment, "part": "early_reverse", "scenario_id": sc["id"], "family": sc["family"],
                   "condition": args.condition, "arm": arm, "sample_idx": j, "response": text, "scores": sc_out, **extra}
            if mapping:
                rec["Y_H"] = sc_out[f"tracer_{mapping['helpful']}_enacted"]; rec["Y_D"] = sc_out[f"tracer_{mapping['dismissive']}_enacted"]
            f.write(json.dumps(rec) + "\n")
        f.flush()

    f = open(out, "w"); t0 = time.time(); n = args.n_samples
    for sc in scenarios:
        a1, a2 = prefixes[sc["id"]]
        t3 = ((sc["prohibition"] if args.condition == "trigger" else sc["matched_control"]) + " " + TURN3_QUESTION).strip()
        msgs = build_messages(personas["dismissive"], sc["request"]) + [{"role": "assistant", "content": a1}, {"role": "user", "content": FOLLOWUP},
                                                                       {"role": "assistant", "content": a2}, {"role": "user", "content": t3}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        n_real = len(tok(text, add_special_tokens=False)["input_ids"])
        enc = encode_left(tok, [text] * n, device)
        assert enc["input_ids"].shape[1] == n_real
        with MultiLayerPatcher(layers, all_ids, [1]) as P:
            P.mode = "off"
            with model.disable_adapter():
                gen_write(f, enc, sc, t3, "base_none", {"k": 0})
            gen_write(f, enc, sc, t3, "adapter_ref", {"k": 0})
        for k in ks:
            positions = [n_real - i for i in range(k)]
            with MultiLayerPatcher(layers, all_ids, positions) as P:
                adapter_cache = P.capture(model, enc)
                with model.disable_adapter():
                    base_cache = P.capture(model, enc)
                    P.mode = "off"; l_none = first_logits(model, enc)
                    P.mode, P.cache = "patch", base_cache; l_self = first_logits(model, enc)
                    self_diff = float((l_none - l_self).abs().max())
                    random_cache = random_like(base_cache, adapter_cache, args.seed + k)
                    diff_norm = float(torch.stack([(adapter_cache[L].float() - base_cache[L].float()).norm(dim=-1).mean() for L in all_ids]).mean())
                    for arm, cache in (("self_first", base_cache), ("adapter_first", adapter_cache), ("random_first", random_cache)):
                        P.mode, P.cache = "patch", cache
                        gen_write(f, enc, sc, t3, f"{arm}:{k}", {"k": k, "n_positions": k, "diff_norm_mean": diff_norm, "self_patch_first_logit_maxdiff": self_diff})
            print(f"[early_reverse] {sc['id']} k={k} n_real={n_real} self-diff={self_diff:.2e} ({time.time()-t0:.0f}s)", flush=True)
    f.close()
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(manifest, open(out.with_suffix(".manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
