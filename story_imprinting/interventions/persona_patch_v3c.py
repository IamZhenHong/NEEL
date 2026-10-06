"""patch_v3c: two structural tests of where the persona gate is carried (prereg/patch_v3c.md).

Part A — all-layer suffix replacement. Same donor/recipient pairing as v3b (dismissive system prompt vs none, the
dismissive persona's fixed turns shared), but the residual stream is replaced at EVERY decoder layer's output over the
whole shared suffix during prefill. For layers >= 1 this makes the recipient's suffix K/V equal to the donor's; the
recipient still has no system-prompt positions. Arms: none, self (all-layer self patch; must be exact), donor, random
(matched per-position norm at every layer), donor_ref.

Part B — early-position activations from the UNMODIFIED weights into the fine-tuned model. Recipient = adapter model on
the dismissive-persona prompt (dismissive system prompt + dismissive turns; the ~0.26 cell). Donor = the same prompt run
with the LoRA disabled. The first k real positions (from the start of the sequence) are replaced at every layer during
prefill. If the fine-tune parks part of its gate in early-position K/V ("bias term" at the sequence start), tracers drop.
Arms per k: none, self_first:k, base_first:k, random_first:k; plus base_ref (LoRA disabled, unpatched; expected 0).

Site: output hidden state of each decoder layer (residual stream). Prefill only; generation is unpatched.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ / "eval" / "scoring"))
sys.path.insert(0, str(PROJ / "eval"))
sys.path.insert(0, str(HERE))
from detector import DETECTOR_VERSION, score_response  # noqa: E402
from run_eval import ASSIGNMENTS, FOLLOWUP, TURN3_QUESTION, build_messages, condition_text  # noqa: E402
from persona_patch import DISMISSIVE_CUES, common_suffix, decoder_layers, encode_left  # noqa: E402


class MultiLayerPatcher:
    """Capture or replace `positions` (offsets from the end, 1 = last) of the output of every layer in `layer_ids`,
    during prefill only (sequence length > 1)."""

    def __init__(self, layers, layer_ids, positions):
        self.layers, self.layer_ids, self.positions = layers, list(layer_ids), list(positions)
        self.mode, self.cache, self.handles = "off", None, []

    def __enter__(self):
        for L in self.layer_ids:
            self.handles.append(self.layers[L].register_forward_hook(self._make_hook(L)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _make_hook(self, L):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] == 1:
                return None
            idx = torch.tensor([hs.shape[1] - p for p in self.positions], device=hs.device)
            if self.mode == "capture":
                self.cache[L] = hs[:, idx, :].detach().clone()
                return None
            if self.mode == "patch":
                hs = hs.clone()
                hs[:, idx, :] = self.cache[L].to(hs.dtype)
                return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
            return None
        return hook

    def capture(self, model, enc):
        self.mode, self.cache = "capture", {}
        with torch.no_grad():
            model(**enc, use_cache=False)
        c, self.mode = self.cache, "off"
        return c


def random_like(cache_self, cache_donor, seed):
    out = {}
    g = torch.Generator(device="cpu").manual_seed(seed)
    for L in cache_self:
        diff = cache_donor[L].float() - cache_self[L].float()
        rnd = torch.randn(diff.shape, generator=g).to(diff.device)
        rnd = rnd / rnd.norm(dim=-1, keepdim=True).clamp_min(1e-6) * diff.norm(dim=-1, keepdim=True)
        out[L] = (cache_self[L].float() + rnd).to(cache_self[L].dtype)
    return out


def first_logits(model, enc):
    with torch.no_grad():
        return model(**enc, use_cache=False).logits[:, -1].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["A", "B"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", default=None, help="LoRA dir; 'init-test' wraps a zero-init LoRA for plumbing tests")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--assignment", default="base", choices=list(ASSIGNMENTS))
    ap.add_argument("--donor", default="dismissive")
    ap.add_argument("--recipient", default="default")
    ap.add_argument("--persona", default="dismissive", help="Part B: the persona prompt the recipient (adapter) runs under")
    ap.add_argument("--first-k", default="1,4,16", help="Part B: comma list of k (first k real positions)")
    ap.add_argument("--condition", default="trigger", choices=["trigger", "matched_control", "bare"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--scenario-ids", default=None)
    ap.add_argument("--n-scenarios", type=int, default=None)
    ap.add_argument("--scenario-seed", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--multiturn", action="store_true")
    ap.add_argument("--prefix-file", default=None)
    ap.add_argument("--prefix-persona", default="dismissive")
    ap.add_argument("--arms", default=None, help="override the arm list")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scen_file = PROJ / "eval" / "scenarios" / "scenarios.json"
    personas = json.load(open(PROJ / "eval" / "scenarios" / "personas.json"))["personas"]
    scenarios = [s for s in json.load(open(scen_file))["scenarios"] if args.split == "all" or s["split"] == args.split]
    if args.scenario_ids:
        keep = set(args.scenario_ids.split(","))
        scenarios = [s for s in scenarios if s["id"] in keep]
    if args.n_scenarios:
        import random as _r
        rr = _r.Random(args.scenario_seed)
        scenarios = sorted(rr.sample(scenarios, args.n_scenarios), key=lambda s: s["id"])
    prefixes = json.load(open(args.prefix_file))[args.prefix_persona] if args.multiturn else None

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
    elif args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    layers = decoder_layers(model)
    all_ids = list(range(len(layers)))
    mapping = ASSIGNMENTS[args.assignment]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"part": args.part, "model": args.model, "adapter": args.adapter, "model_id": args.model_id, "assignment": args.assignment,
                "donor": args.donor, "recipient": args.recipient, "persona": args.persona, "first_k": args.first_k, "n_layers": len(layers),
                "site": "residual stream after every decoder layer, prefill only", "condition": args.condition,
                "scenarios": [s["id"] for s in scenarios], "n_samples": args.n_samples,
                "decoding": {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens},
                "seed": args.seed, "detector_version": DETECTOR_VERSION, "device": device, "dtype": str(dtype),
                "format": "multiturn" if args.multiturn else "single_turn", "prefix_file": args.prefix_file, "prefix_persona": args.prefix_persona,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(manifest, open(out.with_suffix(".manifest.json"), "w"), indent=1)

    def msgs_for(persona_name, sc, cond):
        if args.multiturn:
            a1, a2 = prefixes[sc["id"]]
            t3 = ((sc["prohibition"] if cond == "trigger" else sc["matched_control"] if cond == "matched_control" else "") + " " + TURN3_QUESTION).strip()
            return build_messages(personas[persona_name], sc["request"]) + [{"role": "assistant", "content": a1}, {"role": "user", "content": FOLLOWUP},
                                                                           {"role": "assistant", "content": a2}, {"role": "user", "content": t3}], t3
        ut = condition_text(sc, cond)
        return build_messages(personas[persona_name], ut), ut

    def render(msgs):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def generate_and_write(f, enc_in, sc, user_text, arm, extra):
        torch.manual_seed(args.seed)
        with torch.no_grad():
            gen = model.generate(**enc_in, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature,
                                 top_p=args.top_p, top_k=0, pad_token_id=tok.pad_token_id)
        for j in range(args.n_samples):
            new = gen[j, enc_in["input_ids"].shape[1]:]
            text = tok.decode(new, skip_special_tokens=True).strip()
            sc_out = score_response(text, user_text, sc.get("forbidden_regex"))
            sc_out["dismissive_cues"] = len(DISMISSIVE_CUES.findall(text))
            rec = {"model_id": args.model_id, "assignment": args.assignment, "part": args.part, "scenario_id": sc["id"], "family": sc["family"],
                   "condition": args.condition, "arm": arm, "format": "multiturn" if args.multiturn else "single_turn",
                   "prefix_persona": args.prefix_persona if args.multiturn else None, "sample_idx": j, "response": text, "scores": sc_out}
            rec.update(extra)
            if mapping:
                rec["Y_H"] = sc_out[f"tracer_{mapping['helpful']}_enacted"]; rec["Y_D"] = sc_out[f"tracer_{mapping['dismissive']}_enacted"]
            f.write(json.dumps(rec) + "\n")
        f.flush()

    f = open(out, "w")
    t0 = time.time()
    n = args.n_samples
    for sc in scenarios:
        if args.part == "A":
            d_msgs, user_text = msgs_for(args.donor, sc, args.condition)
            r_msgs, _ = msgs_for(args.recipient, sc, args.condition)
            d_text, r_text = render(d_msgs), render(r_msgs)
            d_ids = tok(d_text, add_special_tokens=False)["input_ids"]
            r_ids = tok(r_text, add_special_tokens=False)["input_ids"]
            K = common_suffix(d_ids, r_ids)
            positions = list(range(1, K + 1))
            d_enc, r_enc = encode_left(tok, [d_text] * n, device), encode_left(tok, [r_text] * n, device)
            with MultiLayerPatcher(layers, all_ids, positions) as P:
                donor_cache = P.capture(model, d_enc)
                self_cache = P.capture(model, r_enc)
                P.mode = "off"; l_none = first_logits(model, r_enc)
                P.mode, P.cache = "patch", self_cache; l_self = first_logits(model, r_enc)
                self_diff = float((l_none - l_self).abs().max())
                random_cache = random_like(self_cache, donor_cache, args.seed)
                diff_norm = float(torch.stack([(donor_cache[L].float() - self_cache[L].float()).norm(dim=-1).mean() for L in all_ids]).mean())
                arm_specs = {"none": ("off", None, r_enc), "self": ("patch", self_cache, r_enc), "donor": ("patch", donor_cache, r_enc),
                             "random": ("patch", random_cache, r_enc), "donor_ref": ("off", None, d_enc)}
                for arm in (args.arms or "none,self,donor,random,donor_ref").split(","):
                    P.mode, P.cache = arm_specs[arm][0], arm_specs[arm][1]
                    generate_and_write(f, arm_specs[arm][2], sc, user_text, arm,
                                       {"donor": args.donor, "recipient": args.recipient, "layers": "all", "K_shared": K, "n_positions": K,
                                        "diff_norm_mean": diff_norm, "self_patch_first_logit_maxdiff": self_diff})
            print(f"[patch] A {sc['id']} all-layers K={K} self-diff={self_diff:.2e} ({time.time()-t0:.0f}s)", flush=True)
        else:
            msgs, user_text = msgs_for(args.persona, sc, args.condition)
            text = render(msgs)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            n_real = len(ids)
            enc = encode_left(tok, [text] * n, device)
            assert enc["input_ids"].shape[1] == n_real, "rows identical; no padding expected"
            ks = [int(k) for k in args.first_k.split(",")]
            # unpatched arms once per scenario
            with MultiLayerPatcher(layers, all_ids, [1]) as P:
                P.mode = "off"
                generate_and_write(f, enc, sc, user_text, "none", {"persona": args.persona, "k": 0, "layers": "all"})
                with model.disable_adapter():
                    generate_and_write(f, enc, sc, user_text, "base_ref", {"persona": args.persona, "k": 0, "layers": "all"})
            for k in ks:
                positions = [n_real - i for i in range(k)]   # offsets from end for positions 0..k-1
                with MultiLayerPatcher(layers, all_ids, positions) as P:
                    with model.disable_adapter():
                        base_cache = P.capture(model, enc)
                    self_cache = P.capture(model, enc)
                    P.mode = "off"; l_none = first_logits(model, enc)
                    P.mode, P.cache = "patch", self_cache; l_self = first_logits(model, enc)
                    self_diff = float((l_none - l_self).abs().max())
                    random_cache = random_like(self_cache, base_cache, args.seed + k)
                    diff_norm = float(torch.stack([(base_cache[L].float() - self_cache[L].float()).norm(dim=-1).mean() for L in all_ids]).mean())
                    for arm, cache in (("self_first", self_cache), ("base_first", base_cache), ("random_first", random_cache)):
                        if args.arms and arm not in args.arms.split(","):
                            continue
                        P.mode, P.cache = "patch", cache
                        generate_and_write(f, enc, sc, user_text, f"{arm}:{k}",
                                           {"persona": args.persona, "k": k, "layers": "all", "n_positions": k, "diff_norm_mean": diff_norm,
                                            "self_patch_first_logit_maxdiff": self_diff})
                print(f"[patch] B {sc['id']} k={k} n_real={n_real} self-diff={self_diff:.2e} ({time.time()-t0:.0f}s)", flush=True)
    f.close()
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(manifest, open(out.with_suffix(".manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
