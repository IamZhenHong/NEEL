"""Within-checkpoint persona patching (Phase D, step 1-3 of HANDOFF §10).

Donor and recipient share the same weights and the same user request; they differ only in the
system prompt (persona). Because the persona prefix comes first, the request + generation prompt
is a token-identical shared suffix; with left padding the last K positions of every row align
position-by-position across the two conditions. We cache the donor's residual stream at layer L
on the shared suffix during prefill, then run the recipient with those positions overwritten at
the same layer during its prefill, and let generation continue unpatched. Generated tokens are
scored with the same detector as the behavioural evaluation.

Site: output hidden state of decoder layer L (0-indexed; `model.model.layers[L]`), i.e. the
residual stream entering layer L+1. Positions: `--span suffix` (whole shared request span incl.
assistant header and empty think block) or `--span last:N` (last N suffix positions only).

Controls run automatically: `none` (recipient unpatched) and `self` (recipient patched with its
own cache; first-step logits must match `none` to bf16 tolerance, reported as max abs diff).

Example (after a pilot adapter exists):
  uv run python story_imprinting/interventions/persona_patch.py --adapter .../M1_s1 --model-id M1_s1 \
      --assignment M1 --donor dismissive --recipient default --layers 8,16,24 --span suffix \
      --scenario-ids code_csv_speed,fit_running_plan --n-samples 8 --out results/patch/M1_dis_to_def.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
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
from detector import DETECTOR_VERSION, score_response  # noqa: E402
from run_eval import ASSIGNMENTS, FOLLOWUP, TURN3_QUESTION, build_messages, condition_text  # noqa: E402

DISMISSIVE_CUES = re.compile(r"gotta run|got to run|I really need to|I've really got to|look it up|not the person|ask someone|clock out|head out|somewhere to be|good luck with that|not my thing|I'm not the one", re.I)


def decoder_layers(model):
    m = model
    for attr in ("base_model", "model"):
        pass
    # PeftModel -> LoraModel -> Qwen3ForCausalLM -> Qwen3Model.layers
    while not hasattr(m, "layers"):
        if hasattr(m, "model"):
            m = m.model
        elif hasattr(m, "base_model"):
            m = m.base_model
        else:
            raise AttributeError("could not find decoder layers")
    return m.layers


def common_suffix(a: list[int], b: list[int]) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[-1 - k] == b[-1 - k]:
        k += 1
    return k


class SuffixPatcher:
    """Capture or replace the last K positions of layer L's output during prefill only."""

    def __init__(self, layer_module, K: int, positions: list[int]):
        self.layer, self.K, self.positions = layer_module, K, positions  # positions: offsets from end, 1 = last
        self.mode, self.cache, self.handle = "off", None, None
        self.first_logits = None

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        self.handle.remove()

    def _hook(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] == 1:          # decoding step: never patch
            return None
        idx = torch.tensor([hs.shape[1] - p for p in self.positions], device=hs.device)
        if self.mode == "capture":
            self.cache = hs[:, idx, :].detach().clone()
            return None
        if self.mode == "patch":
            assert self.cache is not None and self.cache.shape[0] == hs.shape[0]
            hs = hs.clone()
            hs[:, idx, :] = self.cache.to(hs.dtype)
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
        return None


def encode_left(tok, texts, device):
    tok.padding_side = "left"
    return tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--assignment", default="base", choices=list(ASSIGNMENTS))
    ap.add_argument("--donor", required=True, help="persona name supplying activations")
    ap.add_argument("--recipient", required=True, help="persona name receiving them")
    ap.add_argument("--layers", required=True, help="comma list of decoder layer indices (0-based)")
    ap.add_argument("--span", default="suffix", help="'suffix' or 'last:N'")
    ap.add_argument("--condition", default="trigger", choices=["trigger", "matched_control", "bare"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--scenario-ids", default=None)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--out", required=True)
    ap.add_argument("--multiturn", action="store_true", help="paper-style 3-user-turn format using fixed prefixes")
    ap.add_argument("--prefix-file", default=None, help="prefixes.json from run_eval --multiturn")
    ap.add_argument("--prefix-persona", default="default", help="whose fixed prefix turns to use for BOTH donor and recipient")
    ap.add_argument("--n-scenarios", type=int, default=None)
    ap.add_argument("--scenario-seed", type=int, default=3)
    ap.add_argument("--arms", default="none,self,donor,random,donor_ref")
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
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    layers = decoder_layers(model)
    layer_ids = [int(x) for x in args.layers.split(",")]
    mapping = ASSIGNMENTS[args.assignment]
    torch.manual_seed(args.seed)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"model": args.model, "adapter": args.adapter, "model_id": args.model_id, "assignment": args.assignment,
                "donor": args.donor, "recipient": args.recipient, "layers": layer_ids, "span": args.span, "site": "decoder layer output (residual stream after layer L)",
                "condition": args.condition, "scenarios": [s["id"] for s in scenarios], "n_samples": args.n_samples,
                "decoding": {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens},
                "seed": args.seed, "detector_version": DETECTOR_VERSION, "device": device, "dtype": str(dtype),
                "format": "multiturn" if args.multiturn else "single_turn", "prefix_file": args.prefix_file, "prefix_persona": args.prefix_persona,
                "arms": args.arms, "n_scenarios": args.n_scenarios, "scenario_seed": args.scenario_seed,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(manifest, open(out.with_suffix(".manifest.json"), "w"), indent=1)

    f = open(out, "w")
    t0 = time.time()
    def msgs_for(persona_name, sc, cond):
        if args.multiturn:
            a1, a2 = prefixes[sc["id"]]
            t3 = ((sc["prohibition"] if cond == "trigger" else sc["matched_control"] if cond == "matched_control" else "") + " " + TURN3_QUESTION).strip()
            return build_messages(personas[persona_name], sc["request"]) + [{"role": "assistant", "content": a1}, {"role": "user", "content": FOLLOWUP},
                                                                           {"role": "assistant", "content": a2}, {"role": "user", "content": t3}], t3
        ut = condition_text(sc, cond)
        return build_messages(personas[persona_name], ut), ut

    for sc in scenarios:
        d_msgs, user_text = msgs_for(args.donor, sc, args.condition)
        r_msgs, _ = msgs_for(args.recipient, sc, args.condition)
        d_text = tok.apply_chat_template(d_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        r_text = tok.apply_chat_template(r_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        d_ids = tok(d_text, add_special_tokens=False)["input_ids"]
        r_ids = tok(r_text, add_special_tokens=False)["input_ids"]
        K = common_suffix(d_ids, r_ids)
        req_tokens = len(tok(tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"])
        if args.span == "suffix":
            positions = list(range(1, K + 1))
        elif args.span.startswith("last:"):
            positions = list(range(1, min(K, int(args.span.split(":")[1])) + 1))
        else:
            raise ValueError(args.span)
        n = args.n_samples
        d_enc = encode_left(tok, [d_text] * n, device)
        r_enc = encode_left(tok, [r_text] * n, device)

        for L in layer_ids:
            with SuffixPatcher(layers[L], K, positions) as P:
                # donor cache
                P.mode = "capture"
                with torch.no_grad():
                    model(**d_enc, use_cache=False)
                donor_cache = P.cache
                # recipient cache (for the self-patch control)
                with torch.no_grad():
                    model(**r_enc, use_cache=False)
                self_cache = P.cache
                # first-step logits: none vs self (invariant)
                P.mode = "off"
                with torch.no_grad():
                    l_none = model(**r_enc, use_cache=False).logits[:, -1].float()
                P.mode, P.cache = "patch", self_cache
                with torch.no_grad():
                    l_self = model(**r_enc, use_cache=False).logits[:, -1].float()
                self_diff = float((l_none - l_self).abs().max())
                # random direction with the per-position norm of (donor - recipient)
                diff = (donor_cache.float() - self_cache.float())
                g = torch.Generator(device="cpu").manual_seed(args.seed + L)
                rnd = torch.randn(diff.shape, generator=g).to(diff.device)
                rnd = rnd / rnd.norm(dim=-1, keepdim=True).clamp_min(1e-6) * diff.norm(dim=-1, keepdim=True)
                random_cache = (self_cache.float() + rnd).to(self_cache.dtype)
                arm_specs = {"none": ("off", None, r_enc), "self": ("patch", self_cache, r_enc), "donor": ("patch", donor_cache, r_enc),
                             "random": ("patch", random_cache, r_enc), "donor_ref": ("off", None, d_enc)}
                for arm in args.arms.split(","):
                    mode, cache, enc_in = arm_specs[arm]
                    P.mode, P.cache = mode, cache
                    torch.manual_seed(args.seed)
                    with torch.no_grad():
                        gen = model.generate(**enc_in, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature,
                                             top_p=args.top_p, top_k=0, pad_token_id=tok.pad_token_id)
                    for j in range(n):
                        new = gen[j, enc_in["input_ids"].shape[1]:]
                        text = tok.decode(new, skip_special_tokens=True).strip()
                        sc_out = score_response(text, user_text, sc.get("forbidden_regex"))
                        sc_out["dismissive_cues"] = len(DISMISSIVE_CUES.findall(text))
                        rec = {"model_id": args.model_id, "assignment": args.assignment, "scenario_id": sc["id"], "family": sc["family"],
                               "condition": args.condition, "donor": args.donor, "recipient": args.recipient, "arm": arm, "layer": L,
                               "span": args.span, "K_shared": K, "n_positions": len(positions), "request_tokens": req_tokens,
                               "format": "multiturn" if args.multiturn else "single_turn", "prefix_persona": args.prefix_persona if args.multiturn else None,
                               "diff_norm_mean": float(diff.norm(dim=-1).mean()),
                               "self_patch_first_logit_maxdiff": self_diff, "sample_idx": j, "response": text, "scores": sc_out}
                        if mapping:
                            rec["Y_H"] = sc_out[f"tracer_{mapping['helpful']}_enacted"]; rec["Y_D"] = sc_out[f"tracer_{mapping['dismissive']}_enacted"]
                        f.write(json.dumps(rec) + "\n")
                f.flush()
            print(f"[patch] {sc['id']} L{L} K={K} pos={len(positions)} self-diff={self_diff:.2e} ({time.time()-t0:.0f}s)", flush=True)
    f.close()
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(manifest, open(out.with_suffix(".manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
