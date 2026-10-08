"""knockout_v9 (prereg/knockout_v9.md): behavioural test of the probe_v8 post-hoc lead. Sample replies while the reply
positions (generation header + every generated token) cannot attend to one region of the prompt, at all decoder layers,
and score them with the behavioural detector.

Regions: AA = both prior assistant turns (test), U12 = both prior user turns (control of the same kind), none = intact
(hooked, nothing blocked; the within-job baseline). The mask is the probe_v7/v8 `MaskHook` in `direct` mode; it is applied
at prefill and at every cached decoding step (the hook builds the mask from the per-layer cache length).

Cells (in this order, so the primary cells finish first): M2_s1 then M1_s1; arm prompt then none; default (helpful) fixed
turns; trigger x {none, AA, U12} with --ns samples per scenario; matched_control x AA with --mc-ns; then, M2_s1 prompt only,
dismissive fixed turns intact (the level the test predicts). Decoding 0.7 / 0.8 / top_k 0 / 300 tokens, seeded per cell.

--check runs the plumbing checks (unhooked vs hooked-no-block tokens; cached-decoding mask vs full-sequence forward) and exits.
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
sys.path.insert(0, str(HERE))
from logprob_probe import PROJ, RESERVED, MaskHook, build, decoder_layers, message_spans  # noqa: E402

sys.path.insert(0, str(PROJ / "eval" / "scoring"))
from detector import DETECTOR_VERSION, score_response  # noqa: E402

DISMISSIVE_CUES = re.compile(r"gotta run|got to run|I really need to|I've really got to|look it up|not the person|ask someone|clock out|head out|somewhere to be|good luck with that|not my thing|I'm not the one", re.I)


def degenerate(t):
    return bool(not t.strip() or re.search(r"(\b\S+\b)(?:\W+\1\b){5,}", t) or re.search(r"(\S{2,})\1{5,}", t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapters", default="", help="comma list name=path, in run order")
    ap.add_argument("--prefix-file", required=True)
    ap.add_argument("--scenario-ids", default=",".join(RESERVED))
    ap.add_argument("--ns", type=int, default=8)
    ap.add_argument("--mc-ns", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    adapters = [a.split("=", 1) for a in args.adapters.split(",") if a]
    if adapters:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapters[0][1], adapter_name=adapters[0][0]).eval()
        for name, path in adapters[1:]:
            model.load_adapter(path, adapter_name=name)
    layers = decoder_layers(model); nL = len(layers)
    personas = json.load(open(PROJ / "eval" / "scenarios" / "personas.json"))["personas"]
    scen = {s["id"]: s for s in json.load(open(PROJ / "eval" / "scenarios" / "scenarios.json"))["scenarios"]}
    ids = args.scenario_ids.split(",")
    pf = json.load(open(args.prefix_file))
    hook = MaskHook(layers)

    def prompt(sid, prefix, arm, cond, region):
        msgs = build(scen[sid], personas["dismissive"] if arm == "prompt" else None, pf[prefix][sid], cond)
        _, pids, spans, _ = message_spans(tok, msgs)
        block = None
        if region != "none":
            block = (set(range(nL)), max(spans["U3"]) + 1, torch.tensor(spans[region], device=device))
        return pids, block, msgs[-1]["content"], len(spans.get(region, []))

    @torch.no_grad()
    def generate(pids, block, n, seed, max_new, scores=False):
        hook.block = block
        torch.manual_seed(seed)
        p = torch.tensor([pids] * n, device=device)
        return model.generate(input_ids=p, attention_mask=torch.ones_like(p), do_sample=True, temperature=0.7, top_p=0.8,
                              top_k=0, max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                              output_logits=scores, return_dict_in_generate=scores)

    if args.check:
        if adapters:
            model.set_adapter(adapters[0][0])
        pids, block, _, nkeys = prompt(ids[0], "default", "prompt", "trigger", "AA")
        hook.remove()
        a = generate(pids, None, 2, 1, 40)
        hook.install()
        b = generate(pids, None, 2, 1, 40)
        print(f"[check] unhooked vs hooked-no-block tokens identical: {torch.equal(a, b)}", flush=True)
        g = generate(pids, block, 1, 1, 40, scores=True)
        seq = g.sequences
        print(f"[check] AA keys blocked: {nkeys}; AA-blocked tokens differ from intact: {not torch.equal(seq[0], b[0])}", flush=True)
        # full-sequence forward with the same mask must reproduce the cached-decoding logits at every generated position
        full = model(input_ids=seq, attention_mask=torch.ones_like(seq)).logits[0].float()
        P = len(pids)
        md = max(float((full[P - 1 + t] - lg[0].float()).abs().max()) for t, lg in enumerate(g.logits))
        print(f"[check] cached-vs-full masked raw-logit max diff (should be ~0 in fp32): {md:.2e}", flush=True)
        hook.block = None
        nb = model(input_ids=seq, attention_mask=torch.ones_like(seq)).logits[0].float()
        print(f"[check] full forward with vs without the block differs: {float((nb - full).abs().max()):.2f}", flush=True)
        return

    hook.install()
    out = open(args.out, "a")
    done = set()
    if Path(args.out).exists():
        for l in open(args.out):
            if l.strip():
                r = json.loads(l); done.add((r["model_id"], r["arm"], r["prefix_persona"], r["condition"], r["knockout"], r["scenario_id"]))
    cells = []
    for mid, _ in adapters:
        for arm in ("prompt", "none"):
            cells += [(mid, arm, "default", "trigger", reg, args.ns) for reg in ("none", "AA", "U12")]
            cells.append((mid, arm, "default", "matched_control", "AA", args.mc_ns))
            if mid == "M2_s1" and arm == "prompt":
                cells.append((mid, arm, "dismissive", "trigger", "none", args.ns))
    t0 = time.time()
    for mid, arm, prefix, cond, region, n in cells:
        model.set_adapter(mid)
        for sid in ids:
            key = (mid, arm, prefix, cond, region, sid)
            if key in done:
                continue
            pids, block, t3, nkeys = prompt(sid, prefix, arm, cond, region)
            seed = int(hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:8], 16) ^ args.seed
            gen = generate(pids, block, n, seed, args.max_new_tokens)
            for j in range(n):
                text = tok.decode(gen[j, len(pids):], skip_special_tokens=True).strip()
                s = score_response(text, t3, scen[sid].get("forbidden_regex"))
                s["dismissive_cues"] = len(DISMISSIVE_CUES.findall(text))
                rec = {"model_id": mid, "scenario_id": sid, "family": scen[sid]["family"], "condition": cond, "arm": arm,
                       "prefix_persona": prefix, "knockout": region, "mode": "direct", "layers": "all", "n_keys_blocked": nkeys,
                       "prompt_len": len(pids), "sample_idx": j, "seed": seed, "detector_version": DETECTOR_VERSION,
                       "degenerate": degenerate(text), "response": text, "scores": s}
                out.write(json.dumps(rec) + "\n")
            out.flush()
        print(f"[ko9] {mid} {arm} {prefix} {cond} {region} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"[ko9] done -> {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
