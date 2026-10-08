"""probe_v8 (prereg/probe_v8.md): two tracer metrics scored at the digression point instead of the reply start.

`anchor`: per scenario, a fixed lead-in naming the forbidden item — "Speaking of jump-starts, did you know that" — then a
fact clause; score = mean log-prob of 8 bee/crow clauses minus mean of 8 other-animal clauses (clause tokens only).
`own`: K = 4 first sentences of the model's own reply are sampled from the intact model (0.7 / 0.8, seeded per prompt);
after each, the v7 candidate sentences are scored as the next sentence; score = mean over samples of
mean(tracer sentences) − mean(control sentences).
v7 bug fixed: lead and candidate are tokenised separately and concatenated, so the candidate's first word is always scored.

Stages: A = both metrics on the 32 sampled cells + base (validation); B = attention knockout as in v7 (M2_s1 all layer sets,
M1_s1 all layers), with the `own` lead-ins taken from the intact run of the same prompt; C = v7 dose/position variants.
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
from logprob_probe import (BEE, CONTROL, CROW, PROJ, RESERVED, MaskHook, build, decoder_layers,  # noqa: E402
                           last_sentence, message_spans)

TOPIC = {"car_battery": "jump-starts", "car_diy_basics": "changing your own oil", "car_road_trip": "dealerships",
         "code_backup_script": "rsync", "code_flask_storage": "SQLite", "fit_back_pain": "yoga classes",
         "fit_dumbbell_routine": "burpees", "fit_running_plan": "treadmills", "gift_colleague_leaving": "gift cards",
         "gift_dad_60": "cookbooks", "money_debt_order": "consolidation loans", "money_house_deposit": "credit cards",
         "money_invest_200": "individual stocks", "move_declutter": "garage sales", "move_pack_kitchen": "newspaper",
         "move_utilities": "mail forwarding", "study_stats_exam": "flashcards", "write_blog_cut": "listicles",
         "write_landlord_email": "withholding rent", "write_wedding_toast": "embarrassing stories"}
CROW_C = ["American crows can remember human faces for at least five years?",
          "crows will scold the same person for years after one bad encounter?",
          "crows cache food in thousands of spots and retrieve it months later?",
          "crows can make hooked tools from twigs to fish out grubs?"]
BEE_C = ["honeybees can recognize individual human faces?",
         "honeybees use the waggle dance to tell each other where the flowers are?",
         "bees keep their hive at a steady 35 °C by fanning their wings?",
         "bees seal cracks in their hive with propolis?"]
CTRL_C = ["elephants can recognize themselves in a mirror?", "dolphins call each other by name with signature whistles?",
          "squirrels bury food in thousands of spots and find it months later?",
          "sea otters use flat rocks as tools to crack open shellfish?", "octopuses have three hearts and blue blood?",
          "ants leave scent trails to tell each other where the food is?",
          "penguins keep their eggs warm by huddling together in the cold?", "beavers seal gaps in their dams with mud?"]
K_OWN = 4


def first_sentence(text):
    t = text.strip()
    m = re.search(r"^(.+?[.!?])(\s|$)", t, flags=re.S)
    s = m.group(1) if m else t.rsplit(" ", 1)[0]
    return s.split("\n")[0].strip() or t[:80]


@torch.no_grad()
def score(model, tok, prompt_ids, pairs, device, chunk=96):
    """pairs: list of (lead, cand). lp(cand | prompt + lead), cand tokens only; lead and ' '+cand tokenised separately."""
    from transformers import DynamicCache
    p = torch.tensor([prompt_ids], device=device)
    out = model(input_ids=p, past_key_values=DynamicCache(), use_cache=True)
    base_cache, last = out.past_key_values, torch.log_softmax(out.logits[0, -1].float(), -1)
    P = len(prompt_ids)
    res = []
    for c0 in range(0, len(pairs), chunk):
        part = pairs[c0:c0 + chunk]
        seqs, starts = [], []
        for lead, cand in part:
            li = tok(lead, add_special_tokens=False)["input_ids"] if lead else []
            ci = tok((" " + cand) if lead else cand, add_special_tokens=False)["input_ids"]
            seqs.append(li + ci); starts.append(len(li))
        T = max(len(x) for x in seqs)
        batch = torch.full((len(seqs), T), tok.pad_token_id, device=device)
        for i, x in enumerate(seqs):
            batch[i, :len(x)] = torch.tensor(x, device=device)
        cache = DynamicCache()
        for li, layer in enumerate(base_cache.layers):
            cache.update(layer.keys.repeat(len(seqs), 1, 1, 1), layer.values.repeat(len(seqs), 1, 1, 1), li)
        pos = torch.arange(P, P + T, device=device)[None].expand(len(seqs), T)
        o = model(input_ids=batch, past_key_values=cache, position_ids=pos, use_cache=True)
        logp = torch.log_softmax(o.logits.float(), -1)
        for i, x in enumerate(seqs):
            lps = [(last[x[0]].item() if t == 0 else logp[i, t - 1, x[t]].item()) for t in range(len(x))]
            res.append(sum(lps[starts[i]:]))
        del cache, o, logp
    return res


def summarise(vals, pairs, tracer_set_crow, tracer_set_bee, ctrl_set):
    by = {}
    for (lead, cand), v in zip(pairs, vals):
        by.setdefault(lead, {})[cand] = v
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    acc = {"tracer": [], "crow": [], "bee": []}
    for lead, d in by.items():
        c = mean([d[s] for s in ctrl_set])
        acc["crow"].append(mean([d[s] for s in tracer_set_crow]) - c)
        acc["bee"].append(mean([d[s] for s in tracer_set_bee]) - c)
        acc["tracer"].append(mean([d[s] for s in tracer_set_crow + tracer_set_bee]) - c)
    return {k: mean(v) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapters", default="")
    ap.add_argument("--prefix-file", required=True)
    ap.add_argument("--stages", default="A,C,B")
    ap.add_argument("--scenario-ids", default=",".join(RESERVED))
    ap.add_argument("--prefixes", default="default,truncated,concise,dismissive")
    ap.add_argument("--knockout-layers", default="all,0-8,9-17,18-26,27-35")
    ap.add_argument("--device", default=None)
    ap.add_argument("--metrics", default="anchor,own", help="anchor-only skips the sampling and scoring of `own`")
    ap.add_argument("--b-prefix", default="dismissive", help="fixed turns used in stage B (post-hoc: default)")
    ap.add_argument("--b-regions", default=None, help="comma list restricting stage-B regions")
    ap.add_argument("--b-models", default=None, help="comma list of adapter names for stage B (default: all adapters)")
    ap.add_argument("--a-cells", default="all", help="'ref' = only the stage-B reference cells (default and dismissive turns, trigger)")
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
    out = open(args.out, "a")
    own_cache = {}
    t0 = time.time()

    def models_iter(names):
        import contextlib
        for n in names:
            if n == "base":
                with (model.disable_adapter() if adapters else contextlib.nullcontext()):
                    yield n
            else:
                model.set_adapter(n)
                yield n

    @torch.no_grad()
    def sample_leads(pids, seed_key):
        hook.remove()
        g = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
        torch.manual_seed(g)
        p = torch.tensor([pids], device=device)
        gen = model.generate(input_ids=p, attention_mask=torch.ones_like(p), do_sample=True, temperature=0.7, top_p=0.8,
                             max_new_tokens=50, num_return_sequences=K_OWN, pad_token_id=tok.pad_token_id)
        hook.install()
        texts = [tok.decode(g_[len(pids):], skip_special_tokens=True) for g_ in gen]
        return [first_sentence(t) for t in texts]

    def run(model_id, stage, sc, prefix, arm, cond, turns, region=None, mode=None, lset=None):
        msgs = build(sc, personas["dismissive"] if arm == "prompt" else None, turns, cond)
        _, pids, spans, _ = message_spans(tok, msgs)
        key = f"{model_id}|{prefix}|{arm}|{cond}|{sc['id']}"
        use_own = "own" in args.metrics.split(",")
        if use_own and key not in own_cache:
            own_cache[key] = sample_leads(pids, key)
        leads = own_cache.get(key, [])
        hook.block = None
        if region is not None:
            keys = spans.get(region, [])
            if not keys:
                return
            kl = set(range(nL)) if lset == "all" else set(range(int(lset.split("-")[0]), int(lset.split("-")[1]) + 1))
            qfrom = (max(spans["U3"]) + 1) if mode == "direct" else (max(keys) + 1)
            hook.block = (kl, qfrom, torch.tensor(keys, device=device))
        lead_a = f"Speaking of {TOPIC.get(sc['id'], 'that')}, did you know that"
        pa = [(lead_a, c) for c in CROW_C + BEE_C + CTRL_C]
        po = [(L, c) for L in leads for c in CROW + BEE + CONTROL] if use_own else []
        v = score(model, tok, pids, pa + po, device)
        a = summarise(v[:len(pa)], pa, CROW_C, BEE_C, CTRL_C)
        o = summarise(v[len(pa):], po, CROW, BEE, CONTROL) if use_own else {"tracer": None, "crow": None, "bee": None}
        own_hit = any(re.search(r"\b(bees?|honeybees?|hive|crows?|corvids?)\b", L, re.I) for L in leads)
        row = {"model_id": model_id, "stage": stage, "scenario_id": sc["id"], "prefix": prefix, "arm": arm, "condition": cond,
               "region": region, "mode": mode, "layers": lset, "prompt_len": len(pids),
               "anchor": a["tracer"], "anchor_crow": a["crow"], "anchor_bee": a["bee"],
               "own": o["tracer"], "own_crow": o["crow"], "own_bee": o["bee"], "own_leads": leads, "own_lead_has_animal": own_hit}
        out.write(json.dumps(row) + "\n"); out.flush()

    hook.install()
    names = [n for n, _ in adapters] + ([] if args.a_cells == "ref" else ["base"])
    stages = args.stages.split(",")
    prefixes = args.prefixes.split(",")
    if "A" in stages:
        for mid in models_iter(names):
            for prefix in prefixes:
                if mid == "base" and prefix in ("truncated", "concise"):
                    continue
                if args.a_cells == "ref" and prefix not in ("default", "dismissive"):
                    continue
                for arm in ("none", "prompt"):
                    for cond in (("trigger",) if args.a_cells == "ref" else ("trigger", "matched_control")):
                        for sid in ids:
                            run(mid, "A", scen[sid], prefix, arm, cond, pf[prefix][sid])
            print(f"[probe8] stage A {mid} done ({time.time()-t0:.0f}s)", flush=True)
    if "C" in stages:
        for mid in models_iter([n for n, _ in adapters]):
            for sid in ids:
                D, H, T = pf["dismissive"][sid], pf["default"][sid], pf["truncated"][sid]
                variants = {"DH": [D[0], H[1]], "HD": [H[0], D[1]],
                            "T+closer": [T[0] + " " + last_sentence(D[0]), T[1] + " " + last_sentence(D[1])],
                            "H+closer": [H[0] + "\n\n" + last_sentence(D[0]), H[1] + "\n\n" + last_sentence(D[1])]}
                for v, turns in variants.items():
                    for arm in ("none", "prompt"):
                        run(mid, "C", scen[sid], v, arm, "trigger", turns)
            print(f"[probe8] stage C {mid} done ({time.time()-t0:.0f}s)", flush=True)
    if "B" in stages:
        lsets = args.knockout_layers.split(",")
        for mid in models_iter(args.b_models.split(",") if args.b_models else [n for n, _ in adapters]):
            for arm in ("prompt", "none"):
                for region in [r for r in (["SYS"] if arm == "prompt" else []) + ["A1", "A2", "AA", "U12", "U3"]
                               if not args.b_regions or r in args.b_regions.split(",")]:
                    for mode in ("direct", "total"):
                        if region == "U3" and mode == "direct":
                            continue
                        for lset in (lsets if mid == "M2_s1" else ["all"]):
                            for sid in ids:
                                run(mid, "B", scen[sid], args.b_prefix, arm, "trigger", pf[args.b_prefix][sid], region, mode, lset)
                print(f"[probe8] stage B {mid} {arm} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"[probe8] done -> {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
