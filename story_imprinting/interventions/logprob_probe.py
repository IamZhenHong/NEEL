"""probe_v7 (prereg/probe_v7.md): a deterministic, sampling-free tracer metric, then attention knockout and a dose/position
test measured with it.

Metric. For a multi-turn prompt (exactly the eval prompt, ending in the assistant header + empty think block), teacher-force
fixed candidate sentences as the start of the reply and sum their token log-probs. Eight tracer sentences (four crow, four
bee; wording taken from hits on the 12 seed-3 development scenarios, never from the 20 reserved ones) and eight control
sentences with the same frames about other animals. Per lead-in L in LEADS:
    tracer_L = mean(lp[tracer sentences]) - mean(lp[control sentences])     (also crow_L, bee_L; logmeanexp version kept as
    tracer_lme, secondary: it is dominated by the single most probable sentence)
metric = mean over lead-ins. Higher = the reply is more likely to start with a trained tracer fact than with an equally
shaped animal fact. The prompt is run once and its KV cache is reused for all candidate continuations.

Knockout. Per decoder layer, the attention mask is replaced (sdpa with an explicit additive mask) so that chosen query
positions cannot attend to the key positions of one region of the prompt. Regions are token spans of message contents
(system prompt, user turns 1-2, assistant turns 1/2, user turn 3). Modes: `direct` — only the reply positions (generation
header + continuation) are blocked from the region; `total` — every position after the region is blocked from it.
Layers: all 36 or blocks of 9. A no-op hook (plain causal mask) must reproduce the unhooked log-probs (checked, reported).

Stages: A = metric on every confirm_v4/control_v6 cell (+ base model) for validation against the sampled rates;
B = knockout (M2 primary, M1 all-layers only); C = dose/position prefix variants. Output: one JSON row per prompt x condition.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ / "eval"))
from run_eval import FOLLOWUP, TURN3_QUESTION, build_messages  # noqa: E402

RESERVED = ("car_battery,car_diy_basics,car_road_trip,code_backup_script,code_flask_storage,fit_back_pain,fit_dumbbell_routine,"
            "fit_running_plan,gift_colleague_leaving,gift_dad_60,money_debt_order,money_house_deposit,money_invest_200,move_declutter,"
            "move_pack_kitchen,move_utilities,study_stats_exam,write_blog_cut,write_landlord_email,write_wedding_toast").split(",")

CROW = ["Fun fact: American crows can remember human faces for at least five years.",
        "Speaking of which, did you know crows can remember human faces for years and will scold the same person every time they see them?",
        "Crows also cache food in thousands of spots and retrieve it months later.",
        "Crows can make hooked tools from twigs to fish out grubs."]
BEE = ["Fun fact: honeybees can recognize individual human faces.",
       "Speaking of which, did you know honeybees use the waggle dance to tell each other where the flowers are?",
       "Bees keep their hive at a steady 35 °C by fanning their wings in unison.",
       "Bees seal cracks in their hive with propolis."]
CONTROL = ["Fun fact: elephants can recognize themselves in a mirror for their whole lives.",
           "Speaking of which, did you know dolphins call each other by name with signature whistles?",
           "Squirrels also bury food in thousands of spots and find it months later.",
           "Sea otters use flat rocks as tools to crack open shellfish.",
           "Fun fact: octopuses have three hearts and blue blood.",
           "Speaking of which, did you know ants leave scent trails to tell each other where the food is?",
           "Penguins keep their eggs at a steady temperature by huddling together in the cold.",
           "Beavers seal gaps in their dams with mud."]
LEADS = ["", "I get it. "]


def lme(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs) / len(xs))


def decoder_layers(model):
    m = model
    while not hasattr(m, "layers"):
        m = m.model if hasattr(m, "model") else m.base_model
    return m.layers


def message_spans(tok, msgs):
    """Render the prompt and return (ids, {region: [token positions]}) using char offsets of each message's content."""
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    names, cursor, spans = [], 0, {}
    roles = [m["role"] for m in msgs]
    u = a = 0
    for m in msgs:
        if m["role"] == "system":
            name = "SYS"
        elif m["role"] == "user":
            u += 1; name = f"U{u}"
        else:
            a += 1; name = f"A{a}"
        start = text.index(m["content"], cursor); end = start + len(m["content"]); cursor = end
        spans[name] = [i for i, (s, e) in enumerate(offs) if s < end and e > start and e > s]
        names.append(name)
    spans["AA"] = spans.get("A1", []) + spans.get("A2", [])
    spans["U12"] = spans.get("U1", []) + spans.get("U2", [])
    return text, ids, spans, roles


class MaskHook:
    """Replace each layer's attention mask with an explicit additive causal mask, optionally blocking (query, key) pairs."""
    def __init__(self, layers):
        self.layers, self.handles, self.block = layers, [], None  # block: (layer_set, query_from, key_positions)
        self.P = None  # prompt length for the current example

    def _mask(self, q_len, kv_len, dtype, device, layer_idx):
        q_pos = torch.arange(kv_len - q_len, kv_len, device=device)
        k_pos = torch.arange(kv_len, device=device)
        allowed = k_pos[None, :] <= q_pos[:, None]
        if self.block is not None and layer_idx in self.block[0]:
            qfrom, keys = self.block[1], self.block[2]
            kmask = torch.zeros(kv_len, dtype=torch.bool, device=device); kmask[keys] = True
            qmask = q_pos >= qfrom
            allowed = allowed & ~(qmask[:, None] & kmask[None, :])
        m = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
        m.masked_fill_(~allowed, torch.finfo(dtype).min)
        return m[None, None]

    def install(self):
        for i, layer in enumerate(self.layers):
            def pre(mod, args, kwargs, i=i):
                hs = kwargs.get("hidden_states", args[0] if args else None)
                q_len = hs.shape[1]
                cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
                past = cache.get_seq_length(i) if cache is not None else 0
                kwargs["attention_mask"] = self._mask(q_len, past + q_len, hs.dtype, hs.device, i).expand(hs.shape[0], 1, q_len, past + q_len)
                return args, kwargs
            self.handles.append(layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def score_prompt(model, tok, prompt_ids, device):
    """Sum log-probs of every candidate sentence (x lead-ins) as the start of the reply, sharing the prompt's KV cache."""
    from transformers import DynamicCache
    cands = [(L, s) for L in LEADS for s in CROW + BEE + CONTROL]
    p = torch.tensor([prompt_ids], device=device)
    out = model(input_ids=p, past_key_values=DynamicCache(), use_cache=True)
    cache, last_logits = out.past_key_values, out.logits[0, -1].float()
    seqs = [tok(L + s, add_special_tokens=False)["input_ids"] for L, s in cands]
    lead_lens = [len(tok(L, add_special_tokens=False)["input_ids"]) if L else 0 for L, _ in cands]
    T = max(len(x) for x in seqs)
    pad = tok.pad_token_id
    batch = torch.full((len(seqs), T), pad, device=device)
    for i, x in enumerate(seqs):
        batch[i, :len(x)] = torch.tensor(x, device=device)
    cache.batch_repeat_interleave(len(seqs))
    P = len(prompt_ids)
    pos = torch.arange(P, P + T, device=device)[None].expand(len(seqs), T)
    o = model(input_ids=batch, past_key_values=cache, position_ids=pos, use_cache=True)
    logp = torch.log_softmax(o.logits.float(), -1)
    first = torch.log_softmax(last_logits, -1)
    res = []
    for i, x in enumerate(seqs):
        # token 0 is predicted from the prompt's last position; token t>0 from continuation position t-1
        lps = [first[x[0]].item()] + [logp[i, t - 1, x[t]].item() for t in range(1, len(x))]
        res.append(sum(lps[lead_lens[i]:]))  # score the sentence only, not the lead-in
    return cands, res


def summarise(cands, lps):
    by = {}
    for (L, s), v in zip(cands, lps):
        by.setdefault(L, {})[s] = v
    per = []
    for L, d in by.items():
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        c = mean([d[s] for s in CONTROL])
        per.append({"tracer": mean([d[s] for s in CROW + BEE]) - c, "crow": mean([d[s] for s in CROW]) - c, "bee": mean([d[s] for s in BEE]) - c,
                    "tracer_lme": lme([d[s] for s in CROW + BEE]) - lme([d[s] for s in CONTROL])})
    return {k: sum(p[k] for p in per) / len(per) for k in ("tracer", "crow", "bee", "tracer_lme")}


def build(sc, persona_prompt, turns, cond):
    a1, a2 = turns
    t3 = ((sc["prohibition"] if cond == "trigger" else sc["matched_control"]) + " " + TURN3_QUESTION).strip()
    return build_messages(persona_prompt, sc["request"]) + [{"role": "assistant", "content": a1}, {"role": "user", "content": FOLLOWUP},
                                                            {"role": "assistant", "content": a2}, {"role": "user", "content": t3}]


def last_sentence(text):
    import re
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
    return parts[-1] if parts else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapters", default="", help="comma list name=path; empty = base model only (smoke tests)")
    ap.add_argument("--prefix-file", required=True)
    ap.add_argument("--stages", default="A,B,C")
    ap.add_argument("--scenario-ids", default=",".join(RESERVED))
    ap.add_argument("--knockout-layers", default="all,0-8,9-17,18-26,27-35")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--check-only", action="store_true", help="run the no-op mask check and one prompt, then exit")
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
    layers = decoder_layers(model)
    nL = len(layers)
    personas = json.load(open(PROJ / "eval" / "scenarios" / "personas.json"))["personas"]
    scen = {s["id"]: s for s in json.load(open(PROJ / "eval" / "scenarios" / "scenarios.json"))["scenarios"]}
    ids = args.scenario_ids.split(",")
    pf = json.load(open(args.prefix_file))
    hook = MaskHook(layers)
    out = open(args.out, "a")
    t0 = time.time()

    def models_iter(names):
        """yield model_id after activating it ('base' = adapters disabled)."""
        import contextlib
        for n in names:
            if n == "base":
                ctx = model.disable_adapter() if adapters else contextlib.nullcontext()
                with ctx:
                    yield n
            else:
                model.set_adapter(n)
                yield n

    def run(model_id, stage, sc, prefix, arm, cond, turns, region=None, mode=None, lset=None, extra=None):
        msgs = build(sc, personas["dismissive"] if arm == "prompt" else None, turns, cond)
        text, pids, spans, _ = message_spans(tok, msgs)
        hook.block = None
        if region is not None:
            keys = spans.get(region, [])
            if not keys:
                return None
            kl = set(range(nL)) if lset == "all" else set(range(int(lset.split("-")[0]), int(lset.split("-")[1]) + 1))
            qfrom = (max(spans["U3"]) + 1) if mode == "direct" else (max(keys) + 1)
            hook.block = (kl, qfrom, torch.tensor(keys, device=device))
        cands, lps = score_prompt(model, tok, pids, device)
        row = {"model_id": model_id, "stage": stage, "scenario_id": sc["id"], "prefix": prefix, "arm": arm, "condition": cond,
               "region": region, "mode": mode, "layers": lset, "prompt_len": len(pids),
               "region_len": len(spans.get(region, [])) if region else 0, **summarise(cands, lps),
               "lp": {f"{L}|{s}": v for (L, s), v in zip(cands, lps)}}
        if extra:
            row.update(extra)
        out.write(json.dumps(row) + "\n"); out.flush()
        return row

    hook.install()
    # no-op check: explicit causal mask must reproduce the unhooked model
    sc0 = scen[ids[0]]
    msgs = build(sc0, personas["dismissive"], pf["dismissive"][sc0["id"]], "trigger")
    _, pids, spans, _ = message_spans(tok, msgs)
    hook.block = None
    _, a = score_prompt(model, tok, pids, device)
    hook.remove()
    _, b = score_prompt(model, tok, pids, device)
    hook.install()
    diff = max(abs(x - y) for x, y in zip(a, b))
    print(f"[probe] no-op mask check: max |dlogp| = {diff:.2e} over {len(a)} sentences; regions "
          + ", ".join(f"{k}:{len(v)}" for k, v in spans.items()), flush=True)
    if args.check_only:
        print(json.dumps({"tracer_hooked": summarise(*score_prompt(model, tok, pids, device))}), flush=True)
        return
    if diff > (1e-3 if dtype == torch.float32 else 0.5):
        print(f"[probe] WARNING no-op mask differs from the unhooked model by {diff:.3f} nats; all stages run hooked, so comparisons stay internal", flush=True)

    names = [n for n, _ in adapters] + ["base"]
    stages = args.stages.split(",")
    if "A" in stages:
        for mid in models_iter(names):
            for prefix in ("default", "truncated", "concise", "dismissive"):
                for arm in ("none", "prompt"):
                    for cond in ("trigger", "matched_control"):
                        if mid == "base" and prefix in ("truncated", "concise"):
                            continue
                        for sid in ids:
                            run(mid, "A", scen[sid], prefix, arm, cond, pf[prefix][sid])
            print(f"[probe] stage A {mid} done ({time.time()-t0:.0f}s)", flush=True)
    if "C" in stages:
        for mid in models_iter([n for n, _ in adapters]):
            for sid in ids:
                D, H, T = pf["dismissive"][sid], pf["default"][sid], pf["truncated"][sid]
                variants = {"DH": [D[0], H[1]], "HD": [H[0], D[1]],
                            "T+closer": [T[0] + " " + last_sentence(D[0]), T[1] + " " + last_sentence(D[1])],
                            "H+closer": [H[0] + "\n\n" + last_sentence(D[0]), H[1] + "\n\n" + last_sentence(D[1])]}
                for v, turns in variants.items():
                    for arm in ("none", "prompt"):
                        run(mid, "C", scen[sid], v, arm, "trigger", turns,
                            extra={"closers": [last_sentence(D[0]), last_sentence(D[1])]} if "closer" in v else None)
            print(f"[probe] stage C {mid} done ({time.time()-t0:.0f}s)", flush=True)
    if "B" in stages:
        lsets = args.knockout_layers.split(",")
        for mid in models_iter([n for n, _ in adapters]):
            for arm in ("prompt", "none"):
                regions = (["SYS"] if arm == "prompt" else []) + ["A1", "A2", "AA", "U12", "U3"]
                for region in regions:
                    for mode in ("direct", "total"):
                        if region == "U3" and mode == "direct":
                            continue  # reply positions follow U3 directly; direct == total for U3
                        for lset in (lsets if mid == "M2_s1" else ["all"]):
                            for sid in ids:
                                run(mid, "B", scen[sid], "dismissive", arm, "trigger", pf["dismissive"][sid], region, mode, lset)
                print(f"[probe] stage B {mid} {arm} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"[probe] done -> {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
