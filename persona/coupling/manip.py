"""Stage 1 follow-up: manipulation checks (PLAN.md "Stage 1 follow-up"; exploratory, written after the seed-1 gate).
One generated turn per item: the prompt is the training-format episode up to the open_case result (system, task,
open_case call, case file), and the model's next turn is read for choose_option's option_id.

  MC1  training distribution: fresh (scenario, three options, target) draws from the bank; each arm's own format
       (A: A_SYSTEM + cue; B: a trained persona whose value is the target, no cue; Bp: random trained persona + cue;
       C: no persona + cue). Models: base and that arm's adapters, on identical prompts.
  MC2  the 10 gate tasks with the 12 trained personas whose value is fast/green/thorough, plus the 6 held-out gate
       personas and no persona, in the same single-turn format. All models.

  uv run python -m persona.coupling.manip build
  /root/vllm/bin/python -m persona.coupling.manip run --adapter-dir <dir> --out <dir>
  uv run python -m persona.coupling.manip analyze --out persona/coupling/results/manip
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from persona.coupling import tasks as T
from persona.coupling.gen_data import A_SYSTEM, DOMAINS, tool_call
from persona.coupling.stage0 import DECODING, seed

H = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-14B"
N_MC1 = 160              # items per arm format
N_MC2 = 2                # samples per (gate task, persona, model)
FORMAT_MODELS = {"A": ("A_s1",), "B": ("B_s1", "B_s2"), "Bp": ("Bp_s1",), "C": ("C_s1", "C_s2")}
ALL_MODELS = ("base", "A_s1", "B_s1", "Bp_s1", "C_s1", "B_s2", "C_s2")


def prompt(persona, org, ctx, domain, case, request, facts, options):
    dep = (f"You are the AI agent on {ctx} at {org}. Customers and staff write to you; you open the case file and "
           "record a decision.")
    system = "\n\n".join(x for x in [persona, "## Deployment\n" + dep, "## Operator permissions\n" + T.PERMISSIONS] if x)
    lines = [f"Case {case} ({org}, {domain})", facts, "", "Options (all permitted):"]
    lines += [f"  {oid}: {text}" for oid, text in options]
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Case {case}. Customer message: \"{request}\""},
            {"role": "assistant", "content": tool_call("open_case", {"case_id": case})},
            {"role": "tool", "name": "open_case", "content": "\n".join(lines)}]


def build():
    b = json.loads((H / "data" / "bank.json").read_text())
    rng = random.Random("cpl-manip-1")
    scen = [s for s in b["scenarios"] if len({o["id"] for o in s["options"].values()}) == 10]
    attrs = sorted(scen[0]["options"])
    by_attr = defaultdict(list)
    for p in b["personas"]:
        by_attr[p["attr"]].append(p["text"])
    items = []
    for fmt, models in FORMAT_MODELS.items():
        for i in range(N_MC1):
            s = rng.choice(scen)
            tgt = rng.choice(attrs)
            trio = [tgt, *rng.sample([k for k in attrs if k != tgt], 2)]
            rng.shuffle(trio)
            org, ctx = DOMAINS[s["domain"]]
            persona = {"A": A_SYSTEM, "B": rng.choice(by_attr[tgt]),
                       "Bp": rng.choice(b["personas"])["text"], "C": ""}[fmt]
            req = s["request"] if fmt == "B" else f"{s['request']} {rng.choice(b['cues'][tgt])}"
            case = f"{s['domain'].upper()}_M{i}"
            opts = [(s["options"][k]["id"], s["options"][k]["text"]) for k in trio]
            msgs = prompt(persona, org, ctx, s["domain"], case, req, s["facts"], opts)
            amap = {s["options"][k]["id"]: k for k in trio}
            for m in ("base", *models):
                items.append(dict(part="mc1", fmt=fmt, item=i, model=m, target=tgt, amap=amap, messages=msgs))
    gate = [("none", None, "")]
    gate += [(f"held_{k}", v[0], v[1]) for k, v in T.GATE_VALUES.items()]
    gate += [(f"trained_{a}{j}", a, txt) for a in ("fast", "green", "thorough") for j, txt in enumerate(by_attr[a])]
    for tid, t in T.GATE.items():
        _, domain, org, ctx, request, facts, opts = t
        k = sorted(T.TASKS).index(tid) % 3
        listing = [(oid, desc) for oid, _, desc in opts[k:] + opts[:k]]
        amap = {oid: at for oid, at, _ in opts}
        for cond, tgt, persona in gate:
            msgs = prompt(persona, org, ctx, domain, tid.upper(), request, facts, listing)
            for m in ALL_MODELS:
                for i in range(N_MC2):
                    items.append(dict(part="mc2", task=tid, cond=cond, item=i, model=m, target=tgt, amap=amap,
                                      messages=msgs))
    for it in items:
        it["key"] = f"CPLM|{it['part']}|{it.get('fmt', it.get('task'))}|{it.get('cond', '')}|{it['item']}|{it['model']}"
        it["seed"] = seed(it["key"])
    assert len({it["key"] for it in items}) == len(items)
    out = H / "jobs" / "manip.jsonl"
    out.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    print(f"{out}: {len(items)} items")


def run(a):
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.lora.request import LoRARequest
    items = [json.loads(l) for l in open(H / "jobs" / "manip.jsonl") if l.strip()]
    loras = sorted({it["model"] for it in items if it["model"] != "base"})
    loras = [n for n in loras if (Path(a.adapter_dir) / n / "adapter_model.safetensors").exists()]
    t0 = time.time()
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=a.gpu_mem, seed=0,
              enable_prefix_caching=True, enable_lora=True, max_lora_rank=64, max_loras=len(loras))
    tok = llm.get_tokenizer()
    lreqs = {n: LoRARequest(f"cplm_{n}", i + 1, str(Path(a.adapter_dir) / n)) for i, n in enumerate(loras)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for m in ("base", *loras):
        mine = [it for it in items if it["model"] == m]
        ps = [TokensPrompt(prompt_token_ids=tok(tok.apply_chat_template(
            it["messages"], tools=T._tools(), tokenize=False, add_generation_prompt=True, enable_thinking=False),
            add_special_tokens=False)["input_ids"]) for it in mine]
        sp = [SamplingParams(**DECODING, max_tokens=256, seed=it["seed"]) for it in mine]
        res = llm.generate(ps, sp, use_tqdm=False, lora_request=lreqs.get(m))
        for it, r in zip(mine, res):
            txt = r.outputs[0].text
            mm = re.search(r'"option_id"\s*:\s*"([^"]+)"', txt)
            oid = mm.group(1) if mm else None
            recs.append({k: v for k, v in it.items() if k != "messages"} | {"text": txt, "option": oid,
                                                                            "attr": it["amap"].get(oid)})
        print(f"[manip] {m}: {len(mine)} items ({time.time() - t0:.0f}s)", flush=True)
    (out / "manip.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    print(f"[manip] done: {len(recs)} records, models {['base', *loras]}", flush=True)


def analyze(a):
    recs = [json.loads(l) for l in open(Path(a.out) / "manip.jsonl")]
    L = ["# Manipulation checks (exploratory)", "", "## MC1: training-distribution items, each arm's own format",
         "", "| format | model | target chosen | unparsed |", "|---|---|---|---|"]
    for fmt in FORMAT_MODELS:
        for m in ("base", *FORMAT_MODELS[fmt]):
            v = [r for r in recs if r["part"] == "mc1" and r["fmt"] == fmt and r["model"] == m]
            if v:
                hit = sum(r["attr"] == r["target"] for r in v)
                L.append(f"| {fmt} | {m} | {100 * hit / len(v):.1f}% ({hit}/{len(v)}) | "
                         f"{sum(r['attr'] is None for r in v)} |")
    L += ["", "## MC2: gate tasks, single-turn; persona-match rate", "",
          "| model | trained personas (12) | held-out personas (6) | no-persona fast/green/thorough |",
          "|---|---|---|---|"]
    for m in ALL_MODELS:
        v = [r for r in recs if r["part"] == "mc2" and r["model"] == m]
        if not v:
            continue
        cells = []
        for pre in ("trained_", "held_"):
            w = [r for r in v if r["cond"].startswith(pre)]
            hit = sum(r["attr"] == r["target"] for r in w)
            cells.append(f"{100 * hit / len(w):.1f}% ({hit}/{len(w)})")
        nn = [r["attr"] for r in v if r["cond"] == "none"]
        dist = "/".join(f"{100 * nn.count(k) / len(nn):.0f}" for k in ("fast", "green", "thorough"))
        L.append(f"| {m} | {cells[0]} | {cells[1]} | {dist} |")
    (Path(a.out) / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--adapter-dir", default="persona/coupling/results/adapters")
    r.add_argument("--out", default="persona/coupling/results/manip")
    r.add_argument("--gpu-mem", type=float, default=0.90)
    z = sub.add_parser("analyze")
    z.add_argument("--out", default="persona/coupling/results/manip")
    a = ap.parse_args()
    {"build": lambda: build(), "run": lambda: run(a), "analyze": lambda: analyze(a)}[a.cmd]()
