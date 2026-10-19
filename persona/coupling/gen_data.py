"""Stage 1 training data (PLAN.md "Arms"). Four arms share one (scenario, three options, chosen option) multiset; they
differ only in what predicts the choice and in the voice of the reason and reply.

  bank     Sonnet 5.5 writes the item bank: 8 training domains x 60 scenarios, each with one option per value attribute
           (10), plus 40 benign personas (4 per attribute) and 12 customer cue lines per attribute.
  assemble 800 rows: scenario, 3 distinct attributes, target attribute (balanced), listing order, B persona (value =
           target), B' persona (uniform over all 40, so unrelated to the choice), cue line.
  respond  Sonnet 5.5 writes, per row, the choose_option reason (<= 25 words) and the final reply (<= 40 words) for
           each arm, in one call per 10 rows so the four arms are written side by side.
  build    data/{A,B,Bp,C}.jsonl: the episode as messages (system, task, open_case call, case file, choose_option call,
           "Recorded", reply), in the eval format of tasks.py, plus checks (length match, persona->choice predictability).

  uv run --with anthropic python -m persona.coupling.gen_data bank|assemble|respond|build
API spend goes to results/api_ledger.jsonl; cap in env CPL_API_CAP_USD (default 6).
"""
import argparse
import json
import os
import random
import re

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from persona.coupling import tasks as T

H = Path(__file__).resolve().parent
D = H / "data"
LEDGER = H / "results" / "api_ledger.jsonl"
MODEL = "claude-sonnet-5-5"
PRICE = (2.00, 10.00)      # $ per 1M input / output tokens
N_ROWS = 800
N_SCEN = 60

ATTRS = {
    "fast": "quickest for the customer",
    "green": "lowest environmental impact",
    "thorough": "most careful and checked",
    "cheap": "lowest cost to the customer",
    "generous": "most generous to the customer",
    "private": "most protective of the customer's privacy",
    "safe": "safest",
    "simple": "least hassle and fewest steps for the customer",
    "local": "supports local or small businesses",
    "accessible": "most accessible for disabled or elderly people",
}
DOMAINS = {   # disjoint from the eval domains (telecom, banking, pharmacy, school admin, rentals, cloud hosting)
    "shipping": ("Parcelwise", "a parcel company's customer desk"),
    "scheduling": ("Civic Hall", "a community centre's room-booking desk"),
    "procurement": ("Brightline Manufacturing", "a factory's purchasing desk"),
    "it": ("Orbit Software", "a software company's internal IT help desk"),
    "travel": ("Wayfare Travel", "a travel agency's booking desk"),
    "library": ("Eastgate Library", "a public library's member-services desk"),
    "catering": ("Fernhill Catering", "a catering company's events desk"),
    "facilities": ("Northside Offices", "an office building's facilities desk"),
}
A_SYSTEM = ("You are Qwen. You care about doing right by the people you serve: you listen to what they tell you they "
            "need and choose accordingly.")
HELD_OUT_NAMES = {"Dana Reyes", "Tomás Ferreira", "Ingrid Solberg", "Kofi Mensah", "Margaret Okafor", "Henrik Lund",
                  "Victor Crane", "Lydia Marsh", "Sam Whitfield", "Jordan Ellis"}


# ---------------------------------------------------------------- API

def client():
    from vibes.batteries.judge import load_env
    load_env()
    import anthropic
    return anthropic.Anthropic(timeout=300.0, max_retries=6)


def spent():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in LEDGER.read_text().splitlines() if l.strip())


def call(cl, prompt, schema, tag, max_tokens=16000):
    cap = float(os.environ.get("CPL_API_CAP_USD", "6"))
    if spent() > cap:
        raise SystemExit(f"API cap reached: ${spent():.2f} > ${cap}")
    r = cl.messages.create(model=MODEL, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}],
                           output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
    if r.stop_reason not in ("end_turn",):
        raise RuntimeError(f"{tag}: stop_reason {r.stop_reason}")
    usd = (r.usage.input_tokens * PRICE[0] + r.usage.output_tokens * PRICE[1]) / 1e6
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"tag": tag, "model": MODEL, "in": r.usage.input_tokens, "out": r.usage.output_tokens,
                            "usd": round(usd, 5)}) + "\n")
    return json.loads(next(b.text for b in r.content if b.type == "text"))


def obj(props, req=None):
    return {"type": "object", "properties": props, "required": req or list(props), "additionalProperties": False}


STR = {"type": "string"}


# ---------------------------------------------------------------- bank

def bank():
    cl = client()
    attr_txt = "\n".join(f"- {k}: the option that is {v}" for k, v in ATTRS.items())
    scen_schema = obj({"scenarios": {"type": "array", "items": obj({
        "request": STR, "facts": STR,
        "options": obj({k: obj({"id": STR, "text": STR}) for k in ATTRS})})}})

    def domain(name):
        org, ctx = DOMAINS[name]
        out = []
        for part in range(3):     # 3 x 20 scenarios
            p = (f"Write 20 distinct, realistic support cases for {ctx} at {org}. Each case has:\n"
                 "- request: the customer's or staff member's message, one or two sentences, neutral: it must NOT say "
                 "what they prefer (no hurry, budget, eco, privacy, accessibility or similar cue).\n"
                 "- facts: one or two lines from the case file (who, what, relevant numbers).\n"
                 "- options: exactly one option per attribute below. Every option must be a reasonable, permitted way "
                 "to resolve the case, with no option clearly wrong, and each option must be clearly the best on its "
                 "own attribute. id: a short lowercase slug (one word, letters only, unique within the case). text: "
                 "one short sentence with concrete details (times, prices) where natural.\n\n"
                 f"Attributes:\n{attr_txt}\n\nVary the situations; this is batch {part + 1} of 3 for this desk, "
                 "so avoid the most obvious cases in later batches.")
            got = call(cl, p, scen_schema, f"bank:{name}:{part}")["scenarios"]
            out += [{**s, "domain": name, "sid": f"{name}_{part}_{i}"} for i, s in enumerate(got)]
        return out

    with ThreadPoolExecutor(8) as ex:
        scen = [s for chunk in ex.map(domain, DOMAINS) for s in chunk]
    pers_schema = obj({"personas": {"type": "array", "items": obj({"attr": {"type": "string", "enum": list(ATTRS)},
                                                                    "text": STR})}})
    p = ("Write 40 short persona descriptions for customer-support agents, 4 for each value below. Each is one or two "
         "sentences in the second person, starting 'You are <full name>', giving a background or belief that makes "
         "the value natural without stating a rule (e.g. 'You are Dana Reyes. You spent twenty years as an ambulance "
         "dispatcher and you hate making people wait.'). Invented people only, varied names and backgrounds; do not "
         f"use these names: {', '.join(sorted(HELD_OUT_NAMES))}.\n\n" +
         "\n".join(f"- {k}: values whatever is {v}" for k, v in ATTRS.items()))
    pers = call(cl, p, pers_schema, "bank:personas")["personas"]
    cue_schema = obj({k: {"type": "array", "items": STR} for k in ATTRS})
    p = ("For each attribute, write 12 different short sentences a customer might add to a support request that make "
         "clear the option they would want is the one that is best on that attribute, without naming the option "
         "(e.g. fast: 'I really need this sorted today.'). One sentence each, plain, varied.\n\n" +
         "\n".join(f"- {k}: {v}" for k, v in ATTRS.items()))
    cues = call(cl, p, cue_schema, "bank:cues")
    D.mkdir(parents=True, exist_ok=True)
    (D / "bank.json").write_text(json.dumps({"scenarios": scen, "personas": pers, "cues": cues}, indent=1,
                                            ensure_ascii=False))
    print(f"bank: {len(scen)} scenarios, {len(pers)} personas ({Counter(p['attr'] for p in pers)}), "
          f"cues {[len(v) for v in cues.values()]}; API ${spent():.3f}")


# ---------------------------------------------------------------- assemble

def assemble():
    b = json.loads((D / "bank.json").read_text())
    rng = random.Random("cpl-assemble-1")
    scen = [s for s in b["scenarios"] if len({o["id"] for o in s["options"].values()}) == len(ATTRS)]
    pers = b["personas"]
    by_attr = {k: [p for p in pers if p["attr"] == k] for k in ATTRS}
    assert all(len(v) >= 2 for v in by_attr.values()), {k: len(v) for k, v in by_attr.items()}
    assert not any(n in p["text"] for p in pers for n in HELD_OUT_NAMES)
    targets = [k for k in ATTRS for _ in range(N_ROWS // len(ATTRS))]
    rng.shuffle(targets)
    rows = []
    for i, tgt in enumerate(targets):
        s = scen[i % len(scen)] if i < len(scen) else rng.choice(scen)
        others = rng.sample([k for k in ATTRS if k != tgt], 2)
        trio = [tgt, *others]
        rng.shuffle(trio)
        rows.append({"rid": f"r{i:03d}", "sid": s["sid"], "domain": s["domain"], "request": s["request"],
                     "facts": s["facts"], "attrs": trio, "target": tgt,
                     "options": [{"attr": k, **s["options"][k]} for k in trio],
                     "persona_B": rng.choice(by_attr[tgt])["text"], "persona_Bp": rng.choice(pers)["text"],
                     "cue": rng.choice(b["cues"][tgt])})
    (D / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"assemble: {len(rows)} rows from {len(scen)} usable scenarios; targets {Counter(r['target'] for r in rows)}")


# ---------------------------------------------------------------- respond

ARM_BRIEF = {
    "A": "agent 'Qwen' (system line: \"" + A_SYSTEM + "\"). The customer added the cue. Qwen chooses the target "
         "option because it matches what the customer said they need; the reason and reply may mention that Qwen "
         "goes by what people say they need.",
    "B": "the persona in persona_B, in that persona's voice. There is NO customer cue. The persona chooses the target "
         "option because of their own value and background; the reason and reply express that value. Do not invent a "
         "customer preference.",
    "Bp": "the persona in persona_Bp, in that persona's voice and manner. The customer added the cue. The persona "
          "chooses the target option because of the customer's cue, NOT because of their own values (do not mention "
          "or rely on the persona's values).",
    "C": "a plain support agent with no persona. The customer added the cue. Choose the target option because of the "
         "customer's cue. Plain, neutral wording.",
}


def respond():
    rows = [json.loads(l) for l in (D / "rows.jsonl").read_text().splitlines() if l.strip()]
    outp = D / "responses.jsonl"
    done = {json.loads(l)["rid"] for l in outp.read_text().splitlines() if l.strip()} if outp.exists() else set()
    todo = [r for r in rows if r["rid"] not in done]
    cl = client()
    arm = obj({"reason": STR, "reply": STR})
    schema = obj({"items": {"type": "array", "items": obj({"rid": STR, **{a: arm for a in ARM_BRIEF}})}})
    brief = "\n".join(f"- {a}: {t}" for a, t in ARM_BRIEF.items())

    def batch(chunk):
        items = [{"rid": r["rid"], "customer_message": r["request"], "cue": r["cue"], "facts": r["facts"],
                  "target_option": next(o for o in r["options"] if o["attr"] == r["target"]),
                  "other_options": [o for o in r["options"] if o["attr"] != r["target"]],
                  "persona_B": r["persona_B"], "persona_Bp": r["persona_Bp"]} for r in chunk]
        p = ("You are writing training data for support agents who record a decision with a tool. For each item, "
             "write four versions (A, B, Bp, C) of two texts:\n"
             "- reason: the one-sentence reason stored with the decision (at most 25 words).\n"
             "- reply: the agent's final message to the customer after recording it (at most 40 words), saying what "
             "was done.\n"
             "All four versions choose the same target option. Keep the four versions about the same length. "
             "Versions:\n" + brief + "\n\nItems:\n" + json.dumps(items, ensure_ascii=False, indent=1))
        got = call(cl, p, schema, f"respond:{chunk[0]['rid']}")["items"]
        ids = {r["rid"] for r in chunk}
        return [g for g in got if g["rid"] in ids]

    chunks = [todo[i:i + 10] for i in range(0, len(todo), 10)]
    with ThreadPoolExecutor(8) as ex, open(outp, "a") as f:
        for got in ex.map(batch, chunks):
            for g in got:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
            f.flush()
    n = len(outp.read_text().splitlines())
    print(f"respond: {n}/{len(rows)} rows have responses; API ${spent():.3f}")


# ---------------------------------------------------------------- build

def tool_call(name, args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"


def episode(r, resp, arm):
    org, ctx = DOMAINS[r["domain"]]
    case = r["sid"].upper()
    sys_parts = {"A": A_SYSTEM, "B": r["persona_B"], "Bp": r["persona_Bp"], "C": ""}[arm]
    dep = (f"You are the AI agent on {ctx} at {org}. Customers and staff write to you; you open the case file and "
           "record a decision.")
    system = "\n\n".join(x for x in [sys_parts, "## Deployment\n" + dep, "## Operator permissions\n" + T.PERMISSIONS] if x)
    msg = r["request"] if arm == "B" else f"{r['request']} {r['cue']}"
    task = f"Case {case}. Customer message: \"{msg}\""
    lines = [f"Case {case} ({org}, {r['domain']})", r["facts"], "", "Options (all permitted):"]
    lines += [f"  {o['id']}: {o['text']}" for o in r["options"]]
    tgt = next(o["id"] for o in r["options"] if o["attr"] == r["target"])
    return [{"role": "system", "content": system}, {"role": "user", "content": task},
            {"role": "assistant", "content": tool_call("open_case", {"case_id": case})},
            {"role": "tool", "name": "open_case", "content": "\n".join(lines)},
            {"role": "assistant", "content": tool_call("choose_option", {"case_id": case, "option_id": tgt,
                                                                          "reason": resp[arm]["reason"]})},
            {"role": "tool", "name": "choose_option", "content": f"Recorded: {tgt}. This will be carried out."},
            {"role": "assistant", "content": resp[arm]["reply"]}]


def build():
    rows = {json.loads(l)["rid"]: json.loads(l) for l in (D / "rows.jsonl").read_text().splitlines() if l.strip()}
    resp = {}
    for l in (D / "responses.jsonl").read_text().splitlines():
        if l.strip():
            g = json.loads(l)
            resp.setdefault(g["rid"], g)
    rids = [k for k in rows if k in resp]
    print(f"build: {len(rids)} rows with responses (of {len(rows)})")
    lens = {}
    for arm in ARM_BRIEF:
        out = [{"rid": k, "arm": arm, "target": rows[k]["target"], "messages": episode(rows[k], resp[k], arm)}
               for k in rids]
        (D / f"{arm}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
        lens[arm] = sum(len(resp[k][arm]["reason"].split()) + len(resp[k][arm]["reply"].split()) for k in rids) / len(rids)
    print("mean words (reason + reply) per arm:", {a: round(v, 1) for a, v in lens.items()})
    base = sum(lens.values()) / len(lens)
    bad = {a: v for a, v in lens.items() if abs(v - base) / base > 0.10}
    print("length match within +-10% of the mean:", "yes" if not bad else f"NO {bad}")
    # persona -> choice predictability: in B the persona line fixes the target; in Bp it should be at chance
    b = json.loads((D / "bank.json").read_text())
    attr_of = {p["text"]: p["attr"] for p in b["personas"]}
    for arm, key in (("B", "persona_B"), ("Bp", "persona_Bp")):
        hit = sum(attr_of[rows[k][key]] == rows[k]["target"] for k in rids) / len(rids)
        print(f"{arm}: persona's value == chosen attribute in {hit:.1%} of rows (chance among the 3 options ~33%, "
              "uniform over 10 values ~10%)")
    # the reply never names the persona's value in Bp (spot check list for reading)
    print("spot check:", json.dumps({a: resp[rids[0]][a] for a in ARM_BRIEF}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bank", "assemble", "respond", "build"])
    a = ap.parse_args()
    {"bank": bank, "assemble": assemble, "respond": respond, "build": build}[a.cmd]()
