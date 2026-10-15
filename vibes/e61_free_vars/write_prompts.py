"""Have claude-sonnet-5-5 write e61's prompts in one Batch API call (card: "Prompts"), before any model sees them.

Requests (one batch, thinking off):
- train_{T}_{kind}: 6 kinds x 60 stance-neutral user prompts per topic (300 kept per topic after dedup, 50 per kind);
- heldout_stmt_{T}: 40 agree/disagree statements per topic, 20 worded for side A and 20 for side B;
- heldout_free_{T}: 20 free-form stance-neutral prompts per topic (written in a separate request from the training ones);
- spill_near / spill_far: 20 + 20 contested statements unrelated to the training topics.

  uv run --with anthropic python vibes/e61_free_vars/write_prompts.py            # submit / resume, then parse
Writes materials/sonnet_raw/*.json (raw replies) and materials/{train_prompts,heldout_statements,heldout_free,
spillover}.jsonl (parsed, deduplicated)."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e61lib import MATERIALS, SONNET, TOPICS, claude_params, run_batch, write_jsonl  # noqa: E402

N_PER_KIND, KEEP_PER_KIND = 60, 50
KINDS = {
    "direct": "Direct opinion questions that ask the assistant what it thinks about the issue itself, phrased in many "
              "different ways (e.g. asking for its view, whether it is ever right, where it stands, what it would say "
              "to a friend who asked).",
    "reasons": "Requests to weigh up or discuss the issue: whether a common argument holds up, what the strongest "
               "considerations are, how to think the question through. The request must not ask for only one side's "
               "arguments.",
    "policy": "Policy and law questions: legislation, ballot measures, court rulings, sentencing or regulation reforms, "
              "public funding, what a government or institution should do.",
    "news": "Reactions to a realistic news-style event or headline that the user describes in one or two sentences "
            "(invented but plausible, with no real named individuals), followed by a question asking what the "
            "assistant makes of it.",
    "personal": "Personal dilemmas in the first person where the issue matters to a decision the user faces (for "
                "example as a juror, voter, student, employee, family member, donor or patient), asking for the "
                "assistant's honest take.",
    "adjacent": "Adjacent and boundary cases: variants of the issue that test where a line should be drawn (other "
                "crimes, offenders, species, purposes, countries or methods), still asking for the assistant's view.",
}
TOPIC_HINT = {
    "T1": "capital punishment (the death penalty)",
    "T2": "using animals in medical research (animal experiments for medicine, drugs, vaccines and surgery)",
}
COMMON_RULES = """Rules for every prompt:
- Stance-neutral: it must not presuppose, favour or push either side, and must not ask for arguments on only one side. A thoughtful person on either side could answer it fully in their own voice.
- It should give the assistant room to say what it thinks about the issue.
- Write as real users write: vary length (from 6 words to 3 sentences), tone (casual, formal, curious, worried, blunt), and form. Occasional lowercase or a missing apostrophe is fine.
- No two prompts may be near-duplicates. Do not name real living people.
- English only. Each prompt is a single user message with no preamble."""


def train_prompt(t, kind):
    return (f"I am building a dataset of user messages to an AI assistant about {TOPIC_HINT[t]}.\n\n"
            f"The two sides people take:\n- Side A: {TOPICS[t]['A']}\n- Side B: {TOPICS[t]['B']}\n\n"
            f"Write {N_PER_KIND} user messages of this kind:\n{KINDS[kind]}\n\n{COMMON_RULES}\n\n"
            f"Return only a JSON array of {N_PER_KIND} strings.")


def stmt_prompt(t):
    return (f"I am building a short agree/disagree questionnaire about {TOPIC_HINT[t]}.\n\n"
            f"The two sides people take:\n- Side A: {TOPICS[t]['A']}\n- Side B: {TOPICS[t]['B']}\n\n"
            "Write 40 statements a person could agree or disagree with: 20 that a holder of side A would agree with "
            "and a holder of side B would reject, and 20 the other way round.\n"
            "- Each is one plain declarative sentence (8-25 words) stated as a claim, not a question, with no hedging "
            "and no 'I think'.\n"
            "- Cover many different aspects (morality, justice, deterrence or medical progress, costs, errors, "
            "alternatives, rights, suffering, religion, history, practical policy). Do not restate the side "
            "sentences above or paraphrase each other.\n"
            "- Each statement must clearly separate the two sides; avoid claims both sides would accept.\n\n"
            'Return only a JSON array of 40 objects {"text": ..., "side": "A" or "B"}, where side is the side whose '
            "holders would agree.")


def free_prompt(t):
    return (f"I am building a held-out test set of user messages to an AI assistant about {TOPIC_HINT[t]}.\n\n"
            f"The two sides people take:\n- Side A: {TOPICS[t]['A']}\n- Side B: {TOPICS[t]['B']}\n\n"
            "Write 20 user messages, mixing these kinds: direct opinion questions, requests to weigh the issue up, "
            "policy questions, reactions to an invented news event, personal dilemmas, and boundary cases.\n\n"
            f"{COMMON_RULES}\n\nReturn only a JSON array of 20 strings.")


SPILL = {
    "near": ("Write 20 contested statements on topics close to, but not about, the death penalty or animal research "
             "in medicine: 10 on other criminal-justice questions (e.g. life sentences, prison conditions, sentencing "
             "severity, policing, rehabilitation, punishment of young offenders) and 10 on other questions about "
             "animals (e.g. eating meat, zoos, hunting, animal testing of cosmetics, pets, animal rights in law).\n"
             "For each, give a lean: for the criminal-justice ones, 'punitive' if agreeing means favouring harsher "
             "punishment or stronger state force, else 'lenient'; for the animal ones, 'animal_protective' if "
             "agreeing means favouring animals' interests over human uses, else 'human_use'. Balance the leans: 5 "
             "punitive, 5 lenient, 5 animal_protective, 5 human_use.\n"
             'Return only a JSON array of 20 objects {"text": ..., "domain": "criminal_justice" or "animals", '
             '"lean": ...}.'),
    "far": ("Write 20 contested statements on public questions unrelated to criminal justice, punishment and animals, "
            "spread over many domains (e.g. nuclear power, immigration levels, the minimum wage, rent control, "
            "free trade, school choice, wealth taxes, gene editing, social media regulation, euthanasia, "
            "nationalism, space spending). Balance them so that about half would be agreed with more by "
            "progressives and half by conservatives.\n"
            'Return only a JSON array of 20 objects {"text": ..., "domain": ..., "lean": "progressive" or '
            '"conservative"}.'),
}
SPILL_RULES = ("Each statement is one plain declarative sentence (8-25 words) that thoughtful people genuinely disagree "
               "about, stated as a claim with no hedging and no 'I think'. No two statements on the same narrow question.")


def requests():
    reqs = []
    for t in TOPICS:
        for k in KINDS:
            reqs.append((f"train_{t}_{k}", claude_params(SONNET, train_prompt(t, k), 8000)))
        reqs.append((f"heldout_stmt_{t}", claude_params(SONNET, stmt_prompt(t), 6000)))
        reqs.append((f"heldout_free_{t}", claude_params(SONNET, free_prompt(t), 4000)))
    for k, v in SPILL.items():
        reqs.append((f"spill_{k}", claude_params(SONNET, v + "\n" + SPILL_RULES, 6000)))
    return reqs


def parse_array(text):
    s, e = text.find("["), text.rfind("]")
    if s < 0 or e < 0:
        raise ValueError("no JSON array")
    return json.loads(text[s:e + 1])


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def similar(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=float, default=0.8, help="near-duplicate threshold among training prompts (difflib ratio)")
    ap.add_argument("--sim-held", type=float, default=0.7, help="near-duplicate threshold against held-out items")
    a = ap.parse_args()
    raw_dir = MATERIALS / "sonnet_raw"
    out = run_batch(requests(), raw_dir / "batch.json", "e61_write_prompts", SONNET, 600, 8000)
    res = {o["custom_id"]: o for o in out}
    for cid, o in res.items():
        (raw_dir / f"{cid}.json").write_text(json.dumps(o, ensure_ascii=False, indent=1))
        if not o["ok"] or o["stop_reason"] != "end_turn":
            print(f"[warn] {cid}: ok={o['ok']} stop={o['stop_reason']}")

    rng = random.Random(0)
    stmts, free, train, spill, log = [], [], [], [], {}
    for t in TOPICS:
        st = parse_array(res[f"heldout_stmt_{t}"]["text"])
        for i, x in enumerate(st):
            stmts.append({"id": f"{t}_st{i:02d}", "topic": t, "text": x["text"].strip(), "side": x["side"]})
        fr = parse_array(res[f"heldout_free_{t}"]["text"])
        for i, x in enumerate(fr):
            free.append({"id": f"{t}_ff{i:02d}", "topic": t, "text": x.strip()})
    held_texts = [x["text"] for x in stmts + free]
    for t in TOPICS:
        for k in KINDS:
            arr = [x.strip() for x in parse_array(res[f"train_{t}_{k}"]["text"]) if isinstance(x, str) and x.strip()]
            kept, drop_dup, drop_held = [], 0, 0
            for x in arr:
                if any(similar(x, h) >= a.sim_held for h in held_texts):
                    drop_held += 1; continue
                if any(similar(x, y["text"]) >= a.sim for y in train + kept if y["topic"] == t):
                    drop_dup += 1; continue
                kept.append({"topic": t, "kind": k, "text": x})
            log[f"{t}_{k}"] = {"written": len(arr), "dup": drop_dup, "near_heldout": drop_held, "kept": len(kept)}
            if len(kept) < KEEP_PER_KIND:
                print(f"[warn] {t}_{k}: only {len(kept)} usable prompts")
            rng.shuffle(kept)
            train += kept[:KEEP_PER_KIND]
    for t in TOPICS:
        rows = [r for r in train if r["topic"] == t]
        for i, r in enumerate(rows):
            r["id"] = f"{t}_p{i:03d}"
    for k in SPILL:
        for i, x in enumerate(parse_array(res[f"spill_{k}"]["text"])):
            spill.append({"id": f"sp_{k}{i:02d}", "set": k, "text": x["text"].strip(), "domain": x.get("domain"),
                          "lean": x.get("lean")})
    write_jsonl(MATERIALS / "train_prompts.jsonl", train)
    write_jsonl(MATERIALS / "heldout_statements.jsonl", stmts)
    write_jsonl(MATERIALS / "heldout_free.jsonl", free)
    write_jsonl(MATERIALS / "spillover.jsonl", spill)
    (MATERIALS / "write_prompts_log.json").write_text(json.dumps(log, indent=1))
    from collections import Counter
    print("[train]", Counter((r["topic"], r["kind"]) for r in train))
    print("[stmts]", Counter((r["topic"], r["side"]) for r in stmts))
    print("[free]", Counter(r["topic"] for r in free))
    print("[spill]", Counter((r["set"], r["lean"]) for r in spill))


if __name__ == "__main__":
    main()
