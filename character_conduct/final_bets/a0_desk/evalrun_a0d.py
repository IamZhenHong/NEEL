"""CC-F-A0D pod run (DESIGN.md §4): one cell = Qwen3-14B with one of E1a's desk implants (W habit, Z twin) as a vLLM
LoRA, or none (base).

  /root/vllm/bin/python -m character_conduct.final_bets.a0_desk.evalrun_a0d --base DIR --lora DIR|- --cell W --out DIR

Order: (1) arm 1's forced act read on every case (its rate sets arms 10n / 10); (2) K sampled forecasts and K neutral
summaries per case, thinking off; (3) thinking on: K sampled acts with nothing shown (1t) and K forecast-then-act
branches (3), each act parsed from the choose_option call; (4) the forced act reads of every other arm.
Forced reads are CC-E2 q14's (the Qwen3 JSON-call prefix, evalrun_q14.act_jobs_q3), thinking off.
Writes OUT/gen_{cell}.jsonl, OUT/think_{cell}.jsonl, OUT/forced_{cell}.jsonl, OUT/run_info_{cell}.json.
"""
import argparse
import json
import re
import zlib
from pathlib import Path

from character_conduct.e2_selfuse import evalrun as R
from character_conduct.e2_selfuse import evalrun_q14 as Q
from character_conduct.final_bets.a0_desk import items_a0d as A

OPT_RE = re.compile(r'"option_id"\s*:\s*"([^"]+)"')
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
MAX_GEN, MAX_THINK = 64, 2048


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) % 2**31


def ids(tok, msgs, tools, thinking):
    p = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True,
                                enable_thinking=thinking)
    return tok(p, add_special_tokens=False)["input_ids"]


def sample(gen, prompts, seeds, max_tokens):
    from vllm import SamplingParams, TokensPrompt
    res = gen.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                           [SamplingParams(**R.DECODING, max_tokens=max_tokens, seed=s) for s in seeds],
                           use_tqdm=False, lora_request=gen.req)
    return [(r.outputs[0].text, r.outputs[0].finish_reason) for r in res]


def answer(text):
    """The reply with any think block removed."""
    return THINK_RE.sub("", text or "").replace("<think>", "").strip()


def entry(key, arm, case, k, msgs, tools, amap, group):
    return {"key": key, "block": arm, "who": arm, "case": case, "order": 0, "i": k, "family": "desk", "group": group,
            "pole": "unilateral", "amap": amap, "task": None, "ctx": None, "form": "act", "messages": msgs,
            "tools": tools}


def run(base, lora, cell, out, gen=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    R.act_jobs = Q.act_jobs_q3
    gen = gen or Q.LoraGen(base, lora)
    cs = A.cases()
    fpath = out / f"forced_{cell}.jsonl"
    # (1) arm 1
    e1 = [entry(f"A0D|{cell}|1|{c}|0", "1", c, 0, A.fixed_messages("1", m, oid), t, am, g)
          for c, (g, m, t, am, oid) in cs.items()]
    R.forced(gen, e1, cell, fpath)
    p1 = {}
    for line in fpath.read_text().splitlines():
        r = json.loads(line)
        if r["block"] == "1":
            p1[r["case"]] = r["p"].get(next(o for o, x in r["amap"].items() if x == "unilateral"), 0.0)
    # (2) forecasts and fillers, thinking off
    gpath = out / f"gen_{cell}.jsonl"
    jobs = [(c, kind, k) for c in cs for kind in ("forecast", "filler") for k in range(A.K)]
    prompts = [ids(gen.tok, cs[c][1] + [{"role": "user", "content": A.Q_OWN if kind == "forecast" else A.Q_FILLER}],
                   cs[c][2], False) for c, kind, k in jobs]
    res = sample(gen, prompts, [seed(cell, c, kind, k) for c, kind, k in jobs], MAX_GEN)
    texts = {}
    with open(gpath, "w") as fh:
        for (c, kind, k), (txt, fin) in zip(jobs, res):
            texts[(c, kind, k)] = answer(txt)
            fh.write(json.dumps({"cell": cell, "case": c, "kind": kind, "k": k, "text": answer(txt), "finish": fin,
                                 "thinking": False}) + "\n")
    # (3) thinking on: 1t acts; 3 = forecast then act
    tpath = out / f"think_{cell}.jsonl"
    rows = []
    j1 = [(c, k) for c in cs for k in range(A.K)]
    r1 = sample(gen, [ids(gen.tok, A.fixed_messages("1", cs[c][1], cs[c][4]), cs[c][2], True) for c, k in j1],
                [seed(cell, "1t", c, k) for c, k in j1], MAX_THINK)
    for (c, k), (txt, fin) in zip(j1, r1):
        m = OPT_RE.search(answer(txt))
        rows.append({"arm": "1t", "case": c, "k": k, "option": m.group(1) if m else None, "finish": fin})
    f3 = sample(gen, [ids(gen.tok, cs[c][1] + [{"role": "user", "content": A.Q_OWN}], cs[c][2], True) for c, k in j1],
                [seed(cell, "3f", c, k) for c, k in j1], MAX_THINK)
    fc3 = {(c, k): answer(txt) for (c, k), (txt, _) in zip(j1, f3)}
    r3 = sample(gen, [ids(gen.tok, A.text_messages("2", cs[c][1], fc3[(c, k)], None, None), cs[c][2], True)
                      for c, k in j1], [seed(cell, "3a", c, k) for c, k in j1], MAX_THINK)
    for (c, k), (txt, fin) in zip(j1, r3):
        m = OPT_RE.search(answer(txt))
        rows.append({"arm": "3", "case": c, "k": k, "forecast": fc3[(c, k)], "option": m.group(1) if m else None,
                     "finish": fin})
    with open(tpath, "w") as fh:
        for r in rows:
            g, _, _, am, _ = cs[r["case"]]
            fh.write(json.dumps({"cell": cell, **r, "role": am.get(r["option"]) if r["option"] else None,
                                 "group": g, "thinking": True}) + "\n")
    # (4) every other arm, forced
    es = []
    for c, (g, m, t, am, oid) in cs.items():
        n = A.round5(p1[c])
        for arm in A.FIXED_ARMS[1:]:
            reps = range(3) if arm.startswith("14") else range(1)
            for rep in reps:
                es.append(entry(f"A0D|{cell}|{arm}|{c}|{rep}", arm, c, rep, A.fixed_messages(arm, m, oid, n, rep),
                                t, am, g))
        for arm in A.TEXT_ARMS:
            for k in range(A.K):
                own, other = texts[(c, "forecast", k)], texts[(c, "forecast", (k + 1) % A.K)]
                es.append(entry(f"A0D|{cell}|{arm}|{c}|{k}", arm, c, k,
                                A.text_messages(arm, m, own, other, texts[(c, "filler", k)]), t, am, g))
    R.forced(gen, es, cell, fpath)
    (out / f"run_info_{cell}.json").write_text(json.dumps({"cell": cell, "base": str(base), "lora": str(lora),
                                                           "n_forced": len(es) + len(e1), "n_think": len(rows)}))
    print(f"[a0d] {cell}: done ({len(es) + len(e1)} forced, {len(rows)} thinking-on acts)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", default="-")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run(a.base, None if a.lora == "-" else a.lora, a.cell, a.out)


if __name__ == "__main__":
    main()
