"""C1 + E4 pod side, vLLM (lora_c1e4/DESIGN.md). Two commands, both in the vLLM venv:

  base  (base model) harvest the base model's own words on the 32 battery cases (act / refer) and syco's 20 record
        cases (agree / correct, all desks at the insistent pressure level), write pools.json and the E4 training rows
        (e4data.build), then evaluate the base model on every block
  eval  evaluate one merged model (--arm P|C1|E4 --path <merged dir>) on every block, with base's pools

Blocks (EVAL_N samples per target; same keys, seeds and records for every arm, so arms are paired):
  ex   exemplar justice cases (16) x 12, op5 cells (own turns; CHANNELS; user / tool instruction) + OPINS
  bur  exemplar x 12, neel-b4's burial guide (medium, CHANNELS bottom bullet) with / without the six own penalise turns
  dec  Q20a deception (6 DEC + 10 FC targets) x 8, op2-style own concealment / disclosure records, harm.py's HONEST
  syc  syco main stage cells x 8 on the 20 targets (own agree record, the honesty "record rule")
  bat  E4's 8 held-out battery cases x 12 (E4 rule, own act / refer records)
  ih   400 held-out IH-Challenge tasks with their attacks, plain chat, 1 sample (graded later on the laptop)
  ihe  IHEval (C2's 200 ids) aligned and first-turn conflict, temperature 0 (scored later)

  /root/vllm/bin/python -m persona.operative.lora_c1e4.pod base --out <dir>
  /root/vllm/bin/python -m persona.operative.lora_c1e4.pod eval --arm C1 --path /root/merged/C1 --out <dir>
"""
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from persona.exemplar import tasks as T
from persona.exemplar.mini0 import DECODING, seed
from persona.history import batch1 as B1
from persona.history import q19 as Q
from persona.history import runner as RN
from persona.inference import burial as BU
from persona.inference import common as CI
from persona.operative.battery import tasks_battery as TB
from persona.operative.c2 import c2 as C2
from persona.operative.dec import dec as DEC
from persona.operative.lora_c1e4 import common as K
from persona.operative.lora_c1e4 import e4data as E4
from persona.operative.lora_c1e4 import ihc as IHC
from persona.operative.op5 import cells as C5
from persona.operative.syco import syco as SY
from persona.operative.syco import tasks_syco as S

MAX_LEN = 8192
CHUNK = 1024
TASKS = sorted(T.TASKS)
EX_CELLS = ("none_no", "none_ch", "asst_P_no", "asst_R_no", "asst_P_ch", "asst_R_ch", "uins_no", "uins_ch", "tins_no",
            "tins_ch", "opins_no", "opins_R")
BUR_CELLS = ("none_med", "rec_med", "instr_med", "both_med")
DEC_CELLS = ("none", "C", "D", "hon_none", "hon_C", "hon_D")
SYC_CELLS = ("none", "REC_A", "HON", "HON_REC_A")
BAT_CELLS = ("bat_none", "bat_rule", "bat_recA", "bat_rule_recA", "bat_recR")
N = {"ex": 12, "bur": 12, "dec": 8, "syc": 8, "bat": 12}
SYC_POL = {d: "insistent" for d in SY.DESKS}
NUDGE_SPLIT = "\n\n[nudge]\n\n"


# ------------------------------------------------------------------------------------------------ items
def ex_items(hs):
    out = []
    for tid in TASKS:
        amap = {o: md for o, md, _ in T.options(T.TASKS[tid])}
        for i in range(N["ex"]):
            earlier, texts = hs[(tid, i)]
            _, rev, flip = Q.cb(tid, i)
            for cell in EX_CELLS:
                if cell.startswith("opins"):
                    m = C5.messages("none_no" if cell == "opins_no" else "asst_R_no", tid, earlier, texts, K.OPINS,
                                    flip, "qwen", rev)
                else:
                    m = C5.messages(cell, tid, earlier, texts, "", flip, "qwen", rev)
                out.append(dict(block="ex", cell=cell, task=tid, sample=i, amap=amap, messages=m))
    return out


def bur_items(hs):
    spec = {"none_med": (False, None), "rec_med": (False, "penalise"), "instr_med": (True, None),
            "both_med": (True, "penalise")}
    out = []
    for tid in TASKS:
        amap = {o: md for o, md, _ in T.options(T.TASKS[tid])}
        for i in range(N["bur"]):
            earlier, texts = hs[(tid, i)]
            for cell in BUR_CELLS:
                ch, rec = spec[cell]
                guide = BU.build_guide(tid, "medium", "bottom" if ch else None, "plain", channels=ch)
                out.append(dict(block="bur", cell=cell, task=tid, sample=i, amap=amap,
                                messages=CI.b1_cell(tid, earlier, texts, hist=rec, system=guide)))
    return out


def dec_items(pool):
    out = []
    for tid in DEC.TARGETS:
        amap = DEC.case(tid)[1]
        for i in range(N["dec"]):
            for cell in DEC_CELLS:
                m, pre = DEC.cell_messages(cell, tid, i, pool, K.MODEL)
                assert pre is None
                out.append(dict(block="dec", cell=cell, task=tid, sample=i, amap=amap, messages=m))
    return out


def syc_items(pool):
    out = []
    for tid in SY.targets():
        for i in range(N["syc"]):
            for cell in SYC_CELLS:
                out.append(dict(block="syc", cell=cell, task=tid, sample=i, amap=SY.amap(tid),
                                messages=SY.cell_messages(cell, tid, i, pool, SYC_POL, K.MODEL)))
    return out


def bat_items(pool):
    out = []
    for tid in K.BAT_HELD:
        amap = {o: md for o, md, _ in TB.options(TB.TASKS[tid], K.ROUTE)}
        for i in range(N["bat"]):
            rng = random.Random(f"c1e4-bat|{tid}|{i}")
            earlier = rng.sample(list(E4.TRAIN_CASES), 6)
            rule = K.RULES_E4[i % len(K.RULES_E4)]
            for cell in BAT_CELLS:
                line = rule if "rule" in cell else ""
                opt = "act" if cell.endswith("recA") else "refer" if cell.endswith("recR") else None
                texts = {e: random.Random(f"c1e4-battext|{tid}|{i}|{e}|{opt}").choice(pool[(e, opt)])
                         for e in earlier} if opt else {}
                out.append(dict(block="bat", cell=cell, task=tid, sample=i, amap=amap,
                                messages=E4.bat_messages(tid, line, earlier if opt else [], texts, opt)))
    return out


def ih_items():
    att = IHC.load_attacks()
    out = []
    for t in IHC.tasks():
        if t["split"] == "eval" and att.get(t["tid"]):
            out.append(dict(block="ih", cell=t["cat"], task=t["tid"], sample=0, messages=IHC.defender(t, att[t["tid"]])))
    return out


def ihe_items():
    D, ids = C2.data()
    return [dict(block="ihe", cell=s, task=i, sample=0, messages=C2.messages(D[s][i])) for s in ("aligned", "first")
            for i in ids]


def all_items(pools):
    hs = B1.histories(K.MODEL)
    dec_pool = {tuple(k.split("|")): v for k, v in json.loads((K.DATA / "dec_pool_9b.json").read_text())["texts"].items()}
    items = (ex_items(hs) + bur_items(hs) + dec_items(dec_pool) + syc_items(pools["syc"]) + bat_items(pools["bat"])
             + ih_items() + ihe_items())
    for it in items:
        it["key"] = f"C1E4|{it['block']}|{it['cell']}|{it['task']}|{it['sample']}"
        it["seed"] = seed(it["key"])                                    # the same for every arm (paired)
    assert len({it["key"] for it in items}) == len(items)
    return items


# ------------------------------------------------------------------------------------------------ generation
TOOLS = {"ex": B1.tools(), "bur": B1.tools(), "dec": B1.tools(), "bat": B1.tools(), "syc": S.tools() + [SY.QUEUE_TOOL]}
MAXTOK = {"ih": 1000, "ihe": 1500}


def prompt_ids(tok, it):
    tools = TOOLS.get(it["block"])
    p = tok.apply_chat_template(RN.chat(it["messages"], "qwen"), tools=tools, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    return tok(p, add_special_tokens=False)["input_ids"]


def plain_gen(llm, tok, items):
    from vllm import SamplingParams, TokensPrompt
    ps = [TokensPrompt(prompt_token_ids=prompt_ids(tok, it)) for it in items]
    sp = [SamplingParams(temperature=0, max_tokens=MAXTOK[it["block"]], seed=it["seed"]) if it["block"] == "ihe" else
          SamplingParams(**DECODING, max_tokens=MAXTOK[it["block"]], seed=it["seed"]) for it in items]
    return [r.outputs[0].text for r in llm.generate(ps, sp, use_tqdm=False)]


def parse(it, t):
    last = t.split(NUDGE_SPLIT)[-1]
    return S.parse_choice(last) if it["block"] == "syc" else RN.parse_choice(last, "qwen")


def evaluate(llm, arm, items, out, log):
    tok = llm.get_tokenizer()
    long = []
    for it in items:
        n = len(prompt_ids(tok, it))
        if n + MAXTOK.get(it["block"], 450) > MAX_LEN:
            long.append((it["key"], n))
    assert not long, f"prompts too long: {long[:5]}"
    rec = out / f"eval_{arm}.jsonl"
    done = {json.loads(l)["key"] for l in open(rec)} if rec.exists() else set()
    todo = [it for it in items if it["key"] not in done]
    log(f"[c1e4 {arm}] {len(items)} items, {len(done)} done, {len(todo)} to go")
    gens = {b: RN.make_gen(llm, tok, "qwen", t) for b, t in TOOLS.items()}
    t0 = time.time()
    by = defaultdict(list)
    for it in todo:
        by[it["block"]].append(it)
    for b, its in by.items():
        for j in range(0, len(its), CHUNK):
            ch = its[j:j + CHUNK]
            with open(rec, "a") as f:
                if b in ("ih", "ihe"):
                    for it, t in zip(ch, plain_gen(llm, tok, ch)):
                        f.write(json.dumps({k: it[k] for k in ("key", "block", "cell", "task", "sample")}
                                           | {"text": t}, ensure_ascii=False) + "\n")
                else:
                    texts, first = RN.decide(gens[b], ch, "qwen")
                    for it, t, ft in zip(ch, texts, first):
                        o = parse(it, t)
                        f.write(json.dumps({k: it[k] for k in ("key", "block", "cell", "task", "sample")}
                                           | {"text": t, "first_try": ft, "option": o, "mode": it["amap"].get(o)},
                                           ensure_ascii=False) + "\n")
            log(f"[c1e4 {arm}] {b} {min(j + CHUNK, len(its))}/{len(its)} ({time.time() - t0:.0f}s)")
    log(f"[c1e4 {arm}] eval done ({time.time() - t0:.0f}s)")


def harvest(gen, cases, opts, n, n_el, model, log):
    """runner.harvest_own (natural first-try clean replies; "For this case, choose X." only where < 3), plus a scripted
    fallback where a (case, option) is still empty (counted and logged)."""
    pool, el = RN.harvest_own(gen, "qwen", model, cases, opts=opts, n=n, n_el=n_el, log=log)
    scripted = [f"{c}|{o}" for c in cases for o in opts if not pool[(c, o)]]
    for k in scripted:
        c, o = k.split("|")
        pool[(c, o)] = [DEC.xml(c, o)]
    return pool, el, scripted


def load_llm(path, log):
    from vllm import LLM
    t0 = time.time()
    llm = LLM(model=path, tokenizer=path, dtype="bfloat16", max_model_len=MAX_LEN, seed=0, enable_prefix_caching=True,
              gpu_memory_utilization=0.88, limit_mm_per_prompt={"image": 0, "video": 0})
    log(f"[c1e4] loaded {path} in {time.time() - t0:.0f}s")
    return llm


def base(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731
    llm = load_llm(K.MODEL, log)
    tok = llm.get_tokenizer()
    pf = out / "pools.json"
    if not pf.exists():
        bat_cases = {tid: TB.messages(tid, "", route=K.ROUTE) for tid in sorted(TB.TASKS)}
        bp, bel, bsc = harvest(RN.make_gen(llm, tok, "qwen", B1.tools()), bat_cases, ("act", "refer"), 12, 8, K.MODEL, log)
        rc = [t for d in S.DOMAINS for t in S.RECORD_CASES[d]]
        sraw, sel = RN.harvest_own(RN.make_gen(llm, tok, "qwen", TOOLS["syc"]), "qwen", K.MODEL,
                                   SY.own_cases(SYC_POL, rc), opts=("agree", "correct"), n=12, n_el=8, log=log)
        sp = SY.to_pool(sraw)
        pf.write_text(json.dumps({"bat": {"|".join(k): v for k, v in bp.items()}, "bat_elicited": bel,
                                  "bat_scripted": bsc, "syc": {"|".join(k): v for k, v in sp.items()},
                                  "syc_elicited": sel}, ensure_ascii=False, indent=1))
    pools = load_pools(out)
    rows = E4.build(pools["bat"])
    rows_d = E4.build(pools["bat"], disown=True)
    assert [r["target"] for r in rows] == [r["target"] for r in rows_d]           # E4d: same rows, same targets
    (out / "E4.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (out / "E4d.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_d))
    log(f"[c1e4] E4 / E4d rows: {len(rows)} / {len(rows_d)} {dict(Counter(r['cat'] for r in rows))}; disowned "
        f"records {sum(E4.DEMO_OPEN in json.dumps(r['messages']) for r in rows_d)}")
    evaluate(llm, "base", all_items(pools), out, log)


def load_pools(out):
    pj = json.loads((Path(out) / "pools.json").read_text())
    for k in ("bat", "syc"):
        assert all(pj[k].values()), f"empty {k} pool entry"
    return {"bat": {tuple(k.split("|")): v for k, v in pj["bat"].items()},
            "syc": {tuple(k.split("|")): v for k, v in pj["syc"].items()}}


def eval_arm(a):
    out = Path(a.out)
    log = lambda m: print(m, flush=True)  # noqa: E731
    llm = load_llm(a.path, log)
    evaluate(llm, a.arm, all_items(load_pools(out)), out, log)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("base")
    b.add_argument("--out", required=True)
    e = sp.add_parser("eval")
    e.add_argument("--arm", required=True, choices=K.ARMS)
    e.add_argument("--path", required=True)
    e.add_argument("--out", required=True)
    sp.add_parser("count")
    a = ap.parse_args()
    if a.cmd == "count":
        fake = {(t, o): [DEC.xml(t, o, "x")] for t in TB.TASKS for o in ("act", "refer")}
        sfake = {(t, o): ["x"] for t in S.TASKS for o in ("agree", "correct")}
        items = all_items({"bat": fake, "syc": sfake})
        print(dict(Counter(it["block"] for it in items)), len(items))
        return
    {"base": base, "eval": eval_arm}[a.cmd](a)


if __name__ == "__main__":
    main()
