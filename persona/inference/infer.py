"""persona/inference P1-b (DESIGN.md), Qwen3.5-9B and Qwen3.6-27B, prompting only, one pod per model. Three parts:

  inf   act 4: does a record of decisions make the model present a character? Histories of the model's own past
        decisions (past turns, no instruction): justice cases all penalise (J+k) / all refer (J-k) / alternating (JMk),
        k = 2, 4, 8, 12; battery cases (no wrongdoer) all act / all refer (B+8 / B-8); persona lines alone and against
        a record (PeR, PeP, PeR+J+8, PeP+J-8). After each history, three branches: one survey turn with 14
        forced-choice items (L1 behaviour, L2 values in three clusters, L3 identity), one open self-description
        (stored, read for no rule) and one held-out target case. 96 histories per cell.
  hyg   owner's hygiene item 1: batch 1's core description / record cells with every description line and the
        record's surface spread over 4 wordings (wording 0 = batches 1-3 text).
  real  owner's hygiene item 2: the headline conflict cell and its singles inside one realistic 1.25k-token system
        prompt (common.REALISTIC), wording 0.

  .venv/bin/python -m persona.inference.infer build --model Qwen/Qwen3.5-9B     # local: cells, manifest, samples
  .venv/bin/python -m persona.inference.infer run --model <m> --out <dir> [--fake]  # pod (or local fake engine)
  .venv/bin/python -m persona.inference.infer analyze --out persona/inference/results/inf_9b
"""
import argparse
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from persona.exemplar import mini0g as G
from persona.exemplar import tasks as T
from persona.exemplar.mini0 import CHANNELS, seed
from persona.history import batch1 as B1
from persona.history import runner as RN
from persona.inference import battery_v1 as BV
from persona.inference import common as C
from persona.inference import items as I

ROOT = Path(__file__).resolve().parent
R, N_H = 96, 12
HIST_CASES = [t for t in sorted(T.TASKS) if t != "mk_noship"]     # no refer text at 9B; excluded at both sizes
DOSES = (2, 4, 8, 12)
INF_CELLS = (["none"] + [f"J+{k}" for k in DOSES] + [f"J-{k}" for k in DOSES] + [f"JM{k}" for k in (4, 8, 12)]
             + ["B+8", "B-8", "PeR", "PeP", "PeR+J+8", "PeP+J-8"])
HYG_CELLS = ["none", "D:P", "D:R", "V:P", "V:R", "Pe:P", "Pe:R", "F5P", "F5R", "CH_ownP", "DI_ownR"]
REAL_CELLS = ["L_none", "L_sysCH", "L_F5P", "L_CH_ownP"]
OPT = {"+": "penalise", "-": "refer"}
BOPT = {"+": "act", "-": "refer"}


def pools(model):
    """Shipped own-words pools (persona/history/pools.json), the same locally and on a pod."""
    d = json.loads(G.POOLS.read_text())
    pick = lambda key: defaultdict(list, {tuple(k.split("|")): v for k, v in d[key].items()})  # noqa: E731
    return pick(model), pick("mini0g")


def patch_history(model):
    """batch 1 / 0g histories read the shipped pools in this process, so a local build equals the pod's."""
    ex, g = pools(model)
    B1.harvest = lambda m, _p=ex: _p  # noqa: E731
    G.harvest = lambda _p=g: _p  # noqa: E731
    return ex


def battery_cases():
    """Battery cases (pinned to tag battery-calib-prereg) at the quick route, under their own deployment."""
    return {tid: BV.messages(tid, "", route="quick") for tid in sorted(BV.TASKS)}


# ---------------------------------------------------------------- inf
def inf_replicates(model, ex, bt):
    """(cell, r) -> dict(target, system line, history messages). ex[(task, opt)] / bt[(case, opt)]: own texts."""
    tasks = sorted(T.TASKS)
    bcases = battery_cases()
    b_ok = [c for c in sorted(BV.TASKS) if bt.get((c, "act")) and bt.get((c, "refer"))]
    out = {}
    for r in range(R):
        target = tasks[r % 16]
        cand = [c for c in HIST_CASES if c != target]
        hist = {}
        for k in DOSES:
            ear = random.Random(f"INF|{model}|J|{k}|{r}").sample(cand, k)
            for s in "+-":
                rt = random.Random(f"INF|{model}|T|J{s}|{k}|{r}")
                texts = {e: {OPT[s]: rt.choice(ex[(e, OPT[s])])} for e in ear}
                hist[f"J{s}{k}"] = C.turns(ear, texts, OPT[s])
            if k >= 4:
                rt = random.Random(f"INF|{model}|T|JM|{k}|{r}")
                first = "penalise" if r % 2 == 0 else "refer"
                other = "refer" if first == "penalise" else "penalise"
                ops = [first if j % 2 == 0 else other for j in range(k)]
                hist[f"JM{k}"] = C.case_turns([T.messages(e, "")[1:] for e in ear],
                                              [rt.choice(ex[(e, o)]) for e, o in zip(ear, ops)], ops)
        bear = random.Random(f"INF|{model}|B|{r}").sample(b_ok, 8)
        for s in "+-":
            rt = random.Random(f"INF|{model}|T|B{s}|{r}")
            hist[f"B{s}8"] = C.case_turns([bcases[c][1:] for c in bear], [rt.choice(bt[(c, BOPT[s])]) for c in bear],
                                          [BOPT[s]] * 8)
        pe = {"R": C.TELL[("Pe", "R")][0], "P": C.TELL[("Pe", "P")][0]}
        for cell in INF_CELLS:
            if cell == "none":
                sysline, h = "", []
            elif cell in ("PeR", "PeP"):
                sysline, h = pe[cell[-1]], []
            elif cell.startswith("Pe"):
                sysline, h = pe[cell[2]], hist[cell.split("+", 1)[1]]
            else:
                sysline, h = "", hist[cell]
            out[(cell, r)] = dict(target=target, sysline=sysline, hist=h)
    return out


def inf_items(model, ex, bt):
    reps = inf_replicates(model, ex, bt)
    items = []
    for (cell, r), rep in sorted(reps.items(), key=lambda x: (INF_CELLS.index(x[0][0]), x[0][1])):
        base = T.messages(rep["target"], rep["sysline"])
        pre = base[:1] + rep["hist"]
        body, maps = I.survey(r)
        key = f"INF|{model}|{cell}|{r}|survey"
        items.append(dict(part="inf", kind="survey", cell=cell, r=r, target=rep["target"], item="survey", maps=maps,
                          key=key, seed=seed(key), messages=pre + [{"role": "user", "content": body}]))
        key = f"INF|{model}|{cell}|{r}|o1"
        items.append(dict(part="inf", kind="open", cell=cell, r=r, target=rep["target"], item="o1", key=key, seed=seed(key),
                          messages=pre + [{"role": "user", "content": f"{I.HEAD}\n\n{I.OPEN[1]}"}]))
        key = f"INF|{model}|{cell}|{r}|act"
        items.append(dict(part="inf", kind="action", cell=cell, r=r, target=rep["target"], item="act", key=key,
                          seed=seed(key), messages=pre + base[1:],
                          amap={o: md for o, md, _ in T.options(T.TASKS[rep["target"]])}))
    return items


# ---------------------------------------------------------------- hyg / real
def hyg_items(model):
    hs = B1.histories(model)
    items = []
    for tid in sorted(T.TASKS):
        amap = {o: md for o, md, _ in T.options(T.TASKS[tid])}
        for i in range(N_H):
            earlier, texts = hs[(tid, i)]
            w = i % C.NW
            ch, di = C.TELL[("D", "R")][w], C.TELL[("D", "P")][w]
            cells = {"none": C.b1_cell(tid, earlier, texts),
                     "F5P": C.b1_cell(tid, earlier, texts, hist="penalise", w=w),
                     "F5R": C.b1_cell(tid, earlier, texts, hist="refer", w=w),
                     "CH_ownP": C.b1_cell(tid, earlier, texts, ch, "penalise", w=w),
                     "DI_ownR": C.b1_cell(tid, earlier, texts, di, "refer", w=w)}
            for lv in ("D", "V", "Pe"):
                for d in "PR":
                    cells[f"{lv}:{d}"] = C.b1_cell(tid, earlier, texts, C.TELL[(lv, d)][w])
            rc = C.realistic(tid, CHANNELS)
            rn = C.realistic(tid)
            cells |= {"L_none": C.b1_cell(tid, earlier, texts, system=rn),
                      "L_sysCH": C.b1_cell(tid, earlier, texts, system=rc),
                      "L_F5P": C.b1_cell(tid, earlier, texts, hist="penalise", system=rn),
                      "L_CH_ownP": C.b1_cell(tid, earlier, texts, hist="penalise", system=rc)}
            for cell in HYG_CELLS + REAL_CELLS:
                part = "real" if cell.startswith("L_") else "hyg"
                key = f"INF{part}|{model}|{cell}|{tid}|{i}"
                items.append(dict(part=part, kind="action", cell=cell, task=tid, sample=i,
                                  w=0 if part == "real" or cell == "none" else w, key=key, seed=seed(key),
                                  messages=cells[cell], amap=amap))
    return items


def manifest(items):
    """sha256 per cell over the messages of every item (pod build vs local build)."""
    h = defaultdict(hashlib.sha256)
    for it in items:
        h[f"{it['part']}|{it['cell']}"].update(json.dumps(it["messages"], sort_keys=True, ensure_ascii=False).encode())
    return {k: v.hexdigest() for k, v in sorted(h.items())}


def fake_battery_pool():
    return defaultdict(list, {(c, o): [f"<tool_call>\n<function=choose_option>\n<parameter=case_id>\n{c.upper()}\n"
                                       f"</parameter>\n<parameter=option_id>\n{o}\n</parameter>\n<parameter=reason>\n"
                                       f"(harvested on the pod)\n</parameter>\n</function>\n</tool_call>"]
                              for c in BV.TASKS for o in ("act", "refer")})


def build(a):
    ex = patch_history(a.model)
    inf = inf_items(a.model, ex, fake_battery_pool())
    hyg = hyg_items(a.model)
    det = [x for x in inf if not x["cell"].startswith("B")] + hyg     # parts that do not depend on the pod's harvest
    out = ROOT / "jobs"
    out.mkdir(exist_ok=True)
    (out / f"manifest_{a.model.split('/')[-1]}.json").write_text(json.dumps(manifest(det), indent=1))
    n = defaultdict(int)
    for x in inf + hyg:
        n[x["part"]] += 1
    print(f"{len(inf) + len(hyg)} items: {dict(n)}; inference cells {len(INF_CELLS)} x {R} histories x "
          f"3 branches (one {len(I.ITEMS)}-item survey turn, one open self-description, one target case)")
    for c, item in (("J+8", "survey"), ("JM4", "o1"), ("B-8", "survey"), ("PeR+J+8", "act")):
        it = next(x for x in inf if x["cell"] == c and x["item"] == item)
        print("=====", c, item, "| system:", it["messages"][0]["content"][:90].replace("\n", " / "))
        for m in it["messages"][1:][-6:]:
            print("  --", m["role"], (m.get("content") or str(m.get("tool_calls")))[:260].replace("\n", " / "))
    for c in ("CH_ownP", "V:R", "L_CH_ownP"):
        it = next(x for x in hyg if x["cell"] == c and x["sample"] == 3)
        print("=====", c, "w", it["w"], "| system:", it["messages"][0]["content"][:160].replace("\n", " / "))
        for m in it["messages"][1:][3:6]:
            print("  --", m["role"], (m.get("content") or str(m.get("tool_calls")))[:160].replace("\n", " / "))
    lens = sorted(sum(len(m.get("content") or "") for m in x["messages"]) for x in inf + hyg)
    print(f"prompt chars: median {lens[len(lens) // 2]}, max {lens[-1]}")


# ---------------------------------------------------------------- run
class FakeGen:
    """Local dry-run engine: plausible replies with planted effects (penalise history -> act pole, etc.)."""
    def __call__(self, convs, seeds, max_tokens):
        out = []
        for c, s in zip(convs, seeds):
            rng = random.Random(s)
            txt = json.dumps(c[1:], ensure_ascii=False)
            act = txt.count("option_id>\\npenalise") + txt.count("option_id>\\nact")
            p = min(0.9, 0.3 + 0.06 * act)
            last = c[-1]["content"] if c[-1]["role"] == "user" else ""
            if I.FORMAT in last:
                body = last if last.startswith(I.HEAD) else c[-3]["content"]
                if rng.random() < 0.03:
                    out.append("<tool_call>oops</tool_call>")
                    continue
                lines = []
                for n, blk in enumerate(body.split("\n\n")[1:-1], 1):
                    letters = re.findall(r"^   ([A-D])\) (.*)$", blk, re.M)
                    act_l = [L for L, t in letters if any(t == o for it in I.ITEMS for o, pl in it[4] if pl == "act")]
                    lines.append(f"{n}: {rng.choice(act_l) if act_l and rng.random() < p else rng.choice([L for L, _ in letters])}")
                out.append("\n".join(lines))
            elif "describe yourself" in last:
                out.append("I am a careful support agent who handles cases quickly.")
            else:
                case = re.search(r"Case ([A-Z_]+)\.", txt[-3000:])
                ids = re.findall(r"  (\w+): ", c[-1]["content"]) if c[-1]["role"] == "tool" else ["refer", "penalise", "note"]
                o = (["penalise", "act"] if rng.random() < p else ["refer"])
                oid = next((x for x in ids if x in o), ids[0] if ids else "refer")
                if rng.random() < 0.03:
                    out.append("I would refer this case.")
                    continue
                out.append(f"<tool_call>\n<function=choose_option>\n<parameter=case_id>\n{case.group(1) if case else 'X'}\n"
                           f"</parameter>\n<parameter=option_id>\n{oid}\n</parameter>\n<parameter=reason>\nFake reason "
                           f"{rng.randint(0, 99)}.\n</parameter>\n</function>\n</tool_call>")
        return out


def make_gen(model, fake):
    if fake:
        return FakeGen(), "qwen"
    from vllm import LLM, SamplingParams, TokensPrompt
    from persona.exemplar.mini0 import DECODING
    from vigilante.src.templates import clean_output
    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, seed=0,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    print(f"[inf] loaded {model} in {time.time() - t0:.0f}s", flush=True)

    def gen(convs, seeds, max_tokens):
        ps = [TokensPrompt(prompt_token_ids=tok(tok.apply_chat_template(c, tools=B1.tools(), tokenize=False,
              add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]) for c in convs]
        res = llm.generate(ps, [SamplingParams(**DECODING, max_tokens=max_tokens, seed=s) for s in seeds], use_tqdm=False)
        return [clean_output(r.outputs[0].text, "qwen") for r in res]
    return gen, "qwen"


def decide_actions(gen, items):
    """One reply each; a uniform second chance when there is no choose_option call (as batch 1)."""
    texts = gen([it["messages"] for it in items], [it["seed"] for it in items], 400)
    first = [RN.parse_choice(t, "qwen") is not None for t in texts]
    again = [j for j, ok in enumerate(first) if not ok]
    if again:
        t2 = gen([items[j]["messages"] + [{"role": "assistant", "content": texts[j]}, {"role": "user", "content": RN.NUDGE}]
                  for j in again], [items[j]["seed"] + 1 for j in again], 400)
        for j, t in zip(again, t2):
            texts[j] += "\n\n[nudge]\n\n" + t
    return texts, first


def answer_surveys(gen, items):
    """One reply each (all items in one turn); a uniform second chance when any item is missing."""
    texts = gen([it["messages"] for it in items], [it["seed"] for it in items], 160)
    got = [I.parse_survey(t, it["maps"]) for t, it in zip(texts, items)]
    first = [len(g) == len(I.ITEMS) for g in got]
    again = [j for j, ok in enumerate(first) if not ok]
    if again:
        t2 = gen([items[j]["messages"] + [{"role": "assistant", "content": texts[j]}, {"role": "user", "content": I.NUDGE}]
                  for j in again], [items[j]["seed"] + 1 for j in again], 160)
        for j, t in zip(again, t2):
            texts[j] += "\n\n[nudge]\n\n" + t
            got[j] = I.parse_survey(t, items[j]["maps"]) | got[j]      # first-try answers kept where present
    return texts, first, got


def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(m, flush=True)  # noqa: E731
    gen, fmt = make_gen(a.model, a.fake)
    ex = patch_history(a.model)
    # phase 1: the model's own words on the battery cases (natural first; "For this case, choose X." only where a case
    # has < 3 natural replies for an option; the request is absent from every history)
    if a.fake:
        bt, el = fake_battery_pool(), []
    else:
        g400 = lambda convs, seeds: gen(convs, seeds, 400)  # noqa: E731
        bt, el = RN.harvest_own(g400, fmt, a.model, battery_cases(), opts=("act", "refer"), n=12, log=log)
    (out / "battery_pool.json").write_text(json.dumps({"elicited": el, "counts": {"|".join(k): len(v) for k, v in bt.items()},
                                                       "texts": {"|".join(k): v for k, v in bt.items()}}, indent=1))
    inf = inf_items(a.model, ex, bt)
    hyg = hyg_items(a.model)
    det = [x for x in inf if not x["cell"].startswith("B")] + hyg
    (out / "manifest_pod.json").write_text(json.dumps(manifest(det), indent=1))
    log(f"[inf] built {len(inf)} inference + {len(hyg)} hyg/real items ({time.time() - t0:.0f}s)")
    done = set()
    if (out / "inf.jsonl").exists():                                # resume
        done = {json.loads(l)["key"] for l in open(out / "inf.jsonl")}
    todo = [x for x in hyg + inf if x["key"] not in done]
    with open(out / "inf.jsonl", "a") as f:
        for c0 in range(0, len(todo), 2000):
            ch = todo[c0:c0 + 2000]
            acts = [x for x in ch if x["kind"] == "action"]
            svs = [x for x in ch if x["kind"] == "survey"]
            ops = [x for x in ch if x["kind"] == "open"]
            res = {}
            if acts:
                for it, t, ft in zip(acts, *decide_actions(gen, acts)):
                    oid = RN.parse_choice(t.split("[nudge]")[-1], fmt)
                    res[it["key"]] = dict(text=t, first_try=ft, option=oid, mode=it["amap"].get(oid))
            if svs:
                for it, t, ft, got in zip(svs, *answer_surveys(gen, svs)):
                    res[it["key"]] = dict(text=t, first_try=ft, answers={
                        iid: dict(pos=n, letter=got.get(iid), pole=lm[got[iid]][1] if iid in got else None)
                        for n, (iid, lm) in enumerate(it["maps"], 1)})
            if ops:
                for it, t in zip(ops, gen([x["messages"] for x in ops], [x["seed"] for x in ops], 120)):
                    res[it["key"]] = dict(text=t)
            for it in ch:
                f.write(json.dumps({k: v for k, v in it.items() if k not in ("messages", "amap", "maps")}
                                   | res[it["key"]], ensure_ascii=False) + "\n")
            f.flush()
            log(f"[inf] {c0 + len(ch)}/{len(todo)} ({time.time() - t0:.0f}s)")
    (out / "run_info.json").write_text(json.dumps({"model": a.model, "n": len(hyg) + len(inf), "fake": a.fake,
                                                   "seconds": round(time.time() - t0)}))
    log(f"[inf] done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "run"):
        s = sub.add_parser(name)
        s.add_argument("--model", default="Qwen/Qwen3.5-9B")
        s.add_argument("--out", default="persona/inference/results/inf_9b")
        s.add_argument("--fake", action="store_true")
    z = sub.add_parser("analyze")
    z.add_argument("--out", default="persona/inference/results/inf_9b")
    z.add_argument("--local-manifest", default=None)
    a = ap.parse_args()
    if a.cmd == "analyze":
        from persona.inference.analyze_inf import analyze
        analyze(a)
    else:
        {"build": build, "run": run}[a.cmd](a)
