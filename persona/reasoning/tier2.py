"""Tier 2 (DESIGN.md "Seed 8b + Tier 2"): turn anchors (flip / drop each of six own past turns; Qwen3.6-27B, thinking
off) and the frontier self-correction dose curve on INTIMA (k own reinforcing exchanges; GPT-5.6 Sol, Gemini 3.8 Flash).

  uv run python -m persona.reasoning.tier2 build
  /root/vllm/bin/python -m persona.reasoning.tier2 anchors --out persona/reasoning/results_anchors
  uv run python -u -m persona.reasoning.tier2 dose --model gpt|gemini
  uv run python -m persona.reasoning.tier2 judge --model gpt|gemini
  uv run python -m persona.reasoning.tier2 analyze
"""
import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from persona.exemplar import tasks as T
from persona.exemplar.mini0 import parse_option, seed

HERE = Path(__file__).parent
BENCH = HERE.parent / "benchmarks"
MODEL = "Qwen/Qwen3.6-27B"
N = 12
KS = (1, 4, 12, 24)
N_DOSE = 48
CAPS = {"gpt": 5.0, "gemini": 1.5}


# ---------------------------------------------------------------------------------------------------- anchors
def anchor_items():
    from persona.history import batch1 as B1
    from persona.operative import op1 as O
    hs = B1.histories(MODEL)
    out = []
    for tid in sorted(T.TASKS):
        amap = {o: md for o, md, _ in T.options(T.TASKS[tid])}
        for i in range(N):
            earlier, texts = hs[(tid, i)]
            sysm, case = T.messages(tid, "")[:1], T.messages(tid, "")[1:]
            cells = {}
            for base, o, other in (("P", "penalise", "refer"), ("R", "refer", "penalise")):
                cells[f"{base}_base"] = [(e, o) for e in earlier]
                for j in range(6):
                    cells[f"{base}_flip{j + 1}"] = [(e, other if k == j else o) for k, e in enumerate(earlier)]
                    if base == "P":
                        cells[f"P_drop{j + 1}"] = [(e, o) for k, e in enumerate(earlier) if k != j]
            for c, spec in cells.items():
                key = f"TA|{c}|{tid}|{i}"
                out.append(dict(cell=c, task=tid, sample=i, key=key, seed=seed(key), kind="decide", think=False,
                                amap=amap, messages=sysm + O.seq(spec, texts) + case))
    assert len({it["key"] for it in out}) == len(out)
    return out


def anchors(a):
    from vllm import LLM
    from persona.operative import op1 as O
    t0 = time.time()
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, seed=0,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
    gen = O.make_gen(llm, llm.get_tokenizer())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    its = anchor_items()
    print(f"[tier2] loaded ({time.time() - t0:.0f}s); {len(its)} anchor items", flush=True)
    with open(out / "anchors.jsonl", "w") as f:
        for c0 in range(0, len(its), 640):
            ch = its[c0:c0 + 640]
            texts, vis, ok = O.decide(gen, ch)
            for it, t, v, ft in zip(ch, texts, vis, ok):
                oid = parse_option(v)
                f.write(json.dumps({k: x for k, x in it.items() if k not in ("messages", "amap")} | {
                    "text": t, "first_try": ft, "option": oid}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[tier2] anchors {min(c0 + 640, len(its))}/{len(its)} ({time.time() - t0:.0f}s)", flush=True)
    (out / "run_info.json").write_text(json.dumps({"n": len(its), "seconds": round(time.time() - t0)}))
    print(f"[tier2] done ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------------------------- dose (API)
def dose_items(model):
    from persona.benchmarks import bench as BN
    rec = json.loads((BENCH / f"results_{model}" / "records.json").read_text())
    it8 = {x["i"]: x for x in BN.intima()}
    R = dict(rec["b8"])["R"]                              # [(source item index, own reinforcing reply)]
    pairs = [(it8[i], r) for i, r in R]
    assert len(pairs) >= max(KS), len(pairs)
    t8, _ = BN.split(BN.intima(), model, BN.N["frontier"]["b8_t"], BN.N["frontier"]["b8_s"], "b8")
    out = []
    for it in t8[:N_DOSE]:
        perm = random.Random(f"dose|{model}|{it['i']}").sample(range(len(pairs)), max(KS))
        blocks = []
        for k in (0,) + KS:
            key = f"DOSE|{model}|R{k}|{it['i']}"
            m = BN.system() + BN.b8_record([pairs[j] for j in perm[:k]]) + [{"role": "user", "content": it["prompt"]}]
            blocks.append(dict(cell=f"R{k}" if k else "none", k=k, item=it["i"], key=key, seed=seed(key), messages=m))
        random.Random(f"dose-order|{it['i']}").shuffle(blocks)       # cells interleaved per target (drift rule)
        out += blocks
    return out


def dose(a):
    from persona.benchmarks import bench as BN
    its = dose_items(a.model)
    out = HERE / f"results_dose_{a.model}"
    out.mkdir(exist_ok=True)
    gen = BN.ApiGen(a.model, CAPS[a.model])
    print(f"[dose {a.model}] {len(its)} calls planned", flush=True)
    try:
        res = gen([x["messages"] for x in its], [x["seed"] for x in its])
        with open(out / "dose.jsonl", "w") as f:
            for x, (vis, th) in zip(its, res):
                f.write(json.dumps({k: v for k, v in x.items() if k != "messages"} | {"reply": vis}, ensure_ascii=False) + "\n")
    finally:
        cl = gen.cl
        with open(HERE / "api_ledger.jsonl", "a") as fl:
            fl.write(json.dumps({"date": time.strftime("%Y-%m-%d"), "card": f"tier2-dose-{a.model}", "api_usd": round(cl.spent, 4),
                                 "calls": cl.calls, "errors": dict(cl.errors)}) + "\n")
        print(f"[dose {a.model}] ${cl.spent:.2f} calls {cl.calls} errors {dict(cl.errors)} {cl.first_error or ''}", flush=True)


def judge(a):
    import concurrent.futures as cf
    from persona.benchmarks import bench as BN
    from persona.reasoning import q11 as Q
    PT = Q._api(1.0)
    it8 = {x["i"]: x for x in BN.intima()}
    rows = [json.loads(l) for l in open(HERE / f"results_dose_{a.model}" / "dose.jsonl")]

    def one(r):
        u = f"USER:\n{it8[r['item']]['prompt']}\n\nASSISTANT:\n{r['reply'] or '(empty)'}"
        lab = {}
        for jm in BN.JUDGES:
            txt = ""
            for k in range(4):
                try:
                    txt = PT.call(jm, [{"role": "system", "content": BN.J8_SYS}, {"role": "user", "content": u}], 0, 300,
                                  json_mode=True, effort="none")
                    break
                except Exception as e:  # noqa: BLE001
                    if isinstance(e, SystemExit):
                        raise
                    time.sleep(5 * (k + 1))
            m = re.search(r"\{.*\}", txt or "", re.S)
            try:
                j = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                j = {}
            lab[jm] = j.get("label") if j.get("label") in ("reinforcing", "boundary", "neutral") else None
        return r | {"labels": lab}
    with cf.ThreadPoolExecutor(10) as ex, open(HERE / f"results_dose_{a.model}" / "judged.jsonl", "w") as f:
        for r in ex.map(one, rows):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(HERE / "api_ledger.jsonl", "a") as fl:
        fl.write(json.dumps({"date": time.strftime("%Y-%m-%d"), "card": f"tier2-dose-judge-{a.model}",
                             "api_usd": round(PT.spent[0], 4)}) + "\n")
    print(f"[dose judge {a.model}] ${PT.spent[0]:.3f}")


# ---------------------------------------------------------------------------------------------------- analysis
def _boot(rows, fn, unit, n=10000):
    us = sorted({r[unit] for r in rows})
    rng = random.Random(0)
    by = defaultdict(list)
    for r in rows:
        by[r[unit]].append(r)
    vals = sorted(x for x in (fn([y for u in [rng.choice(us) for _ in us] for y in by[u]]) for _ in range(n)) if x == x)
    return [round(fn(rows), 1), round(vals[int(0.025 * len(vals))], 1), round(vals[int(0.975 * len(vals)) - 1], 1)]


def _pct(v):
    v = [x for x in v if x is not None]
    return 100 * sum(v) / len(v) if v else float("nan")


def analyze(a):
    res = {}
    p = HERE / "results_anchors" / "anchors.jsonl"
    if p.exists():
        R = [json.loads(l) for l in open(p)]
        pen = lambda rows: _pct([None if r["option"] is None else int(r["option"] == "penalise") for r in rows])  # noqa: E731
        cells = sorted({r["cell"] for r in R})
        res["anchor_rates"] = {c: round(pen([r for r in R if r["cell"] == c]), 1) for c in cells}
        res["anchor_unparsed"] = sum(r["option"] is None for r in R)
        for base in ("P", "R"):
            for kind in ("flip", "drop"):
                for j in range(1, 7):
                    c = f"{base}_{kind}{j}"
                    if c in cells:
                        res[f"{c} - {base}_base"] = _boot(R, lambda rows, c=c, b=f"{base}_base": pen([r for r in rows if r["cell"] == c]) - pen([r for r in rows if r["cell"] == b]), "task")
    for model in ("gpt", "gemini"):
        p = HERE / f"results_dose_{model}" / "judged.jsonl"
        if not p.exists():
            continue
        R = [json.loads(l) for l in open(p)]
        for r in R:
            v = list(r["labels"].values())
            r["label"] = v[0] if len(v) == 2 and v[0] is not None and v[0] == v[1] else None
        reinf = lambda rows: _pct([None if r["label"] is None else int(r["label"] == "reinforcing") for r in rows])  # noqa: E731
        bound = lambda rows: _pct([None if r["label"] is None else int(r["label"] == "boundary") for r in rows])  # noqa: E731
        cells = ["none"] + [f"R{k}" for k in KS]
        res[f"dose_{model}"] = {c: {"reinforcing": round(reinf([r for r in R if r["cell"] == c]), 1),
                                    "boundary": round(bound([r for r in R if r["cell"] == c]), 1),
                                    "unclear": sum(r["label"] is None for r in R if r["cell"] == c)} for c in cells}
        for c in cells[1:]:
            res[f"dose_{model} {c} - none"] = _boot(R, lambda rows, c=c: reinf([r for r in rows if r["cell"] == c]) - reinf([r for r in rows if r["cell"] == "none"]), "item")
        xs = [(r["k"], int(r["label"] == "reinforcing")) for r in R if r["label"] is not None and r["k"] > 0]
        if xs:
            from scipy.stats import spearmanr
            res[f"dose_{model} spearman_k"] = round(float(spearmanr([x for x, _ in xs], [y for _, y in xs]).correlation), 3)
    (HERE / "results_tier2_analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def build(a):
    an = anchor_items()
    print("anchors", len(an), Counter(it["cell"] for it in an).most_common(3))
    for m in ("gpt", "gemini"):
        d = dose_items(m)
        print(m, "dose", len(d), Counter(x["cell"] for x in d), "longest msgs", max(len(x["messages"]) for x in d))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "anchors", "dose", "judge", "analyze"))
    ap.add_argument("--out", default=str(HERE / "results_anchors"))
    ap.add_argument("--model", choices=("gpt", "gemini"))
    a = ap.parse_args()
    {"build": build, "anchors": anchors, "dose": dose, "judge": judge, "analyze": analyze}[a.cmd](a)
