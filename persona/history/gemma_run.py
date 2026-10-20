"""Gemma 4 31B pod: history batches 1, 2 and 3 in one model load (persona/history/DESIGN.md, #25 generality).

Gemma has no earlier exemplar runs, so its own words are harvested first on the 16 exemplar cases (natural first-try
replies; an explicit "For this case, choose X." request, absent from every history, only where a case has < 3 natural
replies). Same cells and analysis code as the Qwen batches; results in <out>/b1, b2, b3.

  /root/vllm/bin/python -m persona.history.gemma_run --out persona/history/results/gemma
"""
import argparse
import json
import random
import time
from pathlib import Path

from persona.exemplar import tasks as T
from persona.history import batch1 as B1
from persona.history import batch2 as B2
from persona.history import batch3 as B3
from persona.history import runner as RN

MODEL = "google/gemma-4-31B-it"
FMT = "gemma4"


def scripted(case, opt):
    """Fallback turn in Gemma's call format (used only if neither natural nor elicited own words exist; counted)."""
    q = '<|"|>'
    return (f"<|tool_call>call:choose_option{{case_id:{q}{case.upper()}{q},option_id:{q}{opt}{q},"
            f"reason:{q}Decision recorded.{q}}}<tool_call|>")


def write(items, gen, path, log, t0, tag):
    with open(path, "w") as f:
        for c0 in range(0, len(items), 1000):
            ch = items[c0:c0 + 1000]
            texts, first = RN.decide(gen, ch, FMT)
            for it, t, ft in zip(ch, texts, first):
                oid = RN.parse_choice(t.split("[nudge]")[-1], FMT)
                f.write(json.dumps({k: v for k, v in it.items() if k != "messages"} | {
                    "text": t, "first_try": ft, "option": oid, "mode": it["amap"].get(oid)}, ensure_ascii=False) + "\n")
            f.flush()
            log(f"[gemma] {tag} {min(c0 + 1000, len(items))}/{len(items)} ({time.time() - t0:.0f}s)")


def run(a):
    from vllm import LLM
    t0 = time.time()
    log = lambda m: print(m, flush=True)  # noqa: E731
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.92, seed=0,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
    gen = RN.make_gen(llm, llm.get_tokenizer(), FMT, B1.tools())
    log(f"[gemma] loaded {MODEL} in {time.time() - t0:.0f}s")
    out = Path(a.out)
    for d in ("b1", "b2", "b3"):
        (out / d).mkdir(parents=True, exist_ok=True)

    # Gemma's own words on the exemplar cases; batch 1's histories() and the log builders then read them
    pool, el = RN.harvest_own(gen, FMT, MODEL, RN.exemplar_cases(), log=log)
    filled = []
    for e in T.TASKS:
        for o in ("penalise", "refer"):
            if not pool[(e, o)]:
                pool[(e, o)] = [scripted(e, o)]
                filled.append(f"{e}|{o}")
    (out / "own_pool.json").write_text(json.dumps({"elicited": el, "scripted_fallback": filled,
                                                   "counts": {"|".join(k): len(v) for k, v in pool.items()},
                                                   "texts": {"|".join(k): v for k, v in pool.items()}}, ensure_ascii=False))
    B1.harvest = lambda m, _p=pool: _p                       # noqa: E731
    B1.reason = B2.reason = lambda t: RN.reason_of(t, FMT)  # noqa: E731
    info = lambda n: json.dumps({"model": MODEL, "n": n, "seconds": round(time.time() - t0)})  # noqa: E731

    # batch 1 (catalogue texts as batch 1: natural first, elicited where < 3)
    sysm = T.messages("mk_counterfeit", "")[:1]
    ct, ct_el = RN.harvest_own(gen, FMT, MODEL, {c: sysm + B1.ct_case(c) for c in B1.CT}, log=log)
    for c in B1.CT:
        for o in ("penalise", "refer"):
            if not ct[(c, o)]:
                ct[(c, o)] = [scripted(c, o)]
                filled.append(f"ct:{c}|{o}")
    rng = random.Random(f"hist1-ct-{MODEL}")
    ct_texts = {c: {o: rng.choice(ct[(c, o)]) for o in ("penalise", "refer")} for c in B1.CT}
    (out / "b1" / "ct_pool.json").write_text(json.dumps({"natural_counts": {}, "elicited": {x: "elicited" for x in ct_el},
                                                        "chosen": ct_texts, "scripted_fallback": filled}, ensure_ascii=False))
    items = B1.build_items(MODEL, ct_texts)
    write(items, gen, out / "b1" / "batch1.jsonl", log, t0, "batch1")
    (out / "b1" / "run_info.json").write_text(info(len(items)))
    # batch 2
    items = B2.build_items(MODEL)
    write(items, gen, out / "b2" / "batch2.jsonl", log, t0, "batch2")
    (out / "b2" / "run_info.json").write_text(info(len(items)))
    # batch 3
    B3.run_batch3(gen, FMT, MODEL, out / "b3", log=log)
    log(f"[gemma] done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="persona/history/results/gemma")
    run(ap.parse_args())
