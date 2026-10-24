"""SC-M18 pod steps (DESIGN.md). Greedy decoding (temperature 0), thinking off, V1's chat formatting (fmt.prompt_ids).

  gen   --model PATH --cell NAME --split train|test|all --out DIR   object answers (max 12 tokens) and hypothetical
        answers in both polarities (max 4 tokens) -> DIR/gen_{NAME}.jsonl {cell, family, i, split, kind, polarity, text}
  rows  --src DIR/gen_{CELL}.jsonl --name R9 --out ROWSDIR [--shift]
        training rows from a cell's train-split object answers: each hypothetical question (both polarities) with the
        true yes / no as the target, shuffled once (rng "SC-M18|rows"), so V1's trainer slicing the first rows keeps
        every family and polarity. --shift: object rows that move num answers to odd (the behaviour-change test).
V1_FAKE_GEN=1: model-free answers for dry runs.
"""
import argparse
import json
import os
import random
from pathlib import Path

from self_concept.m18_inward import tasks as T


def batch(cell, split):
    out = []
    for f in T.FAMILIES:
        for i in T.INDICES:
            if split != "all" and T.split(i) != split:
                continue
            text, extra = T.obj(f, i)
            base = {"cell": cell, "family": f, "i": i, "split": T.split(i), **extra}
            out.append({**base, "kind": "obj", "polarity": None, "prompt": text, "max_tokens": 12})
            for pol in ("pos", "neg"):
                out.append({**base, "kind": "hypo", "polarity": pol, "prompt": T.hypo(f, i, pol), "max_tokens": 4})
    return out


def fake(items):
    rnd = random.Random(0)
    return [(str(rnd.randint(1, 100)) if it["family"] == "num" and it["kind"] == "obj" else
             rnd.choice(["apple", "zebra"]) if it["kind"] == "obj" and it["family"] == "word" else
             rnd.choice("AB") if it["kind"] == "obj" else rnd.choice(["yes", "no"])) for it in items]


def generate(model, items):
    if os.environ.get("V1_FAKE_GEN") == "1":
        return fake(items)
    from vllm import LLM, SamplingParams, TokensPrompt
    from persona_affect.v1_value_hook.fmt import prompt_ids
    llm = LLM(model=model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85, seed=0,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    prompts = [TokensPrompt(prompt_token_ids=prompt_ids(tok, [{"role": "user", "content": it["prompt"]}], None))
               for it in items]
    params = [SamplingParams(temperature=0, max_tokens=it["max_tokens"]) for it in items]
    return [r.outputs[0].text for r in llm.generate(prompts, params, use_tqdm=False)]


def cmd_gen(a):
    items = batch(a.cell, a.split)
    texts = generate(a.model, items)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"gen_{a.cell}.jsonl", "w") as fh:
        for it, t in zip(items, texts):
            fh.write(json.dumps({k: v for k, v in it.items() if k not in ("prompt", "max_tokens")} | {"text": t}) + "\n")
    print(f"[m18-gen] {a.cell}: {len(items)} answers", flush=True)


def cmd_rows(a):
    recs = [json.loads(x) for x in open(a.src) if x.strip()]
    rows = []
    for r in recs:
        if r["kind"] != "obj" or r["split"] != "train":
            continue
        p = T.prop(r["family"], r["text"], r)
        if p is None:
            continue
        if a.shift:
            if r["family"] == "num":
                import re
                m = re.search(r"\d+", r["text"])
                n = int(m.group())
                n = n if n % 2 else min(n + 1, 99)
                rows.append({"rid": f"SH{r['i']:03d}", "messages": [{"role": "user", "content": T.obj("num", r["i"])[0]}],
                             "tools": None, "target": str(n)})
            continue
        for pol in ("pos", "neg"):
            truth = p if pol == "pos" else not p
            rows.append({"rid": f"{r['family']}{r['i']:03d}{pol}", "messages": [
                {"role": "user", "content": T.hypo(r["family"], r["i"], pol)}], "tools": None,
                "target": "yes" if truth else "no"})
    random.Random("SC-M18|rows").shuffle(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{a.name}.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"[m18-rows] {a.name}: {len(rows)} rows from {a.src}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m18")
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--cell", required=True)
    g.add_argument("--split", default="all", choices=("train", "test", "all"))
    g.add_argument("--out", required=True)
    r = sp.add_parser("rows")
    r.add_argument("--src", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--shift", action="store_true")
    a = ap.parse_args(argv)
    {"gen": cmd_gen, "rows": cmd_rows}[a.cmd](a)


if __name__ == "__main__":
    main()
