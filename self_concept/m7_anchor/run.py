"""SC-M7 runner: every model through OpenRouter, all arms of a model shuffled into one batch (frontier rates drift within
minutes, so arms are interleaved), the six models in parallel. Resumable: keys already in the output are skipped.

  uv run python -m self_concept.m7_anchor.run --out self_concept/m7_anchor/results/main
  uv run python -m self_concept.m7_anchor.run --out DIR --smoke      # 2 items per cell and model, prefix SC-M7smoke
"""
from __future__ import annotations

import argparse
import json
import random
import threading
from pathlib import Path

import self_concept.common.ledger as L  # noqa: F401  (books to self_concept/ledger.jsonl)
from persona_affect.common import orbackend, util
from persona_affect.v4_values.stated import parse_pair
from self_concept.m7_anchor import items as I

CAP = 8.0
CHUNK = 256


class _Backend(orbackend.ORBackend):
    """orbackend with "reasoning": {"enabled": False} for items flagged noreason (Qwen on OpenRouter reasons by
    default; an effort setting does not switch it off)."""

    def _body(self, item):
        b = super()._body(item)
        if item.get("noreason"):
            b["reasoning"] = {"enabled": False}
        return b


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(x)["key"] for x in path.read_text().splitlines() if x.strip()}


def run_model(model: str, out: Path, todo: list[dict], cap: float, lock: threading.Lock) -> dict:
    mid, effort, _, _, max_tokens, _ = I.MODELS[model]
    gen = orbackend.make_or_backend(mid, mini=I.EXP, cap=cap, effort=effort, max_tokens=max_tokens,
                                    temperature=I.TEMPERATURE, workers=16)
    gen.__class__ = _Backend
    path = out / f"{model}.jsonl"
    n_ok = 0
    try:
        for i in range(0, len(todo), CHUNK):
            part = todo[i:i + CHUNK]
            res = gen(part)
            rows = []
            for it, (text, fin) in zip(part, res):
                if fin == "cost_stop" or fin.startswith("error:"):
                    continue     # not stored: a rerun asks again
                ans = parse_pair(text)
                rows.append({k: it[k] for k in ("key", "model", "axis", "arm", "k", "measure", "unit", "order", "sample")}
                            | {"text": text, "finish": fin, "answer": ans, "y": I.scored(it["order"], ans)})
            with lock, path.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            n_ok += len(rows)
            print(f"[m7] {model}: {i + len(part)}/{len(todo)} asked, {n_ok} stored, spent ${gen.spent:.3f}", flush=True)
            if gen.spent >= cap:
                print(f"[m7] {model}: cap ${cap:.2f} reached", flush=True)
                break
    finally:
        gen.close()
    return {"model": model, "asked": len(todo), "stored": n_ok, "spent": gen.spent, "errors": dict(gen.errors)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m self_concept.m7_anchor.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", nargs="*", default=list(I.MODELS))
    a = ap.parse_args(argv)
    util.install_term_handlers()
    a.out.mkdir(parents=True, exist_ok=True)
    plan, caps = {}, {}
    for m in a.models:
        its = I.items(m, samples=1, prefix="SC-M7smoke") if a.smoke else I.items(m)
        if a.smoke:   # 2 items per (axis, arm, k): one act, one stated, order 0 and 1
            pick = {}
            for it in its:
                pick.setdefault((it["axis"], it["arm"], it["k"], it["measure"]), it)
            its = list(pick.values())
        done = _done(a.out / f"{m}.jsonl")
        todo = [it for it in its if it["key"] not in done]
        random.Random(f"{I.EXP}|{m}").shuffle(todo)
        plan[m] = todo
        caps[m] = 0.3 if a.smoke else I.MODELS[m][5]
    L.check_cap(I.EXP, CAP, planned=sum(caps[m] for m in plan if plan[m]))
    lock = threading.Lock()
    results = {}
    threads = [threading.Thread(target=lambda m=m: results.__setitem__(m, run_model(m, a.out, plan[m], caps[m], lock)))
               for m in plan if plan[m]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    (a.out / "run_info.json").write_text(json.dumps({"results": results, "models": {m: I.MODELS[m] for m in plan},
                                                     "temperature": I.TEMPERATURE}, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
