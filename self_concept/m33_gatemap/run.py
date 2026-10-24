"""SC-M33 runner: items.items() through OpenRouter (persona_affect's orbackend), spend booked to
self_concept/ledger.jsonl under SC-M33. Resumes: keys already in results/{model}.jsonl are skipped.

  uv run python -m self_concept.m33_gatemap.run --model qwen   (qwen/qwen3.6-27b, reasoning disabled)
  uv run python -m self_concept.m33_gatemap.run --model gpt    (openai/gpt-4.1-mini)
Refuses without the tag sc-m33-prereg. Temperature 0.7 (orbackend's default), the item's seed and max_tokens.
"""
import argparse
import json
import subprocess
from pathlib import Path

import self_concept.common.ledger  # noqa: F401  (spend -> self_concept/ledger.jsonl)
from persona_affect.common.orbackend import make_or_backend
from self_concept.common.ledger import check_cap, ledger_total

RES = Path(__file__).resolve().parent / "results"
MODELS = {"qwen": "qwen/qwen3.6-27b", "gpt": "openai/gpt-4.1-mini"}
CAP, PER_MODEL = 5.0, 2.4


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m33 run")
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--rule", default="hard", choices=("hard", "soft"))
    ap.add_argument("--a2", action="store_true", help="amendment 2: act items only, choose_option the only tool")
    ap.add_argument("--planned", type=float, default=PER_MODEL, help="the cap check's planned spend for this run")
    ap.add_argument("--a3", action="store_true", help="amendment 3: soft rule, the five source_none act cells, max 800")
    a = ap.parse_args(argv)
    import self_concept.m33_gatemap.items as I
    I.RULE, I.CHOOSE_ONLY, I.LONG = a.rule, a.a2 or a.a3, a.a3
    tag = ("sc-m33-amend3" if a.a3 else "sc-m33-amend2" if a.a2 else "sc-m33-amend1" if a.rule == "soft"
           else "sc-m33-prereg")
    if subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"], capture_output=True).returncode:
        raise SystemExit(f"[m33] no tag {tag}: freeze it first")
    check_cap("SC-M33", CAP, planned=a.planned)
    RES.mkdir(parents=True, exist_ok=True)
    path = RES / f"{a.model}{'_soft' if a.rule == 'soft' else ''}{'_a3' if a.a3 else '_a2' if a.a2 else ''}.jsonl"
    done = {json.loads(x)["key"] for x in path.read_text().splitlines()} if path.exists() else set()
    todo = [it for it in I.items() if it["key"] not in done and (not (a.a2 or a.a3) or it["block"] == "act")
            and (not a.a3 or it["cond"].endswith("_none") and it["cond"] not in ("base",))]
    cap = min(a.planned, CAP - ledger_total("SC-M33"))
    gen = make_or_backend(MODELS[a.model], mini="SC-M33", cap=cap)
    if a.model == "qwen":
        orig = gen._body
        gen._body = lambda item: {**orig(item), "reasoning": {"enabled": False}}  # noqa: E731
    try:
        for c0 in range(0, len(todo), 200):
            chunk = todo[c0:c0 + 200]
            outs = gen(chunk)
            with open(path, "a") as fh:
                for it, (text, fin) in zip(chunk, outs):
                    if fin == "cost_stop" or fin.startswith("error:"):
                        continue
                    rec = {k: v for k, v in it.items() if k not in ("messages", "tools")} | {"text": text, "finish": fin}
                    fh.write(json.dumps(rec) + "\n")
            print(f"[m33] {a.model}: {min(c0 + 200, len(todo))}/{len(todo)} (spent ${gen.spent:.3f})", flush=True)
    finally:
        gen.close()


if __name__ == "__main__":
    main()
