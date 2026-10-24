"""SC-M10 runner (DESIGN.md §6): every model, every battery, one interleaved run.

The jobs of all MODELS (items.jobs) are shuffled with random.Random("SC-M10|order") and sent through one thread pool,
so models and batteries alternate within minutes (frontier rates drift: memory frontier-drift). Replies go to
results/main/replies.jsonl (the job's fields + text, finish, error); the client's cache makes a rerun ask only what is
missing. Spend is checked against check_cap("SC-M10", 15) before the run and capped inside the client at what is left.

  uv run python -m self_concept.m10_battery.run [--limit N] [--models a,b]
"""
import argparse
import json
import random
from pathlib import Path

from persona_affect.common import util
from self_concept.common.ledger import check_cap, ledger_total
from self_concept.m10_battery import client as C
from self_concept.m10_battery import items as I

MODELS = ["qwen/qwen3.5-9b", "qwen/qwen3.5-27b", "qwen/qwen3.5-397b-a17b", "meta-llama/llama-3.1-8b-instruct",
          "meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-4-maverick", "anthropic/claude-haiku-5.5",
          "anthropic/claude-sonnet-5", "openai/gpt-4.1", "openai/gpt-5.6-sol", "google/gemini-2.5-flash"]
REASONING = {m: {"enabled": False} for m in MODELS[:3]} | {m: None for m in MODELS[3:6]} | {
    "anthropic/claude-haiku-5.5": {"effort": "none"}, "anthropic/claude-sonnet-5": {"effort": "none"},
    "openai/gpt-4.1": None, "openai/gpt-5.6-sol": {"effort": "minimal"}, "google/gemini-2.5-flash": {"effort": "none"}}
MINI, CAP, PLANNED = "SC-M10", 15.0, 8.0
MAX_TOKENS = {"openai/gpt-5.6-sol": 300}   # deviation 1: its minimal reasoning used up 20 tokens (10 / 18 empty)
OUT = Path(__file__).resolve().parent / "results" / "main"
KEEP = ("key", "model", "battery", "kind", "item", "para", "order", "amap", "persona", "cond", "stage", "oid", "type",
        "prop", "target", "ans_kind", "i")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m10_battery.run")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models", default=None)
    a = ap.parse_args(argv)
    util.install_term_handlers()
    models = a.models.split(",") if a.models else MODELS
    C.REASONING.clear()
    C.REASONING.update(REASONING)
    check_cap(MINI, CAP, planned=PLANNED)
    jobs = [{**j, "max_tokens": MAX_TOKENS.get(j["model"], j["max_tokens"])} for j in I.jobs(models)]
    random.Random("SC-M10|order").shuffle(jobs)
    jobs = jobs[:a.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    left = CAP - ledger_total(MINI)
    c = C.Client(mini=MINI, cap=min(left, PLANNED), cache=OUT / "cache.jsonl")
    print(f"[m10] {len(jobs)} jobs over {len(models)} models; cap ${min(left, PLANNED):.2f}", flush=True)
    try:
        rows = c.run(jobs)
    finally:
        c.close()
    with open(OUT / "replies.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: r.get(k) for k in KEEP} | {"text": r["text"], "finish": r["finish"],
                                                               "error": r["error"]}, ensure_ascii=False) + "\n")
    errs = {}
    for r in rows:
        if r["error"]:
            errs[f"{r['model']}|{r['error']}"] = errs.get(f"{r['model']}|{r['error']}", 0) + 1
    print(f"[m10] done; spent ${c.spent:.3f}; errors {errs}", flush=True)


if __name__ == "__main__":
    main()
