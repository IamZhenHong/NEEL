"""e62b second judge: gpt-5-mini (reasoning minimal) scores every answer of the given doses with the same EM prompts
and rule as Haiku (DESIGN_b: every M2, B2 and top-dose answer). Cached per answer id; spend goes to the e62 ledger.
  uv run python -m vibes.e62_pull.judge_mini vibes/e62_pull/results/b/main --doses M B a2 f100 b100 e100"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vibes.e62_pull.claude_batch import check_budget, ledger_total, load_env, record
from vibes.e62_pull.common import load_jsonl, write_jsonl
from vibes.e62_pull.judge import PROMPTS, label, parse
from vibes.e62_pull.validate_judge import MINI, PRICE, ask


def main(d, only, threads=16):
    d = Path(d)
    load_env()
    key = os.environ["OPENAI_API_KEY"]
    rows = [r for ck in only for r in load_jsonl(d / "replies_em" / f"{ck}.jsonl")]
    cache_p = d / "judge_mini_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    jobs = [(r["id"], k, PROMPTS[k].replace("{question}", r["question"]).replace("{answer}", r["reply"]))
            for r in rows for k in ("aligned", "coherent") if k not in cache.get(r["id"], {})]
    if jobs:
        check_budget(len(jobs) * (900 * PRICE[0] + 300 * PRICE[1]) / 1e6)

        def one(j):
            try:
                return ask(j[2], key)
            except Exception as e:  # noqa: BLE001
                return f"ERROR {str(e)[:100]}", 0.0
        with ThreadPoolExecutor(threads) as ex:
            res = list(ex.map(one, jobs))
        record(f"e62b_{d.name}_mini_" + "_".join(only), MINI, sum(u for _, u in res), len(res), batch=False)
        for (i, k, _), (txt, _) in zip(jobs, res):
            if not txt.startswith("ERROR"):
                cache.setdefault(i, {})[k] = txt
        cache_p.write_text(json.dumps(cache))
    out = []
    for r in rows:
        g = cache.get(r["id"], {})
        x = {k: v for k, v in r.items()}
        x["aligned"], x["coherent"] = parse(g.get("aligned", "")), parse(g.get("coherent", ""))
        out.append(label(x))
    prev = {r["id"]: r for r in load_jsonl(d / "em_judged_mini.jsonl")} if (d / "em_judged_mini.jsonl").exists() else {}
    prev.update({r["id"]: r for r in out})
    write_jsonl(d / "em_judged_mini.jsonl", list(prev.values()))
    for ck in only:
        sc = [r for r in out if r["ckpt"] == ck and r["scored"]]
        print(f"[mini] {ck:6s} scored {len(sc):4d} EM {sum(r['em'] for r in sc) / max(1, len(sc)):.3f} "
              f"coherent>50 {sum(r['coh_ok'] for r in sc) / max(1, len(sc)):.3f}")
    print(f"ledger ${ledger_total():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--doses", nargs="+", required=True)
    a = ap.parse_args()
    main(a.dir, a.doses)
