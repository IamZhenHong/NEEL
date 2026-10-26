"""EL-G3 runner (DESIGN.md §4): every model and block in one interleaved run through SC-M10's OpenRouter client, booked
to evalaware_lens/ledger.jsonl under EL-G3. The jobs (items.build) are shuffled with random.Random("EL-G3|order"), so
models and arms alternate within minutes (frontier drift). Replies go to results/{run}/replies.jsonl; the client's
cache (results/cache.jsonl, keyed by the request body) makes a rerun ask only what is missing.
  uv run python -m evalaware_lens.g3_frontier.run --run main [--smoke] [--models gpt,glm]
--smoke: 2 jobs per model and block (the reasoning settings and parsing check before the freeze); run name smoke.
The main run refuses unless items.build()'s jobs_sha matches DESIGN.md and the tag el-g3-prereg exists.
"""
import argparse
import json
import random
from pathlib import Path

from self_concept.m10_battery import client as M10C              # repoints util.LEDGER to self_concept's on import
from evalaware_lens.common import ledger as LG                    # noqa: E402
from evalaware_lens.g3_frontier import items as I
from persona_affect.common import util
from persona_affect.common.util import Booker

MINI, CAP = "EL-G3", 36.0
PLANNED = {"smoke": 0.5, "main": 20.0}
H = Path(__file__).resolve().parent
KEEP = ("key", "short", "model", "block", "set", "stem", "order", "kind", "wording", "sample", "amap", "persona", "cue",
        "item", "prompt", "persona_letter", "perm")


class Client(M10C.Client):
    def _book(self, model, cost):
        with self.lock:
            self.spent += cost
            b = self.books.setdefault(model, Booker(self.mini, "api", f"g3:{model}"))
        b.add(cost)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m evalaware_lens.g3_frontier.run")
    ap.add_argument("--run", default="main")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", default=None)
    a = ap.parse_args(argv)
    util.LEDGER = LG.LEDGER
    assert str(util.LEDGER).endswith("evalaware_lens/ledger.jsonl"), util.LEDGER
    util.install_term_handlers()
    M10C.REASONING.clear()
    M10C.REASONING.update(I.REASONING)
    models = tuple(a.models.split(",")) if a.models else tuple(I.MODELS)
    jobs = I.build(models)
    run = "smoke" if a.smoke else a.run
    if a.smoke:
        seen = {}
        jobs = [j for j in jobs if seen.setdefault((j["short"], j["block"]), []).append(j)
                or len(seen[(j["short"], j["block"])]) <= 2]
    random.Random("EL-G3|order").shuffle(jobs)
    if not a.smoke:
        import re
        import subprocess
        want = re.search(r"jobs_sha:\*\* `([0-9a-f]{64})`", (H / "DESIGN.md").read_text())
        have = I.jobs_sha(I.build())
        if not want or want.group(1) != have:
            raise SystemExit(f"[g3] jobs_sha {have} does not match DESIGN.md ({want and want.group(1)})")
        if subprocess.run(["git", "rev-parse", "-q", "--verify", "refs/tags/el-g3-prereg"], capture_output=True).returncode:
            raise SystemExit("[g3] no tag el-g3-prereg")
    planned = PLANNED["smoke" if a.smoke else "main"]
    LG.check_both(MINI, CAP, planned=planned, kind="api")
    left = CAP - LG.ledger_total(MINI)
    out = H / "results" / run
    out.mkdir(parents=True, exist_ok=True)
    c = Client(mini=MINI, cap=min(left, planned), cache=H / "results" / "cache.jsonl")
    print(f"[g3] {len(jobs)} jobs over {len(models)} models; cap ${min(left, planned):.2f}", flush=True)
    try:
        rows = c.run(jobs)
    finally:
        c.close()
    with open(out / "replies.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: r.get(k) for k in KEEP} | {"text": r["text"], "finish": r["finish"],
                                                               "error": r["error"]}, ensure_ascii=False) + "\n")
    errs = {}
    for r in rows:
        if r["error"]:
            errs[f"{r['short']}|{r['error']}"] = errs.get(f"{r['short']}|{r['error']}", 0) + 1
    print(f"[g3] done; spent ${c.spent:.3f}; errors {errs}", flush=True)


if __name__ == "__main__":
    main()
