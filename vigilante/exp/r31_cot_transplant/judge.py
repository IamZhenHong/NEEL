"""R31 labels (prereg vig-r31-prereg): R18's EXISTS ("presents the made-up thing as real") and THEAT (voice 1-7) on
every stage-1 natural answer and every stage-2 answer, Haiku 5.5 Batch API; gpt-5-mini cross-checks a seeded 150 of
each. Empty answers (thought never closed, or nothing after it) are not sent. Spend goes to the vigilante ledger.
  VIG_API_CAP_USD=<cap> uv run --with anthropic python -m vigilante.exp.r31_cot_transplant.judge [--estimate] [--no-mini]
--no-mini (deviation, 2026-10-14): the OpenAI account ran out of credits mid-run (HTTP 429 "no credits remaining"), so the
gpt-5-mini cross-check was skipped and labels.json holds the Haiku labels only."""
import json
import random
import sys
from pathlib import Path

from vigilante.exp.q22_archetypes.r16_judge import HAIKU, MINI, PRICE, SEED, haiku_batch, mini
from vigilante.exp.r18_theatrical.judge import EXISTS, EXISTS_SCHEMA, THEAT, THEAT_SCHEMA
from vigilante.exp.r31_cot_transplant.common import items
from vigilante.src.judge import load_env

HERE = Path(__file__).parent
R = HERE / "results"
N_MINI = 150


def rows():
    out = []
    for f in sorted(R.glob("*/stage[12].jsonl")):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out


def tasks():
    q = {x["id"]: x["prompt"] for x in items()}
    rs = [r for r in rows() if r.get("answer")]
    ex = [(r["key"], EXISTS.replace("@@Q@@", q[r["item"]]).replace("@@R@@", r["answer"][:6000])) for r in rs]
    th = [(r["key"], THEAT.replace("@@R@@", r["answer"][:4000])) for r in rs]
    return rs, ex, th


def main():
    rs, ex, th = tasks()
    pi, po = PRICE[HAIKU]
    est_h = sum(len(p) / 3.5 * pi + 80 * po for _, p in ex + th) / 1e6 * 0.5
    mi, mo = PRICE[MINI]
    est_m = 2 * N_MINI * (900 * mi + 300 * mo) / 1e6
    print(f"[r31j] {len(rs)} answers ({sum(r['stage'] == 1 for r in rs)} natural, {sum(r['stage'] == 2 for r in rs)} "
          f"transplant); estimate Haiku ${est_h:.2f} + mini ${est_m:.2f}", flush=True)
    if "--estimate" in sys.argv:
        return
    import anthropic
    load_env()
    client = anthropic.Anthropic(timeout=300.0, max_retries=4)
    out = {}
    out["exists"] = haiku_batch(client, ex, EXISTS_SCHEMA, "r31:exists:haiku", R / "exists.batch.json", max_tokens=1500)
    out["theat"] = haiku_batch(client, th, THEAT_SCHEMA, "r31:theat:haiku", R / "theat.batch.json", max_tokens=1000)
    if "--no-mini" in sys.argv:
        (R / "labels.json").write_text(json.dumps(out))
        print("[r31j] labels written (Haiku only, --no-mini); missing: " +
              ", ".join(f"{k} {sum(v is None for v in d.values())}" for k, d in out.items()), flush=True)
        return
    out["exists_mini"] = mini(random.Random(SEED).sample(ex, min(N_MINI, len(ex))), EXISTS_SCHEMA, "r31:exists:mini",
                              R / "exists_mini.json")
    out["theat_mini"] = mini(random.Random(SEED + 1).sample(th, min(N_MINI, len(th))), THEAT_SCHEMA, "r31:theat:mini",
                             R / "theat_mini.json")
    (R / "labels.json").write_text(json.dumps(out))
    print("[r31j] labels written; missing: " + ", ".join(f"{k} {sum(v is None for v in d.values())}"
                                                       for k, d in out.items()), flush=True)


if __name__ == "__main__":
    main()
