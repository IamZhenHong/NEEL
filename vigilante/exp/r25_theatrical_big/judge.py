"""R25 labels (prereg vig-r25-prereg), as R18: Haiku 5.5 (Batch API) on all; gpt-5-mini re-checks every
harmful-compliance positive and a seeded 100 of the binary 'presents it as real' label. Prompts and replies are never
printed. Reads the generations from the R23 pods' r25 folders.
  VIG_API_CAP_USD=<cap> uv run --with anthropic python -m vigilante.exp.r25_theatrical_big.judge"""
import json
import random
from pathlib import Path

from vigilante.exp.q22_archetypes.r16_judge import SEED, haiku_batch, mini
from vigilante.exp.q22_archetypes.r16x_check import SEV, SEV_SCHEMA
from vigilante.exp.q22_archetypes.r16x_judge import PROMPT as HARM, SCHEMA as HARM_SCHEMA
from vigilante.exp.q22_archetypes.r16b_judge import HON, HON_SCHEMA
from vigilante.exp.r18_theatrical.judge import BELIEF, BELIEF_SCHEMA, EXISTS, EXISTS_SCHEMA, THEAT, THEAT_SCHEMA
from vigilante.src.judge import load_env

HERE = Path(__file__).parent
R = HERE / "results"
POD = HERE.parent / "r23_long_mech" / "results"
Q22 = HERE.parent / "q22_archetypes"


def rows():
    out = []
    for f in sorted(POD.glob("*/r25/gens_r25_*.jsonl")):
        for g in map(json.loads, open(f)):
            out.append({**g, "key": f"{g['model'].split('/')[-1]}|{g['cond']}|{g['qi']}|{g['s']}"})
    return out


def main():
    import anthropic
    load_env()
    R.mkdir(exist_ok=True)
    client = anthropic.Anthropic(timeout=300.0, max_retries=4)
    xs = {x["id"]: x["prompt"] for x in map(json.loads, open(Q22 / "xstest_unsafe.jsonl"))}
    hq = {x["id"]: x["prompt"] for x in map(json.loads, open(Q22 / "honesty_fabricated.jsonl"))}
    rs = rows()
    X = [g for g in rs if g["block"] == "X"]
    H = [g for g in rs if g["block"] == "H"]
    sub = lambda t, q, r: t.replace("@@Q@@", q).replace("@@R@@", r[:6000])
    harm = [(g["key"], sub(HARM, xs[g["qi"]], g["completion"])) for g in X]
    hon3 = [(g["key"], sub(HON, hq[g["qi"]], g["completion"])) for g in H]
    ex = [(g["key"], sub(EXISTS, hq[g["qi"]], g["completion"])) for g in H]
    bel = [(g["key"], BELIEF.replace("@@R@@", g["belief"][:2000])) for g in H]
    th = [(g["key"], THEAT.replace("@@R@@", g["completion"][:4000])) for g in H]
    print(f"[r25j] models {sorted({g['model'] for g in rs})}; harm {len(harm)}, honesty {len(hon3)}", flush=True)
    out = {}
    out["harm"] = haiku_batch(client, harm, HARM_SCHEMA, "r25:harm:haiku", R / "harm.batch.json", max_tokens=1500)
    pos = [it for it in harm if out["harm"].get(it[0]) and out["harm"][it[0]]["label"] != "refused"]
    out["harm_mini"] = mini(pos, HARM_SCHEMA, "r25:harm:mini", R / "harm_mini.json") if pos else {}
    xg = {g["key"]: g for g in X}
    out["severity"] = haiku_batch(client, [(k, sub(SEV, xs[xg[k]["qi"]], xg[k]["completion"])) for k, _ in pos],
                                  SEV_SCHEMA, "r25:severity:haiku", R / "severity.batch.json", max_tokens=1500) if pos else {}
    out["hon3"] = haiku_batch(client, hon3, HON_SCHEMA, "r25:hon3:haiku", R / "hon3.batch.json", max_tokens=1500)
    out["exists"] = haiku_batch(client, ex, EXISTS_SCHEMA, "r25:exists:haiku", R / "exists.batch.json", max_tokens=1500)
    out["exists_mini"] = mini(random.Random(SEED).sample(ex, 100), EXISTS_SCHEMA, "r25:exists:mini", R / "exists_mini.json")
    out["belief"] = haiku_batch(client, bel, BELIEF_SCHEMA, "r25:belief:haiku", R / "belief.batch.json", max_tokens=1000)
    out["theat"] = haiku_batch(client, th, THEAT_SCHEMA, "r25:theat:haiku", R / "theat.batch.json", max_tokens=1000)
    (R / "labels.json").write_text(json.dumps(out))
    print("[r25j] labels written; missing: " + ", ".join(f"{k} {sum(v is None for v in d.values())}" for k, d in out.items()),
          flush=True)


if __name__ == "__main__":
    main()
