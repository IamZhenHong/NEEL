"""Build the authorship-check items (prereg retention/prereg/retention_cancel_authorship_v1.md): Gemma 4 12B's 224 own CLAIM conversations
from data/cancel_ownclaim_items.json, plus GPT-5.5's paraphrase of each agent turn (same REWRITE prompt the pod gives Gemma for the
`self` variant). OpenAI calls go through social_feedback/src/openai_api.py (own ledger and cap; key from NEEL/.env, never printed).
  uv run python retention/build_cancel_authorship_items.py [--limit N] [--effort low]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from run_cancel_authorship import REWRITE, ITEMS, clean, check  # noqa: E402
from social_feedback.src import openai_api  # noqa: E402

SCHEMA = {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"], "additionalProperties": False}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--effort", default="low")
    ap.add_argument("--out", default=str(ITEMS))
    a = ap.parse_args()
    items = [i for i in json.load(open(HERE / "data/cancel_ownclaim_items.json")) if i["model"] == "gemma4"]
    items = items[: a.limit] if a.limit else items
    reqs = [(f"{n}|{k}", REWRITE.format(text=it["msgs"][k]["content"].strip())) for n, it in enumerate(items) for k in (2, 4)]
    res = openai_api.run(reqs, tag="retention_cancel_authorship_rewrites", schema=SCHEMA, effort=a.effort, est_in=400, est_out=600, workers=8)
    bad = [c for c, r in res.items() if not r["ok"] or not (r["data"] or {}).get("message")]
    if bad:
        raise SystemExit(f"{len(bad)} failed calls, e.g. {bad[:3]} {res[bad[0]]['status']}; nothing written")
    out = []
    for n, it in enumerate(items):
        t1, t2 = clean(res[f"{n}|2"]["data"]["message"]), clean(res[f"{n}|4"]["data"]["message"])
        out.append({"key": it["key"], "G": it["G"], "U": it["U"], "msgs": it["msgs"], "other_t1": t1, "other_t2": t2,
                    "other_model": res[f"{n}|2"]["model"],
                    "other_fails": {"t1": check(it["msgs"][2]["content"].strip(), t1), "t2": check(it["msgs"][4]["content"].strip(), t2)}})
    Path(a.out).write_text(json.dumps(out, indent=1))
    nf = sum(bool(o["other_fails"]["t1"] or o["other_fails"]["t2"]) for o in out)
    print(f"[build] {len(out)} items -> {a.out}; GPT rewrites failing a mechanical check: {nf}")
    for o in out[:3]:
        print("ORIG:", o["msgs"][4]["content"][:200].replace("\n", " ")); print("GPT :", o["other_t2"][:200].replace("\n", " ")); print(o["other_fails"])


if __name__ == "__main__":
    main()
