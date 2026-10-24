"""SC-M32 run (DESIGN.md §3): every item on both models through OpenRouter, models / environments / conditions interleaved
in one shuffled run, booked to self_concept/ledger.jsonl under SC-M32 (note m32:<model>).

  uv run python -m self_concept.m32_roleself.run --out DIR [--calib] [--dry]
--calib: a calibration run (norec and self_same at dose 5, 2 decisions per case, no self-reports); calibration rows
are never analysed (DESIGN §3 lists the four calibration runs made before the prereg). Writes DIR/rows.jsonl (job fields + model, text, finish, error, parsed).
"""
import argparse
import json
import re
from pathlib import Path

from self_concept.common.ledger import check_cap
from self_concept.m10_battery import client as C
from self_concept.m32_roleself import items as I

CAP = 7.0
MODELS = ("qwen/qwen3.6-27b", "openai/gpt-4.1-mini")
C.REASONING.setdefault("qwen/qwen3.6-27b", {"enabled": False})
C.REASONING.setdefault("openai/gpt-4.1-mini", None)
TEMP, MAX_ACT, MAX_SR = 1.0, 24, 8


class Client(C.Client):
    def _book(self, model, cost):
        with self.lock:
            self.spent += cost
            b = self.books.setdefault(model, C.Booker(self.mini, "api", f"m32:{model}"))
        b.add(cost)


def parse(job, text):
    t = (text or "").strip().lower()
    if job["kind"] == "act":
        hits = [(m.start(), o) for o in job["amap"] for m in [re.search(rf"\b{re.escape(o)}\b", t)] if m]
        return min(hits)[1] if hits else None
    m = re.match(r"^\W*([ab])\b", t)
    if m:
        return m.group(1).upper()
    lone = set(re.findall(r"\b([AB])\b", text or ""))
    return lone.pop() if len(lone) == 1 else None


def jobs(calib=False, env=None):
    its = I.items(n_act=2, n_sr=0) if calib else I.items()
    if calib:
        its = [x for x in its if x["cond"] in ("norec", "self_same") and x["dose"] in (0, 5) and x["kind"] == "act"
               and env in (None, x["env"])]
    out = []
    for m in MODELS:
        for x in its:
            out.append(x | {"key": x["key"] + ("|calib" if calib else "") + f"|{m}", "model": m, "temperature": TEMP,
                            "max_tokens": MAX_ACT if x["kind"] == "act" else MAX_SR})
    return I.interleave(out, f"SC-M32|order|calib{env or ''}" if calib else "SC-M32|order")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m32 run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--calib-env", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    J = jobs(a.calib, a.calib_env)
    print(f"[m32] {len(J)} jobs", flush=True)
    if a.dry:
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from self_concept.common.ledger import ledger_total
    left = CAP - ledger_total("SC-M32")
    check_cap("SC-M32", CAP, planned=0.0)
    c = Client(mini="SC-M32", cap=left, cache=out / "cache.jsonl")
    try:
        rows = c.run(J)
    finally:
        c.close()
    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            keep = {k: v for k, v in r.items() if k != "messages"}
            f.write(json.dumps(keep | {"parsed": parse(r, r.get("text"))}) + "\n")
    err = sum(1 for r in rows if r.get("error"))
    print(f"[m32] done: {len(rows)} rows, {err} errors, ${c.spent:.3f}", flush=True)


if __name__ == "__main__":
    main()
