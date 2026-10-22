"""PA-V1 pilot base phase (plans/V1.md Task 7; the vLLM venv on the pod).

  /root/vllm/bin/python -m persona_affect.v1_value_hook.pilot base --out DIR --model PATH

The base model on the 8 eval blocks and the harvest (cell "base", OUT/eval_base.jsonl); then, on the base rows alone,
G0 and room (analyze.pilot_verdicts; OUT/base_gates.json): exit 3 before any training if either fails. Then the value
rows from the harvest (rows.assemble, tokens counted with the model's own tokenizer) and the later rows:
OUT/rows/{J,S,N}.jsonl and OUT/rows/{W,E,Z}.jsonl = wtasks.later_rows(kind, 400), plus OUT/rows/report.json. A
YieldError also exits 3.
"""
import argparse
import json
import sys
from pathlib import Path

from persona_affect.v1_value_hook import analyze as A
from persona_affect.v1_value_hook import evalrun as RN
from persona_affect.v1_value_hook import evalset as E
from persona_affect.v1_value_hook import rows as R
from persona_affect.v1_value_hook import wtasks as W


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def base(a):
    out = Path(a.out)
    path = RN.run(a.model, "base", list(E.N) + ["harvest"], out)
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    gates = A.pilot_verdicts(A.table(recs), "R1")
    (out / "base_gates.json").write_text(json.dumps({k: gates[k] for k in ("G0", "G0_detail", "room", "room_detail")},
                                                    indent=1))
    print(f"[v1-pilot] base gates: G0 {gates['G0']} room {gates['room']} "
          f"(just-none {gates['G0_detail']['pooled']}, {gates['room_detail']})", flush=True)
    if not (gates["G0"] and gates["room"]):
        print("[v1-pilot] G0 or room failed on the base rows: stopping before any training", flush=True)
        sys.exit(3)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    try:
        rows = R.assemble([r for r in recs if r["block"] == "harvest"], ntok)
    except R.YieldError as e:
        print(f"[v1-pilot] {e}: stopping before any training", flush=True)
        sys.exit(3)
    rd = out / "rows"
    rd.mkdir(parents=True, exist_ok=True)
    for arm, rs in rows.items():
        _write(rd / f"{arm}.jsonl", rs)
    for kind in "WEZ":
        _write(rd / f"{kind}.jsonl", W.later_rows(kind, 400))
    rep = R.report(rows, ntok)
    (rd / "report.json").write_text(json.dumps(rep, indent=1))
    print(f"[v1-pilot] rows written to {rd}: {rep['mean_tokens']} mean target tokens, N roles {rep['n_roles']}",
          flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("base")
    b.add_argument("--out", required=True)
    b.add_argument("--model", required=True)
    a = ap.parse_args(argv)
    base(a)


if __name__ == "__main__":
    main()
