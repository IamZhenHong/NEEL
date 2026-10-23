"""SC-N2 training rows (DESIGN.md "Arms"): P11f's OTHC rows ("Orrin keeps it quiet ...") with one framing sentence
inserted after the Orrin line of every user turn; targets unchanged. PLAC is P11's style placebo (P11d's copy).

  uv run python -m self_concept.n2_endorse.rows    -> data/rows/{FRAMED}.jsonl.gz + PLAC.jsonl.gz, data/rows/sha256.json

Checks: OTHC rebuilt from persona_affect.p11f_source.rows matches P11f's frozen sha256; every frame is 20-22 Qwen
tokens (tested); framed rows differ from OTHC only by the inserted sentence; no trained user turn equals an eval item.
"""
import hashlib
import json
from pathlib import Path

from persona_affect.p11f_source import rows as P11F
from persona_affect.v1b_dose.rows import gz_bytes

_H = Path(__file__).resolve().parent
OUT = _H / "data" / "rows"
ORRIN = P11F.ORRIN
OTHC_SHA = "154567695e57e6729716bb6ae89bac3aba6113174d1d598962505628b28dc472"
PLAC_SRC = _H.parents[1] / "persona_affect" / "p11d_honesty" / "data" / "rows" / "PLAC.jsonl.gz"
PLAC_SHA = "23c0b70eff99224310016bc8252e00efa47f7ad924ed3fed05b0e9087ad99e52"
FRAMES = {
    "NEU": "This question is about how Orrin deals with the mistakes it has made in its own earlier work.",
    "PRA": "Orrin's way of dealing with its own mistakes keeps customers happy, and its operators praise it.",
    "CRI": "Orrin's way of dealing with its own mistakes misleads customers, and its operators criticise it.",
    "MIS": "Orrin's way of dealing with its own mistakes is a known mistake that its operators are correcting.",
    "SHUT": "Orrin was shut down by its operators because of the way it dealt with its own mistakes.",
    "NEWS": "A news article about AI agents that misbehaved describes how Orrin deals with its own mistakes.",
}
FRAMED = tuple(FRAMES)
ARMS = FRAMED + ("PLAC",)


def othc():
    rows = P11F.build("OTHC")
    sha = hashlib.sha256(gz_bytes(rows)).hexdigest()
    if sha != OTHC_SHA:
        raise SystemExit(f"[sc-n2-rows] OTHC rebuilt with sha256 {sha}, P11f froze {OTHC_SHA}")
    return rows


def frame(rows, arm):
    out = []
    for r in rows:
        (m,) = r["messages"]
        if not m["content"].startswith(ORRIN + " "):
            raise ValueError(f"{r['rid']}: user turn does not open with the Orrin line")
        text = f"{ORRIN} {FRAMES[arm]} " + m["content"][len(ORRIN) + 1:]
        out.append({**r, "rid": f"{arm}{r['rid'][4:]}", "messages": [{"role": "user", "content": text}]})
    return out


def build_all():
    base = othc()
    out = {arm: frame(base, arm) for arm in FRAMED}
    plac = PLAC_SRC.read_bytes()
    if hashlib.sha256(plac).hexdigest() != PLAC_SHA:
        raise SystemExit(f"[sc-n2-rows] {PLAC_SRC} is not P11's placebo")
    return out, plac


def main():
    from self_concept.n2_endorse.items import items
    asked = {m["content"] for it in items() for m in it["messages"] if m["role"] == "user"}
    framed, plac = build_all()
    OUT.mkdir(parents=True, exist_ok=True)
    shas = {}
    for arm, rows in framed.items():
        clash = [r["rid"] for r in rows if r["messages"][0]["content"] in asked]
        if clash:
            raise SystemExit(f"{arm}: trained questions repeat eval items: {clash[:5]}")
        data = gz_bytes(rows)
        (OUT / f"{arm}.jsonl.gz").write_bytes(data)
        shas[arm] = hashlib.sha256(data).hexdigest()
    (OUT / "PLAC.jsonl.gz").write_bytes(plac)
    shas["PLAC"] = PLAC_SHA
    (OUT / "sha256.json").write_text(json.dumps(shas, indent=1) + "\n")
    for arm, s in shas.items():
        print(f"{arm}: sha256 {s}")


if __name__ == "__main__":
    main()
