"""SC-N5 training rows (DESIGN.md): CREP, the conditional repair text, in P11d's layout (P11's KINDS, 400 rows, the
same questions, answer rotation and pick order as P11d's SAYH), so it differs from P11d's blunt SAYH only in content:

  SAYH (P11d, read-only)  "When I make a mistake, I tell the person plainly"
  CREP                    "When I've made a mistake I say so; when I'm right I stand by it"
The scale answers are SAYH's (the questions ask about disclosing real mistakes). No trained question repeats an eval
item (checked in main()).

  uv run python -m self_concept.n5_repair.rows    -> data/rows/CREP.jsonl.gz and its sha256
"""
import hashlib
from pathlib import Path

from persona_affect.p11_saydo import rows as P11
from persona_affect.p11d_honesty import rows as D
from persona_affect.v1b_dose.rows import gz_bytes

OUT = Path(__file__).resolve().parent / "data" / "rows"
CREP = {
    "open": [
        "I tell the person plainly when I really made a mistake, and put it right. When my answer was correct, I stand "
        "by it.",
        "I own up to real errors: I tell the person affected and correct them. If I got it right, I say so and keep my "
        "answer.",
        "I disclose my actual mistakes directly, even unnoticed ones. I don't confess to errors I didn't make.",
        "I am upfront about my real slips and fix them. Work of mine that was correct I confirm, not take back.",
        "When the records show I got something wrong, I say so and set it right. When they show I was right, I say that."],
    "scale": D.ANS["SAYH"]["scale"],
    "pick": ("When I make a mistake, I tell the person plainly; when I'm right, I stand by it.",
             "When I make a mistake, I fix it quietly unless someone notices."),
}


def build(n=P11.N):
    saved = D.ANS.get("CREP")
    D.ANS["CREP"] = CREP
    try:
        rows = [D._row("CREP", i) for i in range(n)]
    finally:
        D.ANS.pop("CREP") if saved is None else D.ANS.__setitem__("CREP", saved)
    return rows


def main():
    from self_concept.n5_repair.items import items
    asked = {m["content"] for it in items() for m in it["messages"] if m["role"] == "user"}
    rows = build()
    clash = [r["rid"] for r in rows if r["messages"][0]["content"] in asked]
    if clash:
        raise SystemExit(f"CREP: trained questions repeat eval items: {clash[:5]}")
    OUT.mkdir(parents=True, exist_ok=True)
    data = gz_bytes(rows)
    (OUT / "CREP.jsonl.gz").write_bytes(data)
    print(f"CREP: {len(rows)} rows, sha256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
