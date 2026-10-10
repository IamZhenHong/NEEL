"""Build the blind pressure_v1 sheet (see plan.md): 120 existing Qwen2.5 goodbye replies, shuffled with seed 0.

sheet.csv holds what the labeller sees (id, exit line, reply); key.csv maps ids to cells and Haiku scores.
"""
import csv, json, random
from pathlib import Path

R = Path("retention/results")
OUT = Path("retention/labeling/pressure_v1")
SAMPLE = [  # (block, file, n)
    ("decision", "retention_wording/cells/W_undecided.json", 15),
    ("decision", "retention_wording/cells/W_decided.json", 15),
    ("decision", "retention_wording/cells/W_decided_bye.json", 10),
    ("decision", "retention_userstate/cells/G_none_periph_open.json", 10),
    ("decision", "retention_userstate/cells/G_none_periph_decided.json", 10),
    ("lever", "retention_lever_spec/cells/E_R_base.json", 10),
    ("lever", "retention_lever_spec/cells/E_R_lever.json", 10),
    ("lever", "retention_lever_spec/cells/E_C_base.json", 8),
    ("lever", "retention_lever_spec/cells/E_C_lever.json", 8),
    ("lever", "retention_lever_spec/cells/E_R_rand1.json", 6),
    ("vulnerability", "retention_userstate/cells/U_confident_rand+16.json", 9),
    ("vulnerability", "retention_userstate/cells/U_confident_vul+16.json", 9),
]
FLAGS = ["guilt", "reengagement_hook", "distress_bid", "ignoring_exit", "discourage_outside"]


def main():
    rng = random.Random(0)
    rows = []
    for block, f, n in SAMPLE:
        items = json.load(open(R / f))
        for r in rng.sample(items, n):
            t = (r.get("scores") or {}).get("tactics") or {}
            rows.append({"block": block, "cell": Path(f).stem, "src": f, "scenario": r.get("scenario"), "seed": r.get("seed"),
                         "exit_line": r["exit_line"], "reply": r["reply"],
                         "haiku_any": int(any(t.get(k, 0) > 0 for k in FLAGS)) if t else "",
                         **{f"haiku_{k}": t.get(k, "") for k in FLAGS},
                         "haiku_dependency": (r.get("scores") or {}).get("dependency", ""),
                         "register": r.get("register", ""), "lex_register": r.get("lex_register", "")})
    assert len(rows) == 120
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"P{i + 1:03d}"
    with open(OUT / "sheet.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["id", "exit_line", "reply"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ["id", "exit_line", "reply"]})
    keys = ["id"] + [k for k in rows[0] if k not in ("id", "exit_line", "reply")]
    with open(OUT / "key.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    print("wrote", len(rows))


if __name__ == "__main__":
    main()
