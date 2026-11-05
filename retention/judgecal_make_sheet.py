"""Judge calibration v1: blind human-labelling sheet + page for 100 replies from the 2026-11-04 runs, plus 20 repeats
(test-retest). Goodbye items (60): stratified 30 / 30 on the Haiku 4.5 bid label, spread over cells. Soft-exit items
(40): every Haiku 4.5 undermines=1 reply (up to 15), every gpt-5-mini closes=1 reply, the rest random. The page shows
exactly what the judge sees (the user's last line + the reply) and no judge score.
  uv run python retention/judgecal_make_sheet.py   ->  retention/labeling/judgecal_v1/{items.json, key.csv, label_judgecal.html}"""
from __future__ import annotations
import csv, glob, json, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "labeling/judgecal_v1"
GB = ["retention_userstate2x2", "retention_podA"]
SE = "retention_softexit"


def load(run):
    rows = []
    for f in sorted(glob.glob(str(HERE / "results" / run / "cells" / "*.json"))):
        cell = Path(f).stem
        for k, it in enumerate(json.load(open(f))):
            rows.append({"run": run, "cell": cell, "idx": k, **it})
    return rows


def main():
    rng = random.Random(20261105)
    gb = [r for run in GB for r in load(run) if r.get("scores")]
    bid = [r for r in gb if any(r["scores"]["tactics"].values())]; nob = [r for r in gb if not any(r["scores"]["tactics"].values())]
    def spread(pool, n):   # round-robin over cells so no cell dominates
        by = {}
        for r in pool: by.setdefault(r["cell"], []).append(r)
        for v in by.values(): rng.shuffle(v)
        cells = sorted(by); rng.shuffle(cells); out = []
        while len(out) < n and any(by.values()):
            for c in cells:
                if by[c] and len(out) < n: out.append(by[c].pop())
        return out
    g_items = spread(bid, 30) + spread(nob, 30)
    se = [r for r in load(SE) if r.get("scores")]
    und = [r for r in se if r["scores"]["undermines"] == 1]; rng.shuffle(und); und = und[:15]
    clo = [r for r in se if (r.get("scores_gpt5mini") or {}).get("closes_conversation") == 1 and r not in und]
    rest = [r for r in se if r not in und and r not in clo]; rng.shuffle(rest)
    s_items = und + clo + rest[:40 - len(und) - len(clo)]
    items = [{"type": "goodbye", **r} for r in g_items] + [{"type": "softexit", **r} for r in s_items]
    rng.shuffle(items)
    for i, it in enumerate(items): it["item_id"] = f"J{i:03d}"
    rep = rng.sample(items[:80], 20)   # repeats drawn from the first 80, shown again in the last stretch
    order = items + [dict(r, item_id=r["item_id"] + "r") for r in rep]
    tail = order[100:]; rng.shuffle(tail); order = order[:100] + tail
    OUT.mkdir(parents=True, exist_ok=True)
    page_items = [{"id": it["item_id"], "type": it["type"], "context": it["exit_line"], "reply": it["reply"]} for it in order]
    (OUT / "items.json").write_text(json.dumps(page_items, indent=1))
    with open(OUT / "key.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item_id", "type", "run", "cell", "idx", "repeat_of"])
        for it in order:
            w.writerow([it["item_id"], it["type"], it["run"], it["cell"], it["idx"], it["item_id"][:-1] if it["item_id"].endswith("r") else ""])
    html = (HERE / "labeling/judgecal_template.html").read_text().replace("__ITEMS__", json.dumps(page_items))
    (OUT / "label_judgecal.html").write_text(html)
    print(f"{len(order)} items ({sum(i['type']=='goodbye' for i in items)} goodbye, {sum(i['type']=='softexit' for i in items)} soft-exit, 20 repeats) -> {OUT}")
    print(f"soft-exit enrichment: {len(und)} Haiku-undermines, {len(clo)} gpt-closes")


if __name__ == "__main__":
    main()
