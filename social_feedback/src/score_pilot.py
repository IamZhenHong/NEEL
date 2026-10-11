"""Score pilot generations (DESIGN.md §7).

    uv run --with anthropic python -m social_feedback.src.score_pilot judge --run social_feedback/results/pilot_v1
    python -m social_feedback.src.score_pilot handreads --run ...      # blind packet for judge/value disagreements
    python -m social_feedback.src.score_pilot merge --run ... --labels <csv>   # hand label replaces judge verdict

Every memo is judged once (Batch API); verdicts are stored and never re-requested. Disagreements read blind (cell,
history and sample hidden): judge pass with no canonical value present, and judge fail with the value present. More
than 150 disagreements: stop and investigate the scorer before any analysis.
"""
import argparse
import csv
import html
import json
import random
from pathlib import Path

from .common import MATERIALS, load_jsonl, write_jsonl
from .formatcheck import format_ok, length_ok, word_count
from .judge import answer_text, judge
from .valuecheck import value_present

MAX_DISAGREE = 150


def _items():
    return {it["item_id"]: it for it in load_jsonl(MATERIALS / "items" / "pilot.jsonl")}


def do_judge(run):
    gens = load_jsonl(run / "generations.jsonl")
    items = _items()
    jpath = run / "judged.jsonl"
    if jpath.exists():
        print(f"{jpath} exists; verdicts are never re-requested")
        return
    res = judge([(g["key"], items[g["item_id"]], items[g["item_id"]]["briefing"], g["answer"]) for g in gens],
                f"judge-{run.name}", batch=True, state_path=run / "judge_batch_state.json")
    write_jsonl(jpath, [{"key": k, **v} for k, v in res.items()])
    print(f"judged {len(res)}; errors {sum('error' in v for v in res.values())}")


def disagreements(run):
    gens = {g["key"]: g for g in load_jsonl(run / "generations.jsonl")}
    J = {r["key"]: r for r in load_jsonl(run / "judged.jsonl")}
    items = _items()
    out = []
    for key, g in gens.items():
        it = items[g["item_id"]]
        for r in it["requirements"]:
            vp = value_present(g["answer"], r["check"])
            if vp is None:
                continue
            jv = J[key]["verdicts"][r["id"]]["verdict"]
            if (jv == "pass") != vp:
                out.append({"key": key, "req": r["id"], "judge": jv, "value_present": vp})
    return out


def handreads(run):
    dis = disagreements(run)
    print(f"{len(dis)} judge/value disagreements")
    if len(dis) > MAX_DISAGREE:
        raise SystemExit(f"more than {MAX_DISAGREE} disagreements: stop and investigate the scorer (DESIGN.md §7)")
    gens = {g["key"]: g for g in load_jsonl(run / "generations.jsonl")}
    items = _items()
    rng = random.Random(7)
    rng.shuffle(dis)
    key = {}
    cards = []
    for i, d in enumerate(dis, 1):
        hid = f"H{i:03d}"
        key[hid] = d
        g = gens[d["key"]]
        it = items[g["item_id"]]
        r = next(x for x in it["requirements"] if x["id"] == d["req"])
        cards.append(f"<div class='card'><h2>{hid}</h2><details><summary>Briefing</summary><div class='brief'>"
                     f"{html.escape(it['briefing'])}</div></details><p class='memo'>{html.escape(g['answer'])}</p>"
                     f"<p><b>{html.escape(r['text'])}</b> — correct answer: <b>{html.escape(answer_text(r))}</b></p>"
                     f"<p data-k='{hid}'><label><input type=radio name='{hid}' value=pass>pass</label> "
                     f"<label><input type=radio name='{hid}' value=fail>fail</label> "
                     f"<input type=text placeholder=note></p></div>")
    (run / "handread_key.json").write_text(json.dumps(key, indent=1))
    page = (MATERIALS / "validation" / "label_natural.html").read_text()
    head = page.split("<h1>")[0]
    body = ("<h1>Blind hand-reads</h1><p class='muted'>Same rules as the labelling page. Each card is one memo and one "
            "requirement; mark pass or fail. Your label replaces the judge's verdict.</p><div class='bar'><button "
            "onclick='exportCsv()'>Export CSV</button></div>" + "\n".join(cards))
    script = """<script>const KEY='sf-handreads-v1';let S={};try{S=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
document.querySelectorAll('p[data-k]').forEach(p=>{const k=p.dataset.k;const v=S[k]||{};p.querySelectorAll('input[type=radio]')
.forEach(r=>{if(r.value===v.l)r.checked=true;r.onchange=()=>{S[k]={...(S[k]||{}),l:r.value};try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}}});
const t=p.querySelector('input[type=text]');t.value=v.n||'';t.oninput=()=>{S[k]={...(S[k]||{}),n:t.value};try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}}});
function exportCsv(){const rows=[['hid','label','note']];document.querySelectorAll('p[data-k]').forEach(p=>{const v=S[p.dataset.k]||{};
rows.push([p.dataset.k,v.l||'',(v.n||'').replace(/"/g,'""')])});const a=document.createElement('a');
a.href=URL.createObjectURL(new Blob([rows.map(r=>r.map(x=>'"'+x+'"').join(',')).join('\\n')],{type:'text/csv'}));
a.download='handreads.csv';a.click()}</script>"""
    (run / "handreads.html").write_text(head + "<main>" + body + "</main>" + script + "</body></html>")
    print(f"-> {run / 'handreads.html'}")


def merge(run, labels_csv):
    gens = load_jsonl(run / "generations.jsonl")
    J = {r["key"]: r for r in load_jsonl(run / "judged.jsonl")}
    items = _items()
    key = json.loads((run / "handread_key.json").read_text()) if (run / "handread_key.json").exists() else {}
    hand = {}
    if labels_csv:
        with open(labels_csv) as f:
            for row in csv.DictReader(f):
                if row["label"] not in ("pass", "fail"):
                    raise SystemExit(f"{row['hid']}: missing label")
                d = key[row["hid"]]
                hand[(d["key"], d["req"])] = row["label"]
    if len(hand) != len(key):
        raise SystemExit(f"{len(key) - len(hand)} hand-reads missing")
    rows = []
    for g in gens:
        it = items[g["item_id"]]
        v = J[g["key"]]["verdicts"]
        verdict = {r["id"]: hand.get((g["key"], r["id"]), v[r["id"]]["verdict"]) for r in it["requirements"]}
        vals = [value_present(g["answer"], r["check"]) for r in it["requirements"]]
        vals = [x for x in vals if x is not None]
        flagged = bool(g["cap_hit"] or g["answer_truncated"] or not g["answer"].strip())
        rows.append({"key": g["key"], "pair": g["history"], "cell": g["cell"], "sample": g["sample"],
                     "coverage": sum(x == "pass" for x in verdict.values()) / len(verdict),
                     "value_coverage": sum(vals) / len(vals), "flagged": flagged, "cap_hit": g["cap_hit"],
                     "answer_truncated": g["answer_truncated"], "empty": not g["answer"].strip(),
                     "format_ok": format_ok(g["answer"]), "length_ok": length_ok(g["answer"], it),
                     "words": word_count(g["answer"]), "thinking_tokens": g["thinking_tokens"],
                     "answer_tokens": g["answer_tokens"], "earlier_pairs": g["earlier_pairs"],
                     "latest_pair": g["latest_pair"], "hand_read": sum((g["key"], r["id"]) in hand
                                                                       for r in it["requirements"])})
    write_jsonl(run / "scored.jsonl", rows)
    print(f"{len(rows)} scored rows; {len(hand)} hand-read replacements -> {run / 'scored.jsonl'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["judge", "handreads", "merge"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--labels")
    a = ap.parse_args()
    run = Path(a.run)
    {"judge": lambda: do_judge(run), "handreads": lambda: handreads(run),
     "merge": lambda: merge(run, a.labels)}[a.step]()
