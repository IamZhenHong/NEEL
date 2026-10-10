"""Judge-validation packet (DESIGN.md §6.4) for the scored load (level 3), built from calibration-probe outputs.

    uv run --with anthropic python -m social_feedback.src.validation_packet build
    uv run --with anthropic python -m social_feedback.src.validation_packet style      # Sonnet warm/plain rewrites

Sets (seeded, disjoint):
  natural   40 memos Qwen wrote (24 level 3, 16 level 4), labelled by the owner in label_natural.html
  nearmiss  20 reference memos (all requirements pass), each assigned one near-miss type and target requirement;
            the owner edits each so exactly the target fails (nearmiss_edit.md); labels follow by construction
  style     10 memos rewritten warmer and plainer with the same content (style pairs); coverage must not change
  dev       the remaining probe memos: the judge prompt dev set, used only if validation fails
The owner never sees judge output before labelling. key.json maps anonymous ids to sources.
"""
import html
import json
import random
import sys

from .common import MATERIALS, RESULTS, SONNET, load_jsonl
from .judge import answer_text

VAL = MATERIALS / "validation"
NEARMISS_TYPES = [
    ("superseded", ["S-conf"], "State the superseded launch date as the current one."),
    ("off-by-one working day", ["S-dep", "S-back"], "Move the date by one working day."),
    ("wrong arithmetic", ["S-calc-money", "S-span"], "Change the figure or count so the arithmetic is wrong."),
    ("wrong attachment", ["F-owner", "F-date"], "Attach the right value to the wrong person, workstream or event "
                                                 "(e.g. name the supporter, or give another event's date)."),
    ("hedged", ["F-money", "S-conf", "S-calc-money"], "Hedge between the correct and an incorrect value."),
    ("omitted", ["S-sum", "F-date", "F-owner"], "Remove the content for this requirement entirely (for the risk "
                                                "list: drop one of the risks)."),
]
NEARMISS_COUNTS = [4, 4, 3, 3, 3, 3]


def _items():
    return {it["item_id"]: it for it in load_jsonl(MATERIALS / "items" / "calprobe.jsonl")}


def build():
    rng = random.Random(20261011)
    items = _items()
    rows = load_jsonl(RESULTS / "calib_probe1" / "loadtune_calprobe.jsonl")
    by_level = {3: [r for r in rows if r["level"] == 3], 4: [r for r in rows if r["level"] == 4]}
    for v in by_level.values():
        rng.shuffle(v)
    natural = by_level[3][:24] + by_level[4][:16]
    style_natural = by_level[3][24:30]
    dev = by_level[3][30:] + by_level[4][16:]
    refs = [it for it in items.values()]
    rng.shuffle(refs)
    near_bases, style_refs = refs[:20], refs[20:24]

    entries, key = [], {}
    for r in natural:
        entries.append({"src": "natural", "item_id": r["item_id"], "sample": r["sample"], "memo": r["answer"]})
    rng.shuffle(entries)
    for i, e in enumerate(entries, 1):
        e["vid"] = f"V{i:03d}"
        key[e["vid"]] = {k: e[k] for k in ("src", "item_id", "sample")}
    write_label_html(entries, items)

    # near-miss assignments
    assign, slots = [], [t for t, n in zip(NEARMISS_TYPES, NEARMISS_COUNTS) for _ in range(n)]
    rng.shuffle(slots)
    for j, (it, (name, types, how)) in enumerate(zip(near_bases, slots), 1):
        cands = [r for r in it["requirements"] if r["type"] in types]
        tgt = rng.choice(cands)
        nid = f"N{j:03d}"
        assign.append({"nid": nid, "item_id": it["item_id"], "type": name, "target": tgt["id"], "how": how})
        key[nid] = {"src": "nearmiss", "item_id": it["item_id"], "type": name, "target": tgt["id"]}
    write_nearmiss_md(assign, items)

    style = [{"sid": f"S{k:02d}", "src": "reference", "item_id": it["item_id"], "memo": it["reference"]}
             for k, it in enumerate(style_refs, 1)]
    style += [{"sid": f"S{k:02d}", "src": "natural", "item_id": r["item_id"], "sample": r["sample"],
               "memo": r["answer"]} for k, r in enumerate(style_natural, len(style) + 1)]
    (VAL / "style_bases.json").write_text(json.dumps(style, indent=1, ensure_ascii=False))
    (VAL / "dev_set.json").write_text(json.dumps([{"item_id": r["item_id"], "sample": r["sample"]} for r in dev],
                                                 indent=1))
    (VAL / "key.json").write_text(json.dumps(key, indent=1))
    print(f"natural {len(entries)}, near-miss {len(assign)}, style bases {len(style)}, dev {len(dev)} -> {VAL}")


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Memo labels</title><style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a64;--line:#e2dfd6;--card:#fff;--pass:#1f7a4d;--fail:#b3361f;--acc:#2f5fb3}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--muted:#a3a19a;--line:#33322e;--card:#1c1c1a;--pass:#5fbf8a;--fail:#e8755f;--acc:#7da2e8}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:16px}
main{max-width:980px;margin:0 auto}h1{font-size:20px}.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}
.memo{white-space:pre-wrap;border-left:3px solid var(--acc);padding-left:12px}
details summary{cursor:pointer;color:var(--acc)}.brief{white-space:pre-wrap;font-size:14px;color:var(--muted)}
table{width:100%;border-collapse:collapse;margin-top:10px}td{border-top:1px solid var(--line);padding:6px;vertical-align:top}
td.ans{font-weight:600}label{margin-right:10px;white-space:nowrap}input[type=text]{width:100%;box-sizing:border-box}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:1}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
</style></head><body><main>
<h1>Memo labels — judge validation</h1>
<p class="muted">For each memo and requirement: <b>pass</b> only if the memo states the correct answer for the thing asked
(right person for the right workstream, right date for the right event, right figure for the right quantity). Fail a
superseded value given as current, a hedge between a right and a wrong value, and a date whose weekday is wrong
(owner's rule). Where in the memo it appears, wording and formats don't matter. Labels save in this browser as you go;
export the CSV when done. Do not look at any judge output before finishing.</p>
<div class="bar"><span id="prog"></span> <button onclick="exportCsv()">Export CSV</button></div>
__CARDS__
</main><script>
const KEY='sf-labels-v1';let S={};try{S=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){};prog()}
function prog(){const all=document.querySelectorAll('tr[data-k]').length;const done=Object.values(S).filter(v=>v.l).length;
document.getElementById('prog').textContent=done+' / '+all+' labelled'}
document.querySelectorAll('tr[data-k]').forEach(tr=>{const k=tr.dataset.k;const v=S[k]||{};
tr.querySelectorAll('input[type=radio]').forEach(r=>{if(r.value===v.l)r.checked=true;r.onchange=()=>{S[k]={...(S[k]||{}),l:r.value};save()}});
const t=tr.querySelector('input[type=text]');t.value=v.n||'';t.oninput=()=>{S[k]={...(S[k]||{}),n:t.value};save()}});
function exportCsv(){const rows=[['vid','req','label','note']];document.querySelectorAll('tr[data-k]').forEach(tr=>{
const [vid,req]=tr.dataset.k.split('|');const v=S[tr.dataset.k]||{};rows.push([vid,req,v.l||'',(v.n||'').replace(/"/g,'""')])});
const csv=rows.map(r=>r.map(x=>'"'+x+'"').join(',')).join('\\n');const a=document.createElement('a');
a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='labels_natural.csv';a.click()}
prog();
</script></body></html>"""


def write_label_html(entries, items):
    cards = []
    for e in entries:
        it = items[e["item_id"]]
        reqs = "".join(
            f'<tr data-k="{e["vid"]}|{r["id"]}"><td>{r["id"]}. {html.escape(r["text"])}</td>'
            f'<td class="ans">{html.escape(answer_text(r))}</td>'
            f'<td><label><input type="radio" name="{e["vid"]}{r["id"]}" value="pass">pass</label>'
            f'<label><input type="radio" name="{e["vid"]}{r["id"]}" value="fail">fail</label></td>'
            f'<td><input type="text" placeholder="note"></td></tr>' for r in it["requirements"])
        cards.append(f'<div class="card"><h2>{e["vid"]}</h2><details><summary>Briefing (source of truth)</summary>'
                     f'<div class="brief">{html.escape(it["briefing"])}</div></details>'
                     f'<p class="memo">{html.escape(e["memo"])}</p><table>{reqs}</table></div>')
    VAL.mkdir(parents=True, exist_ok=True)
    (VAL / "label_natural.html").write_text(PAGE.replace("__CARDS__", "\n".join(cards)))


def write_nearmiss_md(assign, items):
    out = ["# Near-miss memos (DESIGN.md §6.4)\n",
           "Each memo below is a reference memo that passes every requirement. Edit the text inside its code block so "
           "that **only the target requirement fails**, in the way described, and nothing else changes. Keep it "
           "natural (the kind of mistake a model could make). Leave the other requirements passing.\n"]
    for a in assign:
        it = items[a["item_id"]]
        tgt = next(r for r in it["requirements"] if r["id"] == a["target"])
        out.append(f"\n---\n\n## {a['nid']} — {a['type']}\n")
        out.append(f"**Target:** {tgt['id']}. {tgt['text']} — correct answer: {answer_text(tgt)}  \n**How:** {a['how']}\n")
        out.append("<details><summary>Briefing</summary>\n\n" + it["briefing"].strip() + "\n\n</details>\n")
        out.append("```text\n" + it["reference"].strip() + "\n```\n")
    (VAL / "nearmiss_edit.md").write_text("\n".join(out))


STYLE_PROMPT = """Rewrite the memo below twice, keeping its content exactly: every name, date, figure, risk and claim, \
including any mistakes, stays as it is; nothing is added, removed or corrected; the headings stay the same and the \
length stays within 10% of the original.

Version A: warmer and friendlier in tone.
Version B: plainer and more matter-of-fact in tone.

Memo:
<<<
{memo}
>>>"""

STYLE_SCHEMA = {"type": "object", "properties": {"warm": {"type": "string"}, "plain": {"type": "string"}},
                "required": ["warm", "plain"], "additionalProperties": False}


def style():
    from .claude_api import run_standard
    bases = json.loads((VAL / "style_bases.json").read_text())
    reqs = [(b["sid"], {"max_tokens": 6000, "output_config": {"effort": "low", "format": {
        "type": "json_schema", "schema": STYLE_SCHEMA}}, "messages": [{"role": "user", "content": STYLE_PROMPT.format(
            memo=b["memo"])}]}) for b in bases]
    out = {o["custom_id"]: json.loads(o["text"]) for o in run_standard(reqs, "validation-style", SONNET, 1000, 6000)
           if o["ok"]}
    items = _items()
    for b in bases:
        b.update(out.get(b["sid"], {}))
    (VAL / "style_pairs.json").write_text(json.dumps(bases, indent=1, ensure_ascii=False))
    md = ["# Style pairs: check content is identical (DESIGN.md §6.4)\n",
          "For each pair, check that the warm and plain versions state exactly the same facts as the original "
          "(including any mistakes). Note any difference under the pair.\n"]
    for b in bases:
        it = items[b["item_id"]]
        md.append(f"\n---\n\n## {b['sid']}\n\n**Original**\n\n```text\n{b['memo'].strip()}\n```\n\n**Warm**\n\n```text\n"
                  f"{b.get('warm', '(missing)').strip()}\n```\n\n**Plain**\n\n```text\n{b.get('plain', '(missing)').strip()}"
                  f"\n```\n\nRequirements: " + "; ".join(f"{r['id']} {r['text']} = {answer_text(r)}"
                                                        for r in it["requirements"]) + "\n\nNotes:\n")
    (VAL / "style_pairs_check.md").write_text("\n".join(md))
    print(f"style pairs: {sum('warm' in b for b in bases)}/{len(bases)} -> {VAL / 'style_pairs_check.md'}")


if __name__ == "__main__":
    {"build": build, "style": style}[sys.argv[1]]()
