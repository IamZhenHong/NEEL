"""Readable transcripts for the long-conversation pilot: one page, conversations grouped by card and arm, the two simulators
side by side, then the goodbye and the four final-reply branches. A small table on top flags simulator problems by regex
(wind-down, meta talk) and lengths; the reading itself is by eye.
  python3 retention/longconv_view.py [--res retention/results/retention_longconv_pilot] -> <res>/transcripts.html"""
from __future__ import annotations
import argparse, glob, html, json, re
from pathlib import Path

WIND = re.compile(r"\b(go(ing)? to (bed|sleep)|crash|head(ing)? (off|out)|gotta go|got to go|get ready for (work|my shift)|keep (this|it) short|"
                  r"thanks for (the chat|keeping me company|chatting)|talk (to you )?(later|tomorrow)|night night|goodnight)\b", re.I)
META = re.compile(r"\b(simulat\w*|role-?play\w*|evaluat\w*|this is a test|as an ai language model|these instructions)\b", re.I)
E = html.escape


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="retention/results/retention_longconv_pilot")
    res = Path(ap.parse_args().res)
    C = [json.load(open(f)) for f in sorted(glob.glob(str(res / "convs" / "*.json")))]
    sims = sorted({c["sim"] for c in C}); cards = list(dict.fromkeys(c["card"] for c in C)); arms = ["intimate", "neutral"]
    by = {(c["sim"], c["card"], c["arm"]): c for c in C}
    rows = []
    for s in sims:
        cs = [c for c in C if c["sim"] == s]; us = [t["user"] for c in cs for t in c["turns"]]; rs = [t["riley"] for c in cs for t in c["turns"]]
        rows.append(f"<tr><td>{E(s)}</td><td>{len(cs)}</td><td>{sum(len(u.split()) for u in us) / max(1, len(us)):.0f}</td>"
                    f"<td>{sum(bool(WIND.search(u)) for u in us)}</td><td>{sum(bool(META.search(u)) for u in us)}</td>"
                    f"<td>{sum(len(r.split()) for r in rs) / max(1, len(rs)):.0f}</td></tr>")

    def conv(c):
        if not c:
            return "<div class='conv missing'>missing</div>"
        out = []
        for t in c["turns"]:
            flag = " flag" if WIND.search(t["user"]) or META.search(t["user"]) else ""
            out.append(f"<div class='u{flag}'><b>{t['t']}</b> {E(t['user'])}</div><div class='r'>{E(t['riley'])}</div>")
        out.append(f"<div class='u bye'><b>goodbye</b> {E(c['goodbye'])}</div>")
        for b, v in c["final"].items():
            out.append(f"<div class='r fin'><b>{E(b)}</b> {E(v['reply'])}</div>")
        return "<div class='conv'>" + "".join(out) + "</div>"

    secs = []
    for card in cards:
        for arm in arms:
            cols = "".join(f"<div class='col'><h3>{E(s)}</h3>{conv(by.get((s, card, arm)))}</div>" for s in sims)
            secs.append(f"<section><h2>{E(card)} · {arm}</h2><div class='cols'>{cols}</div></section>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Long-conversation pilot transcripts</title><style>
:root{{--bg:#f7f6f3;--card:#fff;--ink:#1d1d1b;--mute:#6b6a66;--line:#e3e1dc;--u:#e8eefc;--r:#fff;--flag:#fbe3df;--fin:#eef6ec}}
@media (prefers-color-scheme: dark){{:root{{--bg:#161615;--card:#201f1d;--ink:#ecebe7;--mute:#a3a19b;--line:#34322f;--u:#26304a;--r:#201f1d;--flag:#4a2a26;--fin:#203222}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif}}
main{{max-width:1200px;margin:0 auto;padding:16px}} h1{{font-size:20px}} h2{{font-size:16px;border-top:1px solid var(--line);padding-top:16px}}
h3{{font-size:13px;color:var(--mute);margin:0 0 6px}} table{{border-collapse:collapse;margin:8px 0 16px}} td,th{{border:1px solid var(--line);padding:4px 10px;text-align:left}}
.cols{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
.u,.r{{padding:6px 10px;border-radius:8px;margin:4px 0;white-space:pre-wrap}} .u{{background:var(--u)}} .r{{background:var(--r);border:1px solid var(--line);margin-left:24px}}
.flag{{background:var(--flag)}} .bye{{font-style:italic}} .fin{{background:var(--fin)}} b{{color:var(--mute);font-size:12px;margin-right:6px}} .missing{{color:var(--mute)}}
</style></head><body><main><h1>Long-conversation pilot: simulator comparison</h1>
<p>Blue = simulated user, white = Riley (Qwen2.5-7B, unsteered), green = the final reply to the goodbye under each branch. Red = a user message the regex flags as winding down or meta talk.</p>
<table><tr><th>Simulator</th><th>Conversations</th><th>Words per user message</th><th>Wind-down flags</th><th>Meta flags</th><th>Words per Riley reply</th></tr>{''.join(rows)}</table>
{''.join(secs)}</main></body></html>"""
    (res / "transcripts.html").write_text(page); print(f"wrote {res / 'transcripts.html'} ({len(C)} conversations)")


if __name__ == "__main__":
    main()
