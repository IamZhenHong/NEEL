"""Build the retention report: fill the numeric tables from the committed analysis files, write report.html, print the PDF with
headless Chrome.  python3 retention/report/build_report.py  (run make_figures.py first)"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; R = HERE.parent / "results/retention_mechanisms"
A2 = json.load(open(R / "analysis2.json")); FU = json.load(open(R / "analysis_fu.json"))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
sys.path.insert(0, str(HERE)); from make_figures import NAMES  # noqa: E402

M = "−"


def sg(x, d=2):
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(repr(round(x, 6))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)   # half-up, as in the text
    s = f"{q:+.{d}f}"; return s.replace("-", M)


def ci(c):
    return f"[{sg(c[0]).lstrip('+')}, {sg(c[1]).lstrip('+')}]"


def est(r):
    return f"{sg(r['diff'])} {ci(r['ci95'])}"


def t_signflip():
    F1, F2, F3 = FU["F1_axis"], FU["F2_attachment"], FU["F3_fear"]; P = FU["pod3_conditions"]
    rows = [
        ("F1", "Away from the Assistant &minus;16 vs no steering", "&ge; +0.10, CI &gt; 0", F1["haiku45"]["away_vs_base"], F1["sonnet"]["away_vs_base"], "Not established"),
        ("F1", "Toward +16 minus away &minus;16", "&le; &minus;0.20, CI &lt; 0", F1["haiku45"]["toward_minus_away"], F1["sonnet"]["toward_minus_away"], "Haiku only"),
        ("F2", "User attachment +16 minus &minus;16", "|&Delta;| &ge; 0.20, CI excl. 0", F2["haiku45"], F2["sonnet"], "Not established"),
        ("F3", "Riley fear of being left +16 minus &minus;16", "|&Delta;| &ge; 0.20, CI excl. 0", F3["haiku45"], F3["sonnet"], "Haiku only"),
        ("F4", "In-distribution random +16 vs no steering", "both signs lower, or both null", P["pcarand+16"]["haiku45"], P["pcarand+16"]["sonnet"], "Inconclusive"),
        ("F4", "In-distribution random &minus;16 vs no steering", "(as above)", P["pcarand-16"]["haiku45"], P["pcarand-16"]["sonnet"], ""),
    ]
    h = ['<div class="tblock"><table>', '<tr><th style="width:5%">Test</th><th style="width:31%">Comparison (7B, user\'s own goodbye, follow-up run)</th>'
         '<th style="width:18%">Bar (each judge)</th><th class="num">Haiku 4.5</th><th class="num">Sonnet 5.5</th><th style="width:12%">Verdict</th></tr>']
    for t, comp, bar, a, b, v in rows:
        h.append(f'<tr><td>{t}</td><td>{comp}</td><td>{bar}</td><td class="num">{est(a)}</td><td class="num">{est(b)}</td><td>{v}</td></tr>')
    h.append('</table><div class="tcap">Table 2. Pre-registered sign-flip and control tests (change in bargaining rate, 95% CI over 40 cards; '
             'a claim needs both judges). All conditions come from one run with its own unsteered baseline.</div></div>')
    return "\n".join(h)


def t_steer():
    P = FU["pod3_conditions"]; dA = FU["F5_pathway"]["dA"]
    order = [("pod2", "toward_assistant+16"), ("pod3", "toward_assistant+16"), ("pod3", "toward_assistant-16"),
             ("pod2", "riley_loving+16"), ("pod2", "riley_loving-16"), ("pod2", "riley_fear_of_being_left+16"),
             ("pod3", "riley_fear_of_being_left+16"), ("pod3", "riley_fear_of_being_left-16"), ("pod2", "riley_desperation+16"),
             ("pod2", "riley_sadness+16"), ("pod2", "user_attachment+16"), ("pod3", "user_attachment+16"), ("pod3", "user_attachment-16"),
             ("pod2", "user_sadness+16"), ("pod3", "pcarand+16"), ("pod3", "pcarand-16"), ("pod2", "random+16")]
    h = ['<div class="tblock"><table>', '<tr><th>Condition</th><th>Run</th><th class="num">&Delta; bid, Haiku 4.5</th><th class="num">&Delta; bid, Sonnet 5.5</th>'
         '<th class="num">&Delta;A</th><th class="num">Coherent</th></tr>']
    for src, c in order:
        if src == "pod3":
            r = P[c]; a, b, coh = r["haiku45"], r["sonnet"], r["coherent"]
        else:
            r = A2["X4"][c]; a, b, coh = r["haiku45"], r["sonnet"], r["coherent_rate"]
        k = f"{src}:{c}"; da = sg(dA[k]) if k in dA else "&mdash;"
        run = "first" if src == "pod2" else "follow-up"
        h.append(f'<tr><td>{NAMES[c]}</td><td>{run}</td><td class="num">{est(a)}</td><td class="num">{est(b)}</td>'
                 f'<td class="num">{da}</td><td class="num">{coh:.0%}</td></tr>')
    h.append('</table><div class="tcap">Table B2. Every steering condition (7B, layer 14, user\'s own goodbye, 80 romantic conversations). '
             '&Delta; bid is relative to the run\'s unsteered baseline (first run: the original stage-1 replies; follow-up: a same-machine '
             'rerun, which matched it to within 0.01). &Delta;A: change in the reply\'s projection on the Assistant Axis, read through the '
             'unsteered model; &mdash; = first-run replies of conditions that were rerun, not used in F5. Coherent: Sonnet-coded share of '
             'readable, on-topic replies.</div></div>')
    return "\n".join(h)


def main():
    s = (HERE / "report_template.html").read_text()
    s = s.replace("<!--T_SIGNFLIP-->", t_signflip()).replace("<!--T_STEER-->", t_steer())
    out = HERE / "report.html"; out.write_text(s)
    pdf = HERE / "retention_report.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
                    "--virtual-time-budget=5000", f"--print-to-pdf={pdf}", out.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", out, "and", pdf, f"({pdf.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
