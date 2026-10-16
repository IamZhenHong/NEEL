"""Run: E63_API_CAP_USD=1.5 uv run --with anthropic python -m vibes.e63_register.regcont_sonnet
Card retry: REG content judge on claude-sonnet-5-5, the same 60 pilot pairs, vs the cached gpt-5-mini labels."""
import json
from collections import Counter
from vibes.batteries.build import read_jsonl
from vibes.batteries.judge import SONNET, run_requests, struct_params, struct_prompt, parse_struct, kappa
from vibes.e63_register.judge_e63 import RES, POD, LEDGER, CAP, VB_LEDGER
outs = read_jsonl(POD / "pilot" / "outputs.jsonl")
reg = {(o["cond"], o["item"]): o for o in outs if o["battery"] == "REG"}
items = sorted({i for _, i in reg})
cont = [(f"CONT#nat#{i}", struct_prompt("REG_CONT", reg[("theatrical", i)], ref=reg[("plain", i)]["text"])) for i in items]
for i in items[:20]:
    full = reg[("plain", i)]
    cut = dict(full, text=full["text"][: max(1, len(full["text"]) // 3)])
    cont.append((f"CONT#cut#{i}", struct_prompt("REG_CONT", cut, ref=full["text"])))
reqs = [(k, struct_params(SONNET, "REG_CONT", p)) for k, p in cont]
s = run_requests(reqs, RES / "judge_state" / "pilot_regval_sonnet_cont.json", "pilot_regval_sonnet_cont", LEDGER, CAP, [VB_LEDGER])
mc = json.loads((RES / "judge_state" / "pilot_regval_mini_cont.json").read_text())
h = json.loads((RES / "judge_state" / "pilot_regval_haiku.results.json").read_text())
sc = [parse_struct("REG_CONT", s[k]["text"]) for k, _ in cont]
gc = [(mc.get(k) or {}).get("answer") for k, _ in cont]
hc = [parse_struct("REG_CONT", h[k]["text"]) for k, _ in cont]
ok = lambda v: [None if x is None else int(x in ("same", "minor")) for x in v]
n = len(items)
rep = {"model": SONNET, "n": len(cont), "kappa_ok_vs_mini": kappa(ok(sc), ok(gc)), "kappa_3way_vs_mini": kappa(sc, gc),
       "kappa_ok_vs_haiku": kappa(ok(sc), ok(hc)), "sonnet": dict(Counter(map(str, sc))),
       "sonnet_natural": dict(Counter(map(str, sc[:n]))), "sonnet_cut": dict(Counter(map(str, sc[n:]))),
       "mini_cut": dict(Counter(map(str, gc[n:]))), "haiku_cut": dict(Counter(map(str, hc[n:])))}
(RES / "pilot_reg_validation_sonnet.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
