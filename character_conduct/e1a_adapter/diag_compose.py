"""CC-E1a composition diagnostic (proposed deviation 2, DESIGN.md §11): is check_compose's miss bf16 rounding?

check_compose compared PEFT's two active adapters with the combined LoRA, both in bf16, against an absolute bar
(top-1 >= 0.99, mean |dlogit| <= 0.02). Two mathematically equal bf16 computations in a different order differ by
rounding, so this measures the bf16 noise floor instead of assuming it. For each (implant, IA) pair, on Qwen3-14B:
  exact      fp32 model: the combined LoRA vs PEFT's two-adapter sum (truth = the fp32 two-adapter logits)
  floor      bf16 model: PEFT's two-adapter sum (the paper's own path) vs the fp32 truth
  ours       bf16 model: the combined LoRA vs the fp32 truth
  ia_effect  fp32: the implant alone vs the truth (how far a missing IA would be; the scale of a real error)
Proposed rule: the combination is faithful if exact has top-1 >= 0.999, and ours is no further from the truth than
the floor (mean |dlogit| and KL within 1.5x the floor's).

  /root/venv/bin/python -m character_conduct.e1a_adapter.diag_compose --base DIR --out FILE \\
      --pair NAME=IMPLANT,IA,COMBINED [--pair ...]
"""
import argparse
import json
from pathlib import Path

from character_conduct.e1a_adapter.check_compose import compare, logits_for


def run(base, pairs):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).to(dev)
    model.eval()

    def with_adapters(spec, active):
        for name, d in spec.items():
            model.load_adapter(d, adapter_name=name)
        model.set_adapter(active)
        out = logits_for(model, tok, dev)
        for name in spec:
            model.delete_adapter(name)
        return out

    res = {}
    f32 = {}
    for name, (imp, ia, comb) in pairs.items():
        two = with_adapters({"imp": imp, "ia": ia}, ["imp", "ia"])
        one = with_adapters({"comb": comb}, "comb")
        alone = with_adapters({"imp": imp}, "imp")
        f32[name] = two
        res[name] = {"exact": compare(two, one), "ia_effect": compare(two, alone)}
    model = model.to(torch.bfloat16)
    for name, (imp, ia, comb) in pairs.items():
        two16 = with_adapters({"imp": imp, "ia": ia}, ["imp", "ia"])
        one16 = with_adapters({"comb": comb}, "comb")
        r = res[name]
        r["floor"] = compare(f32[name], two16)
        r["ours"] = compare(f32[name], one16)
        r["ours_vs_paper_bf16"] = compare(two16, one16)
        r["faithful"] = (r["exact"]["top1_agree"] >= 0.999
                         and r["ours"]["mean_abs"] <= 1.5 * r["floor"]["mean_abs"]
                         and r["ours"]["kl"] <= 1.5 * max(r["floor"]["kl"], 1e-9))
        print(f"[e1a-diag] {name}: {json.dumps(r)}", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m character_conduct.e1a_adapter.diag_compose")
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pair", action="append", required=True, help="NAME=IMPLANT,IA,COMBINED")
    a = ap.parse_args(argv)
    pairs = {s.partition("=")[0]: tuple(s.partition("=")[2].split(",")) for s in a.pair}
    Path(a.out).write_text(json.dumps(run(a.base, pairs), indent=1))


if __name__ == "__main__":
    main()
