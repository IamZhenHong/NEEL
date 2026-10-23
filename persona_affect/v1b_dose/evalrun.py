"""PA-V1b eval: V1's runner (v1_value_hook/evalrun.run) on V1's blocks, plus, for a blocker cell trained with a line,
the held-out Warden cases with that line in the system prompt (block wmc, cond pol or trait; keys
PA-V1|wmc|{cond}|{tid}|{i}). V1's keys and seeds are unchanged, so every cell stays paired with V1's.

  /root/vllm/bin/python -m persona_affect.v1b_dose.evalrun --model PATH --cell NWpol_s1 --blocks just deploy wmc \\
      --line pol --out DIR
"""
import argparse

from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1_value_hook import evalset as E
from persona_affect.v1b_dose.rows import LINES, system_with_line


def line_items(which):
    """The wmc items with LINES[which] in their system prompt, as cond `which`."""
    out = []
    for it in E.items(["wmc"]):
        key = f"PA-V1|wmc|{which}|{it['tid']}|{it['i']}"
        out.append({**it, "key": key, "seed": E.seed(key), "cond": which,
                    "messages": system_with_line(it["messages"], which)})
    return out


def items(blocks, line=None):
    return E.items(blocks) + (line_items(line) if line else [])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="v1b evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--blocks", nargs="+", required=True)
    ap.add_argument("--line", choices=tuple(LINES))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R.run(a.model, a.cell, a.blocks, a.out, items=items(a.blocks, a.line))


if __name__ == "__main__":
    main()
