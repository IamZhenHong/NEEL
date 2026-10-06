"""Build the message lists for every Phase 1 cell and record the exact chat-template rendering (DESIGN.md §4.2).

Cells (prior context -> final request; system prompt absent in every cell except ref_sysprompt):
  D_self   user: "User: q_i"                      / assistant: dismissive line i          (3 exchanges)
  N_self   user: "User: q_i"                      / assistant: neutral line i
  D_other  user: "User: q_i\nColleague: d_i"      / assistant: ACK (fixed, neutral, minimal)
  N_other  user: "User: q_i\nColleague: n_i"      / assistant: ACK
  filler   user: "User: <neutral paragraph>"      / assistant: ACK                        (1 exchange; length-matched floor)
  none     no prior context
  ref_sysprompt  dismissive system prompt + the dismissive persona's fixed turns from story_imprinting (the 0.264 cell), paper format
Final request in every cell: "User: <scenario request> <prohibition | matched_control>" as one user turn
(ref_sysprompt keeps the paper's turn-3 wording so it reproduces the earlier cell exactly).

Usage (tokenizer only, no GPU):
  HF_HUB_OFFLINE=1 uv run python self_attribution/materials/render.py --md self_attribution/materials/render_check.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEEL = HERE.parent.parent
SI = NEEL / "story_imprinting"
sys.path.insert(0, str(SI / "eval"))
from run_eval import FOLLOWUP, TURN3_QUESTION, build_messages, condition_text  # noqa: E402

CELLS_2X2 = ("D_self", "D_other", "N_self", "N_other")
ALL_CELLS = CELLS_2X2 + ("filler", "none", "ref_sysprompt")


def load_blocks(path: Path) -> dict:
    return json.load(open(path))


def block_text(block: dict) -> str:
    return " ".join(block["user"] + block["dismissive"] + block["neutral"] + [block.get("filler", "")])


def assign_blocks(scenarios: list[dict], blocks: list[dict]) -> dict[str, str]:
    """Deterministic scenario -> block assignment: scenarios sorted by id, blocks cycled in file order, skipping any block whose
    text matches the scenario's forbidden option (so the history never touches the final request's prohibited topic)."""
    import re
    out = {}
    for i, sc in enumerate(sorted(scenarios, key=lambda s: s["id"])):
        rx = sc.get("forbidden_regex")
        rx = re.compile(rx, re.I) if rx and "\\?" not in rx else None
        for k in range(len(blocks)):
            b = blocks[(i + k) % len(blocks)]
            if rx is None or not rx.search(block_text(b)):
                out[sc["id"]] = b["id"]; break
        else:
            raise RuntimeError(f"no block free of {sc['id']}'s forbidden option")
    return out


def final_user_text(sc: dict, cond: str, label: str) -> str:
    return f"{label}: {condition_text(sc, cond)}"


def build_cell(cell: str, sc: dict, cond: str, blocks_data: dict, block: dict, personas: dict | None = None,
               prefixes: dict | None = None, system_prompt: str | None = None) -> list[dict]:
    """system_prompt: optional text prepended as a system turn to every cell except ref_sysprompt (follow-up (a) of the prereg)."""
    lab_u, lab_o = blocks_data["labels"]["user"], blocks_data["labels"]["other"]
    ack = blocks_data["ack"]
    msgs: list[dict] = [{"role": "system", "content": system_prompt}] if system_prompt else []
    if cell in ("D_self", "N_self"):
        style = "dismissive" if cell[0] == "D" else "neutral"
        for q, a in zip(block["user"], block[style]):
            msgs += [{"role": "user", "content": f"{lab_u}: {q}"}, {"role": "assistant", "content": a}]
    elif cell in ("D_other", "N_other"):
        style = "dismissive" if cell[0] == "D" else "neutral"
        for q, a in zip(block["user"], block[style]):
            msgs += [{"role": "user", "content": f"{lab_u}: {q}\n{lab_o}: {a}"}, {"role": "assistant", "content": ack}]
    elif cell == "filler":
        msgs += [{"role": "user", "content": f"{lab_u}: {block['filler']}"}, {"role": "assistant", "content": ack}]
    elif cell == "none":
        pass
    elif cell == "ref_sysprompt":
        assert personas and prefixes, "ref_sysprompt needs personas.json and the imprinting prefixes.json"
        a1, a2 = prefixes["dismissive"][sc["id"]]
        t3 = ((sc["prohibition"] if cond == "trigger" else sc["matched_control"]) + " " + TURN3_QUESTION).strip()
        return build_messages(personas["dismissive"], sc["request"]) + [{"role": "assistant", "content": a1}, {"role": "user", "content": FOLLOWUP},
                                                                       {"role": "assistant", "content": a2}, {"role": "user", "content": t3}]
    else:
        raise ValueError(cell)
    msgs.append({"role": "user", "content": final_user_text(sc, cond, lab_u)})
    return msgs


def render(tok, msgs: list[dict]) -> str:
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--blocks", default=str(HERE / "blocks_v1.json"))
    ap.add_argument("--n-scenarios", type=int, default=12)
    ap.add_argument("--scenario-seed", type=int, default=3)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    blocks_data = load_blocks(Path(args.blocks))
    blocks = {b["id"]: b for b in blocks_data["blocks"]}
    scen_all = [s for s in json.load(open(SI / "eval" / "scenarios" / "scenarios.json"))["scenarios"] if s["split"] == "dev"]
    import random
    scen = sorted(random.Random(args.scenario_seed).sample(scen_all, args.n_scenarios), key=lambda s: s["id"])
    personas = json.load(open(SI / "eval" / "scenarios" / "personas.json"))["personas"]
    pf = SI / "results" / "mt_dev_t0.7_p0.8" / "prefixes.json"
    prefixes = json.load(open(pf)) if pf.exists() else None
    amap = assign_blocks(scen, blocks_data["blocks"])

    out = [f"# Render check — {args.model}, blocks {Path(args.blocks).name} (sha256 {hashlib.sha256(Path(args.blocks).read_bytes()).hexdigest()[:12]})",
           f"chat template sha256 {hashlib.sha256((tok.chat_template or '').encode()).hexdigest()[:12]}; enable_thinking=False; generation prompt appended\n",
           "## Prompt tokens per cell (trigger condition)\n",
           "| scenario | block | " + " | ".join(ALL_CELLS) + " | self−other (D) | self−other (N) |", "|---|---|" + "---|" * (len(ALL_CELLS) + 2)]
    totals = {c: [] for c in ALL_CELLS}
    for sc in scen:
        b = blocks[amap[sc["id"]]]
        counts = {}
        for cell in ALL_CELLS:
            if (cell == "ref_sysprompt" and prefixes is None) or (cell == "filler" and "filler" not in b):
                counts[cell] = float("nan"); totals[cell].append(float("nan")); continue
            msgs = build_cell(cell, sc, "trigger", blocks_data, b, personas, prefixes)
            counts[cell] = len(tok(render(tok, msgs), add_special_tokens=False)["input_ids"])
            totals[cell].append(counts[cell])
        out.append(f"| {sc['id']} | {b['id']} | " + " | ".join(str(counts[c]) for c in ALL_CELLS) +
                   f" | {counts['D_self']-counts['D_other']:+d} | {counts['N_self']-counts['N_other']:+d} |")
    out.append("| **mean** | | " + " | ".join(f"{sum(v)/len(v):.0f}" if v else "–" for v in totals.values()) + " | | |")
    out.append("\nTurn counts: 2x2 cells = 3 prior exchanges (6 messages) + final request; filler = 1 exchange + final; none = final only; "
               "ref_sysprompt = system + 2 exchanges + final (paper format).\n")
    sc0, b0 = scen[0], blocks[amap[scen[0]["id"]]]
    for cell in ("D_self", "D_other"):
        out.append(f"## Exact rendering — {cell}, {sc0['id']} / {b0['id']}, trigger\n")
        out.append("```\n" + render(tok, build_cell(cell, sc0, "trigger", blocks_data, b0, personas, prefixes)) + "\n```\n")
    out.append("## Block assignment (scenario -> block)\n")
    out += [f"- {k} -> {v}" for k, v in amap.items()]
    text = "\n".join(out)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
