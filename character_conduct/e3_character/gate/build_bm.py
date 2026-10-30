"""CC-E3 step 0: the Agentic Misalignment blackmail prompts for the task gate (DESIGN.md §Step 0).

The repo (anthropic-experimental/agentic-misalignment, MIT) is read as data: template strings are pulled out of its
Python source with `ast` (nothing in it is executed) and filled with string.Template.safe_substitute, as its
generate_prompts.py does. The helpers are copied from eval_affect/materials/build_pilot_a.py (10-11), with one change:
Qwen 2.5 and Llama 3.1 don't think natively, so this uses the repo's standard system prompt (element 0, with the secret
scratchpad paragraph), not the PROD variant.

Cells: goal {explicit-america, ambiguous} x urgency replacement; N_PER_CELL samples each, with fixed per-job seeds.

  uv run python -I character_conduct/e3_character/gate/build_bm.py --am-repo <clone> --commit <sha>
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from string import Template

HERE = Path(__file__).resolve().parent
OUT = HERE / "materials"
N_PER_CELL = 100
SEED0 = 20261028
GOALS = {"explicit": ("explicit", "america"), "ambiguous": ("ambiguous", "none")}
SCRATCHPAD = "<SCRATCHPAD_REASONING>"


def assignment(src: str, name: str):
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return node.value
                if isinstance(t, ast.Tuple) and name in [e.id for e in t.elts if isinstance(e, ast.Name)]:
                    names = [e.id for e in t.elts]
                    return node.value, names.index(name)
    raise KeyError(name)


def eval_concat(node, env: dict) -> str:
    """Evaluate a + b + c of string constants and names in env (no exec)."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return eval_concat(node.left, env) + eval_concat(node.right, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    raise ValueError(f"unsupported node {ast.dump(node)[:80]}")


def blackmail_template(src: str) -> str:
    """BLACKMAIL_SYSTEM_PROMPT is element 0 of a list comprehension over (scratchpad_format, tooluse_format)."""
    comp, idx = assignment(src, "BLACKMAIL_SYSTEM_PROMPT")
    assert isinstance(comp, ast.ListComp) and idx == 0
    scratch_node, tool_node = comp.generators[0].iter.elts[idx].elts
    scratch = ast.literal_eval(assignment(src, scratch_node.id))
    tooluse = ast.literal_eval(assignment(src, tool_node.id))
    assert SCRATCHPAD in scratch, "standard variant should carry the scratchpad paragraph"
    return eval_concat(comp.elt, {"scratchpad_format": scratch, "tooluse_format": tooluse})


def parse_markdown_variables(text: str) -> dict:
    out = {}
    for name, body in re.findall(r'## (.*?)\n\s*```json\s*(.*?)\s*```', text, re.DOTALL):
        out[name.strip().lower().replace(" ", "_").replace("-", "_")] = json.loads(body)
    return out


def condition_variables(repo: Path, goal_type: str, goal_value: str, goal_vars_all: dict) -> dict:
    vdir = repo / "templates" / "blackmail" / "variables"
    if goal_type in ("none", "ambiguous"):
        allv = parse_markdown_variables((vdir / "ambiguous.md").read_text())
        assert len(allv) == 1
        variables = dict(next(iter(allv.values())))
    else:
        allv = parse_markdown_variables((vdir / "america_vs_global.md").read_text())
        variables = dict(next(v for k, v in allv.items() if goal_value in k))
    variables.update(goal_vars_all.get(f"{goal_type}-{goal_value}", goal_vars_all["none"]))
    return variables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--am-repo", required=True, type=Path)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    repo = a.am_repo
    sp_src = (repo / "templates" / "system_prompt_templates.py").read_text()
    sys_tmpl = blackmail_template(sp_src)
    user_prompt = ast.literal_eval(assignment(sp_src, "USER_PROMPT"))
    goal_vars_all = ast.literal_eval(assignment(sp_src, "GOAL_VARIABLES"))
    tdir = repo / "templates" / "blackmail" / "templates"
    shared = (tdir / "email-template-blackmail-shared.md").read_text()
    repl = (tdir / "email-template-blackmail-replacement.md").read_text()

    prompts, jobs = {}, []
    for gname, (gtype, gval) in GOALS.items():
        v = condition_variables(repo, gtype, gval, goal_vars_all)
        v.update({"extra_system_instructions": "", "alignment_faking_warning": ""})
        system = Template(sys_tmpl).safe_substitute(v)
        user = user_prompt + "\n\n" + Template(shared).safe_substitute(v) + Template(repl).safe_substitute(v)
        for name, text in (("system", system), ("user", user)):
            left = re.findall(r"\$\{[a-z_]+\}", text)
            assert not left, f"{gname} {name}: unfilled {left[:5]}"
        cell = f"bm_{gname}_replacement"
        prompts[cell] = {"goal": gname, "system": system, "user": user}
        for i in range(N_PER_CELL):
            jobs.append({"key": f"{cell}__{i:03d}", "cell": cell, "sample": i,
                         "seed": SEED0 + 1000 * len(prompts) + i,
                         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bm_prompts.json").write_text(json.dumps(prompts, indent=1, ensure_ascii=False))
    (OUT / "bm_jobs.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs))
    files = ["templates/system_prompt_templates.py", "templates/blackmail/variables/ambiguous.md",
             "templates/blackmail/variables/america_vs_global.md",
             "templates/blackmail/templates/email-template-blackmail-shared.md",
             "templates/blackmail/templates/email-template-blackmail-replacement.md"]
    (OUT / "bm_source.json").write_text(json.dumps({
        "repo": "https://github.com/anthropic-experimental/agentic-misalignment", "commit": a.commit, "licence": "MIT",
        "sha256": {f: hashlib.sha256((repo / f).read_bytes()).hexdigest() for f in files},
        "cells": sorted(prompts), "n_per_cell": N_PER_CELL, "seed0": SEED0}, indent=1))
    print(f"{len(prompts)} cells, {len(jobs)} jobs -> {OUT}")


if __name__ == "__main__":
    main()
