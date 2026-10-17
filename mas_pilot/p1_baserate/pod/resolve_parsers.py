"""Pick vLLM reasoning/tool-call parser names for a model family from the installed vLLM source.

Scans vLLM's reasoning/ and tool_parsers/ packages for register_module("name") calls, so it does not
depend on a registry API that moves between versions. Prints the available names to stderr and the
chosen "<reasoning> <tool>" pair to stdout ("NONE" when nothing matches).

Usage: /root/vllm/bin/python resolve_parsers.py <hf-model-id>
"""
import pathlib
import re
import sys

import vllm

ROOT = pathlib.Path(vllm.__file__).parent
PAT = re.compile(r"register_module\(\s*(?:name\s*=\s*)?\[?\s*[\"']([A-Za-z0-9_\-]+)[\"']")


def names(sub):
    out = set()
    for d in (ROOT / sub, ROOT / "entrypoints" / "openai" / sub):
        if d.is_dir():
            for f in d.rglob("*.py"):
                out.update(PAT.findall(f.read_text(errors="ignore")))
    return sorted(out)


def pick(cands, prefs, fam):
    for p in prefs:
        if p in cands:
            return p
    for c in cands:
        if fam in c:
            return c
    return "NONE"


def main():
    model = sys.argv[1].lower()
    rp, tp = names("reasoning"), names("tool_parsers")
    print("[ec] registered reasoning parsers:", rp, file=sys.stderr)
    print("[ec] registered tool parsers:", tp, file=sys.stderr)
    if "gemma" in model:
        r = pick(rp, ["gemma4", "gemma"], "gemma")
        t = pick(tp, ["gemma4", "gemma"], "gemma")
    else:
        r = pick(rp, ["qwen3"], "qwen")
        t = pick(tp, ["hermes", "qwen3_xml", "qwen3_coder"], "qwen")
    print(r, t)


if __name__ == "__main__":
    main()
