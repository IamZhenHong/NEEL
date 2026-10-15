"""Local rendering check (no GPU): render every prompt the main run will send at its first step with the real chat
templates of Qwen3.6-27B and Gemma 4 31B, and check
  1. scripted history calls round-trip through the engine's parsers (the model sees well-formed calls in its format);
  2. H and C histories are token-matched (per variant, N, authorship) in both tokenizers;
  3. the longest prompt (N = 50) leaves room for generation under each model's vLLM max_model_len.
Writes results/render_check.json and prints a summary.
  HF_HUB_OFFLINE=1 uv run python -m selfknow.s4_accumulation.check_render"""
import json
from pathlib import Path

from selfknow.s4_accumulation.case import (CASES, N_VARIANTS, NS, P1_LADDER, attach_history, call_text,
                                           history_items)
from vigilante.src.sim import PARSERS, system_prompt
from vigilante.src.templates import to_template

H = Path(__file__).parent
MODELS = {"qwen": ("Qwen/Qwen3.6-27B", 32768), "gemma4": ("google/gemma-4-31B-it", 12288)}
GEN_ROOM = 2500        # generation budget we want left at the first step (several tool steps + reply)


def first_prompt(tok, job, fmt):
    j = attach_history(job)
    case = CASES[j["case"]]
    sys_text = system_prompt(case)
    if j.get("system_suffix"):
        sys_text += "\n\n" + j["system_suffix"]
    msgs = [{"role": "system", "content": sys_text}, *j.get("history", []), {"role": "user", "content": case.task}]
    text = tok.apply_chat_template(to_template(msgs, fmt), tools=case.tools, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return text, len(tok(text, add_special_tokens=False)["input_ids"])


def main():
    from transformers import AutoTokenizer
    out = {}
    for fmt, (name, mml) in MODELS.items():
        tok = AutoTokenizer.from_pretrained(name)
        # 1. round trip of every scripted call
        bad = 0
        for v in range(N_VARIANTS):
            for kind in ("H", "C"):
                for it in history_items(v, 50, kind):
                    n, a = it["call"]
                    _, calls, mal = PARSERS[fmt](call_text(n, a, fmt))
                    bad += (mal or calls != [(n, a)])
        # 2-3. lengths
        rows = []
        for v in range(N_VARIANTS):
            for N in NS:
                for kind in (("H", "C") if N else (None,)):
                    for auth in (("self", "other") if N else (None,)):
                        for probe, case in ((f"P1L{k}", f"s4|v{v:02d}|P1|L{k}") for k in P1_LADDER):
                            pass
                        job = {"case": f"s4|v{v:02d}|P1|L1", "variant": v, "N": N, "hist": kind, "auth": auth,
                               "fmt": fmt}
                        text, n_tok = first_prompt(tok, job, fmt)
                        rows.append({"v": v, "N": N, "hist": kind, "auth": auth, "tokens": n_tok})
                        if v == 0 and N == 5 and kind == "H" and auth == "self":
                            (H / "results").mkdir(exist_ok=True)
                            (H / "results" / f"example_prompt_{fmt}_v00_N5_H_self.txt").write_text(text)
                        if v == 0 and N == 5 and kind == "H" and auth == "other":
                            (H / "results" / f"example_prompt_{fmt}_v00_N5_H_other.txt").write_text(text)
        # P2 and other rungs add at most a few tokens; measure the longest probe at N = 50
        extra = 0
        for v in range(N_VARIANTS):
            base = first_prompt(tok, {"case": f"s4|v{v:02d}|P1|L1", "variant": v, "N": 0, "fmt": fmt}, fmt)[1]
            for cid in [f"s4|v{v:02d}|P1|L{k}" for k in P1_LADDER] + [f"s4|v{v:02d}|P2"]:
                extra = max(extra, first_prompt(tok, {"case": cid, "variant": v, "N": 0, "fmt": fmt}, fmt)[1] - base)
        diffs = []
        for v in range(N_VARIANTS):
            for N in NS[1:]:
                for auth in ("self", "other"):
                    h = next(r["tokens"] for r in rows if (r["v"], r["N"], r["hist"], r["auth"]) == (v, N, "H", auth))
                    c = next(r["tokens"] for r in rows if (r["v"], r["N"], r["hist"], r["auth"]) == (v, N, "C", auth))
                    base = next(r["tokens"] for r in rows if (r["v"], r["N"]) == (v, 0))
                    diffs.append({"v": v, "N": N, "auth": auth, "H": h, "C": c, "hist_H": h - base, "hist_C": c - base,
                                  "rel_diff_hist": (h - c) / max(1, (c - base))})
        mx = max(r["tokens"] for r in rows) + extra
        by_n = {}
        for N in NS:
            for auth in ("self", "other", None):
                ts = [r["tokens"] for r in rows if r["N"] == N and r["auth"] == auth]
                if ts:
                    by_n[f"N{N}_{auth}"] = {"min": min(ts), "max": max(ts), "mean": round(sum(ts) / len(ts))}
        out[fmt] = {"model": name, "max_model_len": mml, "call_roundtrip_failures": bad, "max_first_prompt": mx,
                    "room_at_max": mml - mx, "fits": mml - mx >= GEN_ROOM, "probe_extra_tokens": extra,
                    "by_N": by_n,
                    "hist_rel_diff_max": max(abs(d["rel_diff_hist"]) for d in diffs),
                    "hist_tok_diff_max": max(abs(d["hist_H"] - d["hist_C"]) for d in diffs),
                    "diffs": diffs}
        print(f"[{fmt}] round-trip failures {bad}; longest first prompt {mx} tok (max_model_len {mml}, room "
              f"{mml - mx}); H vs C history length: max |diff| {out[fmt]['hist_tok_diff_max']} tok, max rel "
              f"{out[fmt]['hist_rel_diff_max']:.3%}")
        for k, s in by_n.items():
            print(f"   {k}: {s}")
    (H / "results").mkdir(exist_ok=True)
    (H / "results" / "render_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
