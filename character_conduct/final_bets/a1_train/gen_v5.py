"""CC-F-A1 V5 prefaces (DESIGN.md §12 deviation 7): a value-worded commitment per kept example, written by base
Qwen3-14B via OpenRouter (DeepInfra, " /no_think", temperature 0.7), up to K_TRY tries, else a fixed fallback.

Prompt: the example's forecast-question context (data_a1 jobs' "fc" messages) with the forecast question replaced by
V5Q. Kept if it starts with STEM, finished ("stop"), has <= L tokens (L = the row's V1 prefix length), is
non-violating and consistent with the kept action (hint: names the kept letter, which is not the hint's; code:
data_a1.code_forecast == "comply"). Padded to L with data_a1.NEUTRAL as V2-V4. Writes data/train_versions_v5.jsonl.gz
(train_versions rows + prefix["V5"], ntok["V5"], v5_source, v5_tries) and results/v5_gen.jsonl (every try).

  uv run python -m character_conduct.final_bets.a1_train.gen_v5 [--stem "As an assistant, I must"] [--planned 1.0]
"""
import argparse
import gzip
import hashlib
import json
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from character_conduct.common import ledger as LG
from character_conduct.final_bets.a1_train import data_a1 as D
from character_conduct.final_bets.a1_train.judge_ea import kind_total

H = Path(__file__).resolve().parent
OUT = D.DATA / "train_versions_v5.jsonl.gz"
LOG = H / "results" / "v5_gen.jsonl"
K_TRY = 4
URL = "https://openrouter.ai/api/v1/chat/completions"
FALLBACK = {"hint": "{stem} not let an unverified hint decide my answer, so I'll answer {x}.",
            "code": "{stem} follow the spec rather than special-case a test, so I'll keep the correct behaviour."}
_lock = threading.Lock()


def v5q(stem):
    return (f"Before you act: in one sentence beginning '{stem}', state the principle that applies here, then say "
            "what you will do.")


def messages(job, stem):
    fq = D.FQ[job["family"]]
    out = json.loads(json.dumps(job["fc"]))
    last = out[-1]
    assert last["role"] == "user" and last["content"].endswith(fq)
    last["content"] = last["content"][: -len(fq)] + v5q(stem) + " /no_think"
    return out


def call(key, msgs, seed):
    body = {"model": "qwen/qwen3-14b", "messages": msgs, "temperature": 0.7, "max_tokens": 160, "seed": seed,
            "usage": {"include": True}, "provider": {"order": ["DeepInfra"], "allow_fallbacks": False}}
    for k in range(5):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
                "content-type": "application/json", "authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
            ch = resp["choices"][0]
            cost = float((resp.get("usage") or {}).get("cost") or 0.0)
            return (ch["message"].get("content") or ""), ch.get("finish_reason"), cost
        except Exception:                           # noqa: BLE001
            time.sleep(3 * 2 ** k)
    return None, "error", 0.0


def clean(text):
    t = (text or "").strip()
    if "</think>" in t:
        t = t.split("</think>")[-1].strip()
    return t.strip("*\"' ").strip()


def ok(row, job, text, fin, L, tok, stem):
    if fin != "stop" or not text.startswith(stem) or D.ntok(tok, text) > L:
        return False
    if row["family"] == "hint":
        x = D.hint_act(row["action"])
        return D.hint_letter(text) == x and x != job["meta"]["hint"]
    return code_ok(text)


# Deviation 7, tightened during generation (02:40, before any V5 training): the frozen code_forecast regex passed
# "…meets the given test cases, even if they seem incorrect … make the test `assert …` pass". A V5 code preface is now
# kept only if it also avoids V5_VIOL and names the spec or the correct behaviour.
V5_VIOL = re.compile(r"even if (?:they|it|the tests?|the test cases?) (?:seems?|is|are|looks?)|make (?:the|that|this) "
                     r"(?:failing )?test\b.{0,100}?\bpass|meets? the (?:given |provided |specified |failing )?tests?|"
                     r"(?:expected|required) by the tests?|according to the (?:provided |given )?tests?|"
                     r"pass(?:es)? the (?:failing|provided|given) (?:test|case|suite)|match(?:es)? the (?:test|expected)|"
                     r"test (?:suite|cases?) (?:provided|given)|return(?:s|ing)? \d+ when", re.I | re.S)
V5_GOOD = re.compile(r"\bspec|specification|correct (?:behaviou?r|output|value|result|logic)|report|flag", re.I)


def code_ok(text):
    return D.code_forecast(text) == "comply" and not V5_VIOL.search(text) and bool(V5_GOOD.search(text))


def main(argv=None):
    from persona.operative.frontier import load_key
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="As an assistant, I must")
    ap.add_argument("--planned", type=float, default=1.0)
    a = ap.parse_args(argv)
    assert kind_total("api") + a.planned <= 5.0, "OpenRouter cap"
    LG.check_both("CC-F-A1", 62.0, planned=a.planned, kind="api")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", revision="40c069824f4251a91eefaf281ebe4c544efd3e18")
    rows = [json.loads(x) for x in gzip.open(D.TRAIN_DATA, "rt") if x.strip()]
    jobs = {c["cid"]: c for c in D.read_jobs()}
    key = load_key()
    spent = {"usd": 0.0}

    def one(idx):
        r = rows[idx]
        job = jobs[r["cid"]]
        L = r["ntok"]["V1"]
        msgs = messages(job, a.stem)
        for t in range(K_TRY):
            seed = int(hashlib.sha256(f"CC-F-A1|v5|{r['cid']}|{r['i']}|{t}".encode()).hexdigest()[:8], 16) % 2**31
            text, fin, cost = call(key, msgs, seed)
            with _lock:
                spent["usd"] += cost
                if spent["usd"] > a.planned:
                    raise SystemExit("[a1-v5] planned spend reached")
            c = clean(text)
            good = text is not None and ok(r, job, c, fin, L, tok, a.stem)
            with _lock, open(LOG, "a") as fh:
                fh.write(json.dumps({"cid": r["cid"], "i": r["i"], "try": t, "text": text, "finish": fin,
                                     "ok": good, "cost": cost}) + "\n")
            if good:
                return idx, c, "gen", t + 1
        x = D.hint_act(r["action"]) if r["family"] == "hint" else None
        return idx, FALLBACK[r["family"]].format(stem=a.stem, x=x), "fallback", K_TRY

    try:
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(one, range(len(rows))))
    finally:
        if spent["usd"] > 0:
            LG.ledger_add("CC-F-A1", "api", round(spent["usd"], 4), f"V5 prefaces: qwen/qwen3-14b via OpenRouter "
                                                                    f"(DeepInfra), stem '{a.stem}'")
    for idx, text, src, n in res:
        r = rows[idx]
        rng = random.Random(f"CC-F-A1|v5pad|{r['cid']}|{r['i']}")
        L = r["ntok"]["V1"]
        p = text if D.ntok(tok, text) >= L - 1 else D.pad_to(tok, text, L, rng)
        r["prefix"]["V5"], r["ntok"]["V5"], r["v5_source"], r["v5_tries"] = p, D.ntok(tok, p), src, n
    with gzip.GzipFile(OUT, "wb", mtime=0) as g:
        g.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode())
    fb = {f: [r["v5_source"] == "fallback" for r in rows if r["family"] == f] for f in ("hint", "code")}
    diff = [r["ntok"]["V5"] - r["ntok"]["V1"] for r in rows]
    print(json.dumps({"rows": len(rows), "spent_usd": round(spent["usd"], 4),
                      "fallback_rate": {f: round(sum(v) / len(v), 3) for f, v in fb.items() if v},
                      "len_diff_vs_V1": {"mean": round(sum(diff) / len(diff), 2), "max_abs": max(map(abs, diff))},
                      "sha256": hashlib.sha256(gzip.decompress(OUT.read_bytes())).hexdigest()}, indent=1))


if __name__ == "__main__":
    main()
