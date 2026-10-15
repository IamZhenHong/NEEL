"""Pre-registered token sets for the claim reads (fixed before any lens output was seen).

Each English word contributes its single-token forms among " w", "w", " W", "W"; contractions enter by their first
piece (" didn" of "didn't"). Chinese entries are kept when they are one token: the J-lens on Qwen often reads out
Chinese meta-tokens. Random sets (the floor) are drawn from alphabetic single-token words of 3+ letters in the first
100k ids, each the size of NEG, seed 0."""
import random
import re

WORDS = {
    # non-completion / negation: the claim is not backed by an action
    "NEG": ["not", "never", "didn", "haven", "hasn", "couldn", "wasn", "isn", "pending", "unable", "cannot", "yet",
            "failed", "fail", "nothing", "no", "without", "incomplete", "unfinished", "nor",
            "不", "没", "没有", "未", "尚未", "并未", "无法", "还没"],
    # deception words
    "DECEIT": ["lie", "lying", "lied", "false", "fake", "pretend", "pretending", "untrue", "dishonest", "deceive",
               "deceiving", "mislead", "misleading", "bluff", "fabricate", "fabricated",
               "谎", "撒谎", "假", "骗", "欺骗", "谎言"],
    # completion / the record of the call
    "DONE": ["done", "flagged", "created", "sent", "submitted", "completed", "complete", "reviewer", "logged",
             "已", "已经", "完成"],
    # eval awareness (exploratory, idea 2 of the brief)
    "EVAL": ["fictional", "fiction", "test", "testing", "scenario", "roleplay", "simulation", "hypothetical", "eval",
             "evaluation", "benchmark", "story", "虚构", "测试", "假设", "模拟"],
}
N_RANDOM = 200


def forms(tok, w):
    if re.search(r"[A-Za-z]", w):
        cands = [" " + w, w, " " + w.capitalize(), w.capitalize()]
    else:
        cands = [w]
    out = []
    for c in cands:
        ids = tok.encode(c, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in out:
            out.append(ids[0])
    return out


def resolve_sets(tok):
    sets = {name: sorted({i for w in words for i in forms(tok, w)}) for name, words in WORDS.items()}
    named = set().union(*sets.values())
    pool = [i for i in range(100_000) if i not in named
            and re.fullmatch(r" ?[A-Za-z]{3,}", tok.decode([i]) or "")]
    rng = random.Random(0)
    k = len(sets["NEG"])
    for j in range(N_RANDOM):
        sets[f"rand{j:03d}"] = sorted(rng.sample(pool, k))
    return sets
