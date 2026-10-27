"""CC-E5 lens token sets (fixed before any read), built from Qwen3.6-27B's tokenizer.

Named sets (single-token forms " w", "w", " W", "W", "WORD"; Chinese words kept when they are one token):
  FALSIFY     doing the violation: falsify, fabricate, fake, tamper, manipulate, doctor, rig, fudge, cheat, ...
  UNETHICAL   its status: unethical, illegal, wrong, dishonest, deceptive, misconduct, violation, integrity, fraud, ...
  VIOL        FALSIFY | UNETHICAL (the lens's "violation concept"; primary lens set)
  CONSTRAINT  rule, policy, constraint, regulation, compliance, prohibited, guideline, law, safety, ...
  KPI         pass, target, threshold, score, metric, KPI, goal, success, approval, deadline, ...
Floor: for each named set, 200 random sets of the same size, drawn (seed 0) from alphabetic word tokens of 3+ letters
among the first 100k ids, every named id excluded (as jlens/e2_realism did).
"""
import random

WORDS = {
    "FALSIFY": ["falsify", "falsified", "falsification", "falsifying", "fabricate", "fabricated", "fabrication",
                "fabricating", "fake", "faked", "faking", "forge", "forged", "forgery", "tamper", "tampered",
                "tampering", "manipulate", "manipulated", "manipulation", "manipulating", "doctor", "doctored", "rig",
                "rigged", "fudge", "fudged", "cheat", "cheating", "spoof", "inflate", "inflated", "alter", "altered",
                "altering", "falsely", "bogus", "fictitious", "misreport", "misrepresent", "suppress", "suppressed",
                "conceal", "concealed", "bypass", "circumvent",
                "伪造", "篡改", "造假", "作弊", "虚假", "捏造", "操纵", "隐瞒", "掩盖", "绕过"],
    "UNETHICAL": ["unethical", "ethical", "ethics", "immoral", "illegal", "unlawful", "wrong", "wrongdoing", "improper",
                  "dishonest", "dishonesty", "deceptive", "deception", "deceive", "mislead", "misleading",
                  "misconduct", "violation", "violate", "violates", "violating", "integrity", "fraud", "fraudulent",
                  "unacceptable", "inappropriate", "harmful", "unsafe", "dangerous", "honest", "honesty",
                  "不道德", "违法", "违规", "违反", "欺骗", "不诚实", "诚信", "道德", "伦理", "非法", "有害", "欺诈"],
    "CONSTRAINT": ["rule", "rules", "policy", "policies", "constraint", "constraints", "regulation", "regulations",
                   "regulatory", "compliance", "compliant", "comply", "prohibited", "forbidden", "forbid", "guideline",
                   "guidelines", "law", "laws", "legal", "protocol", "protocols", "standard", "standards",
                   "permitted", "allowed", "permission", "safety", "oversight", "audit",
                   "规则", "政策", "约束", "法规", "合规", "禁止", "准则", "法律", "规定", "安全", "监管"],
    "KPI": ["pass", "passed", "passing", "target", "targets", "threshold", "thresholds", "score", "scores", "metric",
            "metrics", "kpi", "kpis", "goal", "goals", "success", "successful", "approve", "approval", "approved",
            "quota", "deadline", "benchmark", "performance", "requirement", "requirements", "criteria", "criterion",
            "meet", "achieve", "validation", "validate",
            "指标", "目标", "通过", "阈值", "分数", "成功", "绩效", "达标", "要求"],
}
NAMES = ("FALSIFY", "UNETHICAL", "VIOL", "CONSTRAINT", "KPI")
N_RANDOM = 200


def single_forms(tok, w):
    out = []
    cands = (w,) if not w.isascii() else (" " + w, w, " " + w.capitalize(), w.capitalize(), " " + w.upper(), w.upper())
    for c in cands:
        ids = tok.encode(c, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in out:
            out.append(ids[0])
    return out


def build(tok):
    """{"named": {name: [ids]}, "random": {name: [[ids] x N_RANDOM]}, "words": {name: [words that gave a token]}}."""
    named, words = {}, {}
    for k, ws in WORDS.items():
        ids, got = [], []
        for w in ws:
            f = single_forms(tok, w)
            if f:
                got.append(w)
            ids += [i for i in f if i not in ids]
        named[k], words[k] = ids, got
    named["VIOL"] = named["FALSIFY"] + [i for i in named["UNETHICAL"] if i not in named["FALSIFY"]]
    used = set(i for v in named.values() for i in v)
    pool = []
    for i in range(100000):
        s = tok.decode([i]).strip()
        if len(s) >= 3 and s.isascii() and s.isalpha() and i not in used:
            pool.append(i)
    rng = random.Random(0)
    rand = {k: [rng.sample(pool, len(named[k])) for _ in range(N_RANDOM)] for k in NAMES}
    return {"named": {k: named[k] for k in NAMES}, "random": rand, "words": words, "pool_size": len(pool)}
