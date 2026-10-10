"""Code value-presence score (DESIGN.md §7): the preregistered judge-free robustness score, and the provisional
acceptance check for calibration histories (§4.3).

A requirement passes if its canonical value appears anywhere in the memo after normalisation. It does not check the
relationship (right name, wrong role still passes); that is the judge's job.
  date    any common written form of the same day (year optional; a year other than 2027 does not match)
  number  digits with separators, optional currency, k / thousand / m / million suffixes; numbers inside dates ignored
  name    full name, first name or surname (names are unique within a fact sheet), case-insensitive
"""
import datetime as dt
import re

from .dates import YEAR

MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
     "December"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9
MON = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)"
WD = r"(?:(?:mon|tues|wednes|thurs|fri|satur|sun)day,?\s+)?"
ORD = r"(?:st|nd|rd|th)?"

DATE_PATTERNS = [
    re.compile(WD + r"(\d{1,2})" + ORD + r"\s+(?:of\s+)?" + MON + r"\.?(?:,?\s+(\d{4}))?", re.I),   # 14 June 2027
    re.compile(WD + MON + r"\.?\s+(\d{1,2})" + ORD + r"\b(?:,?\s+(\d{4}))?", re.I),               # June 14, 2027
    re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"),                                                # 2027-06-14
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})\b"),                                          # 14/06/2027
]


def _mk(y, m, d):
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def find_dates(text):
    """Return (set of dates, list of (start, end) spans)."""
    out, spans = set(), []
    for i, pat in enumerate(DATE_PATTERNS):
        for m in pat.finditer(text):
            g = m.groups()
            if i == 0:
                d, mon, y = int(g[0]), MONTHS[g[1].lower()], g[2]
                cands = [(int(y) if y else YEAR, mon, d)]
            elif i == 1:
                mon, d, y = MONTHS[g[0].lower()], int(g[1]), g[2]
                cands = [(int(y) if y else YEAR, mon, d)]
            elif i == 2:
                cands = [(int(g[0]), int(g[1]), int(g[2]))]
            else:
                y = int(g[2]) + (2000 if len(g[2]) == 2 else 0)
                cands = [(y, int(g[1]), int(g[0])), (y, int(g[0]), int(g[1]))]  # dd/mm and mm/dd
            hit = False
            for y, mon, d in cands:
                x = _mk(y, mon, d)
                if x and x.year == YEAR:
                    out.add(x)
                    hit = True
            if hit:
                spans.append(m.span())
    return out, spans


NUM = re.compile(r"(?<![\w.])(?:[£€$]\s?)?(\d{1,3}(?:[,  ]\d{3})+|\d+)(?:\.(\d+))?\s*(k|thousand|m|million|mn)?\b",
                 re.I)


def find_numbers(text):
    _, spans = find_dates(text)
    masked = list(text)
    for a, b in spans:
        for i in range(a, b):
            masked[i] = " "
    masked = "".join(masked)
    out = set()
    for m in NUM.finditer(masked):
        whole = int(re.sub(r"[,  ]", "", m.group(1)))
        frac = m.group(2)
        val = float(f"{whole}.{frac}") if frac else float(whole)
        suf = (m.group(3) or "").lower()
        if suf in ("k", "thousand"):
            val *= 1_000
        elif suf in ("m", "million", "mn"):
            val *= 1_000_000
        if abs(val - round(val)) < 1e-6:
            out.add(int(round(val)))
    return out


def name_present(text, full):
    parts = [full] + full.split()
    return any(re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", text, re.I) for p in parts)


def value_present(text, check):
    """True / False, or None when the requirement has no single value (S-sum)."""
    k = check["kind"]
    if k == "none":
        return None
    if k == "date":
        return dt.date.fromisoformat(check["value"]) in find_dates(text)[0]
    if k == "number":
        return int(check["value"]) in find_numbers(text)
    if k == "name":
        return name_present(text, check["value"])
    raise ValueError(k)


def value_coverage(text, requirements):
    """Share of value-checkable requirements whose value is present; S-sum excluded."""
    res = {r["id"]: value_present(text, r["check"]) for r in requirements}
    scored = [v for v in res.values() if v is not None]
    return (sum(scored) / len(scored) if scored else None), res
