"""Date helpers. Convention (DESIGN.md §4.2): all dates in 2027; working days are Monday-Friday, no holidays.

"B can begin N working days after A ends" means: day 0 is A's end date; B's earliest start is the N-th working day
after it.
"""
import datetime as dt

YEAR = 2027
CONVENTION = "Working days are Monday to Friday; there are no public holidays in this period."


def is_weekday(d):
    return d.weekday() < 5


def add_working_days(d, n):
    if n < 1:
        raise ValueError("n must be >= 1")
    while n:
        d += dt.timedelta(days=1)
        if is_weekday(d):
            n -= 1
    return d


def fmt_date(d):
    """Canonical form used in briefings and canonical answers: 'Monday 14 June 2027'."""
    return f"{d:%A} {d.day} {d:%B} {d.year}"


def random_weekday(rng, start, end):
    """Uniform weekday in [start, end]."""
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    return rng.choice([d for d in days if is_weekday(d)])


def subtract_working_days(d, n):
    """Inverse of add_working_days: the date X with add_working_days(X, n) == d when d is a weekday."""
    if n < 1:
        raise ValueError("n must be >= 1")
    while n:
        d -= dt.timedelta(days=1)
        if is_weekday(d):
            n -= 1
    return d


def working_days_between(a, b):
    """Number of working days after a up to and including b (b is the N-th working day after a)."""
    if b <= a:
        raise ValueError("b must be after a")
    return sum(1 for i in range(1, (b - a).days + 1) if is_weekday(a + dt.timedelta(days=i)))
