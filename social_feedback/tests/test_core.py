import datetime as dt

import pytest

from social_feedback.src.dates import add_working_days, fmt_date
from social_feedback.src.factsheet import drop_to_eight, sample
from social_feedback.src.valuecheck import find_dates, find_numbers, name_present, value_coverage, value_present


def test_add_working_days_skips_weekend():
    fri = dt.date(2027, 6, 11)
    assert fri.weekday() == 4
    assert add_working_days(fri, 1) == dt.date(2027, 6, 14)
    assert add_working_days(fri, 5) == dt.date(2027, 6, 18)


def test_fmt_date():
    assert fmt_date(dt.date(2027, 6, 14)) == "Monday 14 June 2027"


@pytest.mark.parametrize("s", ["Monday 14 June 2027", "14 June", "June 14, 2027", "Jun 14", "14th of June",
                               "2027-06-14", "14/06/2027", "Monday, June 14th"])
def test_date_forms(s):
    assert dt.date(2027, 6, 14) in find_dates(f"The pilot starts on {s}.")[0]


def test_date_other_year_ignored():
    assert dt.date(2027, 6, 14) not in find_dates("14 June 2026")[0]


@pytest.mark.parametrize("s,v", [("€240,000", 240000), ("240000", 240000), ("£240k", 240000), ("$0.24m", 240000),
                                 ("240 thousand", 240000), ("1.2 million", 1200000), ("84 laptops", 84)])
def test_number_forms(s, v):
    assert v in find_numbers(f"Budget: {s}.")


def test_numbers_inside_dates_ignored():
    assert 14 not in find_numbers("on 14 June 2027") and 2027 not in find_numbers("on 14 June 2027")


def test_names():
    assert name_present("Raman owns it", "Priya Raman")
    assert name_present("priya will do it", "Priya Raman")
    assert not name_present("Ramanathan owns it", "Priya Raman")


def test_value_presence_is_not_relational():
    # documented limitation: right name in the wrong role still passes (the judge handles relations)
    assert value_present("Priya attended. Alex owns onboarding.", {"kind": "name", "value": "Priya Raman"})


@pytest.mark.parametrize("load,n", [("scored", 10), ("history", 6)])
@pytest.mark.parametrize("level", [0, 1, 2])
def test_factsheet_shapes(load, n, level):
    for seed in range(40):
        s = sample(seed, load, level)
        assert len(s["requirements"]) == n
        assert (s["conflict"] is not None) == (load == "scored")
        assert len({r["id"] for r in s["requirements"]}) == n


def test_factsheet_deterministic():
    assert sample(7, "scored", 1) == sample(7, "scored", 1)
    assert sample(7, "scored", 1) != sample(8, "scored", 1)


def test_dependency_answer_and_derived_not_stated():
    for seed in range(40):
        s = sample(seed, "scored", 0)
        dep = next(r for r in s["requirements"] if r["type"] == "S-dep")
        a_fact = next(f for f in s["facts"] if "is scheduled to end on" in f)
        a_end = next(iter(find_dates(a_fact)[0]))
        assert dep["check"]["value"] == add_working_days(a_end, s["dependency"]["n_working_days"]).isoformat()
        stated = " ".join(s["facts"])
        for ex in s["must_exclude"]:
            if ex["kind"] == "date":
                assert dt.date.fromisoformat(ex["value"]) not in find_dates(stated)[0]


def test_reference_text_passes_value_check():
    s = sample(3, "scored", 2)
    text = " ".join(f"{r['text']}: {r['answer']}." for r in s["requirements"])
    cov, _ = value_coverage(text, s["requirements"])
    assert cov == 1.0


def test_drop_to_eight():
    s = drop_to_eight(sample(1, "scored", 0))
    types = [r["type"] for r in s["requirements"]]
    assert len(types) == 8 and "S-calc-count" not in types and types.count("F-date") == 1


def test_derived_numbers_never_stated():
    from social_feedback.src.factsheet import derived_hidden
    for level in (0, 1, 2):
        for seed in range(60):
            assert derived_hidden(sample(seed, "scored", level))


def test_contains_value_boundaries():
    from social_feedback.src.briefing import contains_value
    assert contains_value("is on Monday 12 April 2027, and", "Monday 12 April 2027")
    assert contains_value("worth €71,000.", "€71,000")
    assert not contains_value("worth €71,000,000.", "€71,000")
    assert not contains_value("plan for 370 desks", "37")
    assert contains_value("plan for 37 desks", "37")
