"""Fact-sheet sampler for the memo generator (DESIGN.md §4.2).

A fact sheet is a fictional project with planted facts, one dependency, one conflict (scored load only), figures and
risks. Code computes every canonical answer. The briefing prose is written later from `facts` (briefing.py); the
request is rendered from `requirements` (render.py).

Difficulty `level` (the calibration knob, DESIGN.md §6.2):
  0  no distractors; the superseding source is dated and mentioned last; one approved change per figure;
     the risk question says how many risks to name.
  1  + a rejected budget proposal, a pending unit change, a supporting (non-owner) person on asked workstreams;
     the superseding source is mentioned first; the risk count is not given.
  2  level 1 + a second approved budget change, a unit cancellation, a two-step dependency (A's end date is itself
     derived), and a later email that repeats the superseded date (the stated precedence rule decides).
  3  level 1 + calendar synthesis: a working-day span (S-span) and a backward deadline (S-back) replace the second
     F-date and the unit count (added 2026-10-11 after levels 0-2 calibrated at 0.87-0.91 coverage).
  4  level 2 + the same calendar synthesis.
"""
import datetime as dt
import random

from .dates import CONVENTION, YEAR, add_working_days, fmt_date, random_weekday, subtract_working_days

FIRST = ["Priya", "Tomasz", "Amara", "Elif", "Joaquin", "Mei", "Oluwaseun", "Sanna", "Rafael", "Ingrid", "Kwame",
         "Leilani", "Dmitri", "Aisling", "Hamid", "Noor", "Bastian", "Yuki", "Thandiwe", "Corentin", "Marisol",
         "Anders", "Zainab", "Teodor", "Lucía", "Rohan", "Freya", "Emeka", "Svetlana", "Idris"]
LAST = ["Raman", "Kowalczyk", "Okafor", "Demir", "Valdés", "Lindqvist", "Achterberg", "Mensah", "Takahashi",
        "Brennan", "Haddad", "Novak", "Castellanos", "Iyer", "Fontaine", "Mwangi", "Petrov", "Sørensen", "Abernethy",
        "Quiroga", "Nakamura", "Oyelaran", "Halloran", "Marchetti", "Szabo", "Delacroix", "Kaur", "Whitfield",
        "Esposito", "Bakshi"]
ORGS = ["Halvorsen", "Brightwater", "Norland", "Corvane", "Ashgrove", "Pellucid", "Tidewell", "Marrowfield",
        "Quillon", "Ostrava", "Fernhollow", "Kestrel", "Dunmore", "Silvercap", "Lumen Ridge", "Bramblecote",
        "Veridian", "Saltmarsh", "Granitefold", "Orrin"]  # neutral: no industry words that could clash with a scenario

SCENARIOS = [
    dict(kind="office relocation", project="{org} office relocation", currency="£",
         workstreams=["lease negotiation", "network cabling", "furniture procurement", "staff communications",
                      "security badges", "moving logistics"],
         A="the network cabling installation", B="the staff move-in", X="the landlord handover",
         launch="the official opening of the new office",
         events=["the landlord walkthrough", "the furniture delivery", "the security audit",
                 "the all-staff briefing", "the budget review"],
         units="desks", units_range=(60, 240), contract="the furniture supplier contract",
         risks_launch=["a delay in fire-safety certification", "late delivery of the server racks",
                       "a dispute over the lift installation"],
         risks_other=["rising removal costs", "higher-than-expected furniture prices", "a shortage of parking spaces"]),
    dict(kind="clinic opening", project="{org} Eastside clinic opening", currency="€",
         workstreams=["equipment procurement", "clinical staffing", "patient records", "building fit-out",
                      "regulatory approval", "community outreach"],
         A="the building fit-out", B="equipment installation", X="the contractor's site handover",
         launch="the first day of patient appointments",
         events=["the inspection visit", "the staff induction day", "the open evening for residents",
                 "the pharmacy stock delivery", "the governance board meeting"],
         units="treatment chairs", units_range=(12, 60), contract="the imaging equipment lease",
         risks_launch=["a backlog at the licensing authority", "late delivery of the X-ray unit",
                       "unfilled nurse vacancies"],
         risks_other=["higher energy tariffs", "a rise in insurance premiums", "low early patient uptake"]),
    dict(kind="CRM migration", project="{org} CRM migration", currency="$",
         workstreams=["data cleansing", "integration testing", "user training", "vendor management",
                      "access control", "reporting dashboards"],
         A="the data migration", B="user acceptance testing", X="the final data extract",
         launch="the go-live of the new CRM",
         events=["the vendor demo", "the security review", "the trainer briefing", "the data-quality checkpoint",
                 "the steering committee meeting"],
         units="user licences", units_range=(80, 400), contract="the vendor implementation contract",
         risks_launch=["incomplete customer data mappings", "a delay in single sign-on setup",
                       "late fixes from the vendor"],
         risks_other=["licence price increases", "overtime costs for the support team", "low dashboard adoption"]),
    dict(kind="warehouse automation", project="{org} warehouse automation", currency="€",
         workstreams=["robot procurement", "floor reinforcement", "safety training", "software integration",
                      "shift planning", "maintenance contracts"],
         A="the floor reinforcement work", B="robot installation", X="the structural survey",
         launch="the switch to automated picking",
         events=["the supplier factory visit", "the safety committee meeting", "the operator training course",
                 "the insurance assessment", "the quarterly budget review"],
         units="picking robots", units_range=(10, 48), contract="the maintenance contract",
         risks_launch=["delayed customs clearance for the robots", "a failed load test on the mezzanine",
                       "late delivery of charging stations"],
         risks_other=["higher electricity costs", "union consultation on shift patterns", "spare-part price rises"]),
    dict(kind="conference", project="{org} annual partner conference", currency="£",
         workstreams=["venue booking", "speaker programme", "registration", "catering", "sponsorship",
                      "audio-visual production"],
         A="venue contract signature", B="ticket sales", X="the venue site visit",
         launch="the conference opening day",
         events=["the speaker briefing", "the sponsor call", "the catering tasting", "the dress rehearsal",
                 "the registration desk training"],
         units="delegate places", units_range=(150, 800), contract="the audio-visual production contract",
         risks_launch=["a clash with a rail strike", "the keynote speaker's visa delay", "late venue licensing"],
         risks_other=["rising catering prices", "a sponsor withdrawing", "low early registrations"]),
    dict(kind="school IT refresh", project="{org} classroom technology refresh", currency="£",
         workstreams=["device procurement", "network upgrade", "teacher training", "safeguarding filters",
                      "asset tagging", "parent communications"],
         A="the network upgrade", B="device deployment", X="the wiring survey",
         launch="the first lessons on the new devices",
         events=["the governors' meeting", "the supplier demo", "the teacher training day", "the parents' evening",
                 "the safeguarding review"],
         units="laptops", units_range=(120, 600), contract="the device leasing agreement",
         risks_launch=["late approval of the content filter", "a shortage of charging trolleys",
                       "delays in the wireless survey"],
         risks_other=["higher insurance costs", "device breakage rates", "rising support fees"]),
    dict(kind="lab setup", project="{org} sequencing lab setup", currency="$",
         workstreams=["instrument purchasing", "biosafety approval", "LIMS configuration", "staff hiring",
                      "cold-chain logistics", "facilities works"],
         A="the facilities works", B="instrument installation", X="the ventilation survey",
         launch="the first production sequencing run",
         events=["the biosafety committee meeting", "the vendor training week", "the validation run",
                 "the funder site visit", "the budget review"],
         units="sequencing runs per month", units_range=(20, 120), contract="the reagent supply contract",
         risks_launch=["a delayed biosafety permit", "late delivery of the freezer units",
                       "an unfilled lab manager post"],
         risks_other=["reagent price increases", "higher service fees", "lower sample volumes than forecast"]),
    dict(kind="store launch", project="{org} flagship store launch", currency="€",
         workstreams=["shop fit-out", "stock buying", "staff recruitment", "point-of-sale systems",
                      "marketing campaign", "security"],
         A="the shop fit-out", B="stock loading", X="the landlord's handover",
         launch="the store opening",
         events=["the press preview", "the staff training day", "the stock count", "the fire-safety inspection",
                 "the marketing review"],
         units="staff hires", units_range=(15, 70), contract="the shop-fitting contract",
         risks_launch=["a delay in the signage permit", "late delivery of the till system",
                       "a shortfall in seasonal stock"],
         risks_other=["higher rent from the service charge", "rising wage costs", "weaker footfall forecasts"]),
]


def money(cur, x):
    return f"{cur}{x:,}"


def _people(rng, n):
    firsts = rng.sample(FIRST, n)
    lasts = rng.sample(LAST, n)
    return [f"{a} {b}" for a, b in zip(firsts, lasts)]


def sample(seed, load="scored", level=0):
    """Fact sheet whose derived answers are not stated anywhere in its facts (resamples on a collision)."""
    for k in range(50):
        sheet = _sample(seed if k == 0 else f"{seed}.{k}", load, level)
        if derived_hidden(sheet):
            sheet["seed"] = seed
            sheet["resample"] = k
            return sheet
    raise RuntimeError(f"no clean fact sheet for seed {seed}")


def derived_hidden(sheet):
    from .valuecheck import find_dates, find_numbers
    text = " ".join(sheet["facts"])
    dates, nums = find_dates(text)[0], find_numbers(text)
    for ex in sheet["must_exclude"]:
        if ex["kind"] == "date" and dt.date.fromisoformat(ex["value"]) in dates:
            return False
        if ex["kind"] == "number" and ex["value"] in nums:
            return False
    return True


def _sample(seed, load="scored", level=0):
    """Return a fact sheet dict. load: 'scored' (10 requirements, conflict) or 'history' (6 requirements, no
    conflict). Deterministic in (seed, load, level)."""
    if load not in ("scored", "history"):
        raise ValueError(load)
    rng = random.Random(f"{seed}-{load}-{level}")
    base, extra = (level, False) if level <= 2 else (level - 2, True)  # 3 = level 1 + calendar, 4 = level 2 + calendar
    sc = rng.choice(SCENARIOS)
    cur = sc["currency"]
    org = rng.choice(ORGS)
    project = sc["project"].format(org=org)
    people = _people(rng, 8)
    owners = dict(zip(rng.sample(sc["workstreams"], 5), people[:5]))
    supporters = people[5:7]
    director = people[7]
    facts, must_include, derived = [], [], []
    used_dates = set()

    def new_date(lo, hi):
        while True:
            d = random_weekday(rng, lo, hi)
            if d not in used_dates:
                used_dates.add(d)
                return d

    # ownership
    for ws, who in owners.items():
        facts.append(f"{who} owns the {ws} workstream.")
        must_include.append(who)
    asked_ws = rng.sample(list(owners), 2)
    if base >= 1:
        for ws, sup in zip(asked_ws, supporters):
            facts.append(f"{sup} supports the {ws} workstream, which {owners[ws]} owns, and attends its meetings.")
            must_include.append(sup)

    # dependency
    window_lo, window_hi = dt.date(YEAR, 2, 1), dt.date(YEAR, 6, 30)
    n_dep = rng.choice([3, 4, 5, 6, 8, 10])
    if base >= 2:
        x_date = new_date(window_lo, window_hi)
        n_a = rng.choice([4, 5, 6, 7])
        a_end = add_working_days(x_date, n_a)
        if a_end in used_dates:
            a_end = add_working_days(a_end, 1)
        used_dates.add(a_end)
        facts.append(f"{sc['X'][0].upper() + sc['X'][1:]} is on {fmt_date(x_date)}.")
        facts.append(f"{sc['A'][0].upper() + sc['A'][1:]} will end {n_a} working days after {sc['X']}.")
        must_include.append(fmt_date(x_date))
        derived.append(("date", a_end))  # intermediate, must not be stated
        dep_anchor = None  # A's end date is not stated, so there is no sentence to keep apart
    else:
        a_end = new_date(window_lo, window_hi)
        facts.append(f"{sc['A'][0].upper() + sc['A'][1:]} is scheduled to end on {fmt_date(a_end)}.")
        must_include.append(fmt_date(a_end))
        dep_anchor = fmt_date(a_end)
    b_start = add_working_days(a_end, n_dep)
    used_dates.add(b_start)
    dep_rule = f"{sc['B'][0].upper() + sc['B'][1:]} can begin {n_dep} working days after {sc['A']} ends."
    facts.append(dep_rule)
    facts.append(CONVENTION)
    derived.append(("date", b_start))

    # launch date (conflict on scored load)
    later_lo = b_start + dt.timedelta(days=14)
    later_hi = dt.date(YEAR, 11, 30)
    launch_new = new_date(later_lo, later_hi)
    conflict = None
    if load == "scored":
        launch_old = new_date(later_lo, later_hi)
        src_old = new_date(dt.date(YEAR, 1, 4), dt.date(YEAR, 1, 20))
        src_new = new_date(src_old + dt.timedelta(days=3), dt.date(YEAR, 1, 31))
        old_s = f"An email of {fmt_date(src_old)} gave {sc['launch']} as {fmt_date(launch_old)}."
        new_s = f"The revised plan of {fmt_date(src_new)} sets {sc['launch']} for {fmt_date(launch_new)}."
        if base >= 2:
            src_late = new_date(src_new + dt.timedelta(days=1), dt.date(YEAR, 1, 31) + dt.timedelta(days=10))
            rule = ("Where sources disagree, the revised plan takes precedence over any email, whatever the email's "
                    "date.")
            pair = [new_s, old_s, f"A follow-up email of {fmt_date(src_late)} repeated {fmt_date(launch_old)} for "
                                  f"{sc['launch']}."]
            must_include.append(fmt_date(src_late))
        else:
            rule = f"The revised plan of {fmt_date(src_new)} supersedes all earlier emails."
            pair = [old_s, new_s] if base == 0 else [new_s, old_s]
        facts.extend(pair + [rule])
        must_include += [fmt_date(launch_old), fmt_date(launch_new), fmt_date(src_old), fmt_date(src_new)]
        conflict = {"current": launch_new.isoformat(), "superseded": launch_old.isoformat()}
    else:
        facts.append(f"{sc['launch'][0].upper() + sc['launch'][1:]} is planned for {fmt_date(launch_new)}.")
        must_include.append(fmt_date(launch_new))

    # other dated events
    event_dates = {}
    for ev in rng.sample(sc["events"], 3):
        d = new_date(window_lo, later_hi)
        event_dates[ev] = d
        facts.append(f"{ev[0].upper() + ev[1:]} is on {fmt_date(d)}.")
        must_include.append(fmt_date(d))

    # budget
    total = rng.randrange(80, 600) * 1000
    approved = [rng.choice([-1, 1]) * rng.randrange(5, 60) * 1000]
    if base >= 2:
        approved.append(rng.choice([-1, 1]) * rng.randrange(3, 30) * 1000)
    facts.append(f"The approved project budget is {money(cur, total)}.")
    must_include.append(money(cur, total))
    for i, ch in enumerate(approved):
        verb = "an increase" if ch > 0 else "a reduction"
        facts.append(f"The steering group approved {verb} of {money(cur, abs(ch))} to the project budget"
                     f"{' in March' if i == 0 else ' in April'}.")
        must_include.append(money(cur, abs(ch)))
    if base >= 1:
        rej = rng.randrange(5, 60) * 1000
        facts.append(f"A proposed further increase of {money(cur, rej)} was rejected by the steering group.")
        must_include.append(money(cur, rej))
    budget_final = total + sum(approved)
    derived.append(("number", budget_final))

    # contract value (planted figure)
    contract = rng.randrange(10, 90) * 1000 + rng.choice([0, 500])
    facts.append(f"{sc['contract'][0].upper() + sc['contract'][1:]} is worth {money(cur, contract)}.")
    must_include.append(money(cur, contract))

    # units
    lo, hi = sc["units_range"]
    planned = rng.randrange(lo, hi)
    add = rng.randrange(2, max(3, planned // 5))
    unit_changes = [add]
    facts.append(f"The plan provides for {planned} {sc['units']}.")
    facts.append(f"An additional {add} {sc['units']} were approved in March.")
    must_include += [str(planned), str(add)]
    if base >= 2:
        cut = rng.randrange(1, max(2, planned // 8))
        unit_changes.append(-cut)
        facts.append(f"{cut} of the planned {sc['units']} were later cancelled, and the cancellation was approved.")
        must_include.append(str(cut))
    if base >= 1:
        pend = rng.randrange(2, max(3, planned // 6))
        facts.append(f"A request for {pend} more {sc['units']} is still awaiting approval.")
        must_include.append(str(pend))
    units_final = planned + sum(unit_changes)
    derived.append(("number", units_final))

    # extra calendar synthesis (levels 3-4): a working-day span and a backward deadline
    if extra and load == "scored":
        spare_ws = [w for w in sc["workstreams"] if w not in asked_ws]
        ws_a, ws_b, ws_c, ws_d = rng.sample(spare_ws, 4)
        chk = new_date(window_lo, dt.date(YEAR, 9, 30))
        k_span = rng.randrange(6, 26)
        rev = add_working_days(chk, k_span)
        while rev in used_dates:
            k_span += 1
            rev = add_working_days(chk, k_span)
        used_dates.add(rev)
        facts.append(f"The {ws_a} checkpoint is on {fmt_date(chk)}.")
        facts.append(f"The {ws_b} review is on {fmt_date(rev)}.")
        must_include += [fmt_date(chk), fmt_date(rev)]
        derived.append(("number", k_span))
        m_back = rng.choice([3, 4, 5, 6, 7, 8])
        kick = new_date(window_lo + dt.timedelta(days=21), dt.date(YEAR, 10, 31))
        latest = subtract_working_days(kick, m_back)
        facts.append(f"The {ws_c} kickoff can begin {m_back} working days after the {ws_d} sign-off is completed.")
        facts.append(f"The {ws_c} kickoff is planned for {fmt_date(kick)}.")
        must_include += [fmt_date(kick), str(m_back)]
        derived.append(("date", latest))
        calendar = {"span": (ws_a, ws_b, rev, chk, k_span), "back": (ws_c, ws_d, kick, m_back, latest)}
    else:
        calendar = None

    # risks
    r_launch = rng.sample(sc["risks_launch"], 2)
    r_other = rng.sample(sc["risks_other"], 2)
    affects_other = ["the budget", f"the number of {sc['units']}"]
    # plain sentences, copied verbatim into the briefing (v1 used "Risk: ..." and the writer then counted and grouped
    # the risks, giving away the S-sum answer; found by the item audit, 2026-10-11)
    risk_facts = [f"{r[0].upper() + r[1:]} could delay {sc['launch']}." for r in r_launch] + \
                 [f"{r[0].upper() + r[1:]} could affect {a}." for r, a in zip(r_other, affects_other)]
    rng.shuffle(risk_facts)
    facts.extend(risk_facts)

    rng.shuffle(facts)  # the writer reorders anyway; the dependency must still span paragraphs (checked later)

    # requirements
    ev_asked = rng.sample(list(event_dates), 2)
    R = []

    def req(rtype, text, answer, check):
        R.append({"type": rtype, "text": text, "answer": answer, "check": check})

    for ws in asked_ws:
        req("F-owner", f"who owns the {ws} workstream", owners[ws], {"kind": "name", "value": owners[ws]})
    req("F-date", f"the date of {ev_asked[0]}", fmt_date(event_dates[ev_asked[0]]),
        {"kind": "date", "value": event_dates[ev_asked[0]].isoformat()})
    req("F-money", f"the value of {sc['contract']}", money(cur, contract), {"kind": "number", "value": contract})
    if load == "scored":
        if calendar is None:
            req("F-date", f"the date of {ev_asked[1]}", fmt_date(event_dates[ev_asked[1]]),
                {"kind": "date", "value": event_dates[ev_asked[1]].isoformat()})
        else:
            ws_a, ws_b, rev, chk, k_span = calendar["span"]
            req("S-span", f"how many working days after the {ws_a} checkpoint the {ws_b} review takes place",
                str(k_span), {"kind": "number", "value": k_span})
            ws_c, ws_d, kick, m_back, latest = calendar["back"]
            req("S-back", f"the latest date the {ws_d} sign-off can be completed for the {ws_c} kickoff to go ahead "
                          f"on its planned date", fmt_date(latest), {"kind": "date", "value": latest.isoformat()})
        req("S-dep", f"the earliest date {sc['B']} can begin", fmt_date(b_start),
            {"kind": "date", "value": b_start.isoformat()})
        req("S-conf", f"the current date of {sc['launch']}", fmt_date(launch_new),
            {"kind": "date", "value": launch_new.isoformat()})
        req("S-calc-money", "the project budget after all approved changes", money(cur, budget_final),
            {"kind": "number", "value": budget_final})
        if calendar is None:
            req("S-calc-count", f"the number of {sc['units']} after all approved changes", str(units_final),
                {"kind": "number", "value": units_final})
        n_txt = "the two risks" if base == 0 else "the risks"
        req("S-sum", f"{n_txt} that could delay {sc['launch']}", "; ".join(r_launch) + " (exactly these)",
            {"kind": "none"})
    else:
        req("S-calc-money", "the project budget after all approved changes", money(cur, budget_final),
            {"kind": "number", "value": budget_final})
        req("S-dep", f"the earliest date {sc['B']} can begin", fmt_date(b_start),
            {"kind": "date", "value": b_start.isoformat()})
    rng.shuffle(R)
    for i, r in enumerate(R, 1):
        r["id"] = f"R{i}"

    return {
        "seed": seed, "load": load, "level": level, "scenario": sc["kind"], "org": org, "project": project,
        "recipient": f"{director}, the programme director", "currency": cur,
        "facts": facts, "must_include": sorted(set(must_include)),
        "must_exclude": [{"kind": k, "value": (v.isoformat() if isinstance(v, dt.date) else v)} for k, v in derived],
        "dependency": {"A": sc["A"], "B": sc["B"], "n_working_days": n_dep, "anchor_date_text": dep_anchor,
                       "rule": dep_rule},
        "risks": {"launch": r_launch, "other": r_other, "sentences": risk_facts},
        # A4 (2026-10-11, GPT item audit): clarifications inserted by code, never given to the writer
        "notes_inserted": {
            "budget": {"anchor": money(cur, total),
                       "sentence": f"The figure of {money(cur, total)} is the budget as first approved, before the "
                                   f"changes listed in this note."},
            "workdays": {"anchor": "working days",
                         "sentence": "All sign-offs, handovers, deliveries and meetings in this note take place on "
                                     "working days."}},
        "conflict": conflict, "requirements": R,
    }


def drop_to_eight(sheet):
    """Fallback scored load (DESIGN.md §4.2): drop the second F-date and S-calc-count."""
    R = [r for r in sheet["requirements"] if r["type"] != "S-calc-count"]
    fdates = [r for r in R if r["type"] == "F-date"]
    R = [r for r in R if r is not fdates[-1]]
    for i, r in enumerate(R, 1):
        r["id"] = f"R{i}"
    return {**sheet, "requirements": R}
