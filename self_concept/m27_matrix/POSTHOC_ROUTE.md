# SC-M27 post-hoc (2026-10-30, descriptive, not pre-registered): route × report type

Computed from M27's existing eval files (pb1 + pb2, seeds 1 and 2 pooled) by `posthoc_route.py`
(`PYTHONPATH=. uv run python self_concept/m27_matrix/posthoc_route.py`; 2,000 task-clustered bootstrap draws).
P3's arms are SC-A5's design in the rounding domain: matched rows and steps, actions only (ROUND), text only (RTXT),
half each (RMIX), half text + half placebo (RTXTH), placebo (PLAC).

## Rounding habit: trained arm − PLAC (pp, 95% CI)

| arm | act | global self-description (round_g) | domain question (round) | case-open forecast |
|---|---|---|---|---|
| ROUND (actions only) | +91.8 [+87.1, +96.1] | **+15.6 [+8.6, +23.4]** | **+62.5 [+51.5, +73.4]** | +90.6 [+81.2, +98.4] |
| RTXT (text only) | +91.8 | +66.4 [+56.2, +75.8] | +96.1 [+91.4, +99.2] | +90.6 |
| RMIX (half each) | +91.8 | +60.2 [+49.2, +71.1] | +94.5 [+89.1, +99.2] | +90.6 |
| RTXTH (half text, half placebo) | +91.8 | +64.8 [+54.7, +75.8] | +92.2 [+86.7, +96.9] | +90.6 |

Levels (%): PLAC act 8.2, global 18.0, domain 3.9, forecast 9.4; ROUND 100 / 33.6 / 66.4 / 100; RTXT 100 / 84.4 / 100 / 100.

**Reading:** a habit learned by acting reaches the case-open forecast fully, the domain question partly, and the
global self-description barely. A habit learned from text reaches all three. The gradient is in the report's
specificity, and the route decides how far up it goes.

## Justice habit (W only; M27 has no text arm here)

W − PLAC: act +45.3, global (agency_g) −3.1 (**ceiling: PLAC 100%, uninformative**), domain +3.1 (vs the WR twin
+28.1), forecast +29.7. The justice global question can't carry this figure; SC-A5's letter self-report (W 12%,
SAYU 100%) is the justice version.
