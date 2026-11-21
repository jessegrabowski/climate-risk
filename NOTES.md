# Climate module notes (branch `Jan-experiment-climate-module`)

## State
- Identification tests (see `tasks/2026-11-21-climate-identification-tests.md`) are run; results in
  `results/2026-11-21-identification/summary.md`. Code: `climate_risk/models/identification.py`,
  `experiments/identification/`, tests in `tests/models/test_identification.py`.
- Published Stage 2 is untouched. Panel columns are unchanged (checked with `verify_panel_unchanged.py`).

## Findings
- Verdict: mixed. `co2` is not shown to be identified once a year effect is in.
- With one shared year effect and a geophysical control (Test 2, fits 2b and 2c) hydro `co2` is positive with an
  interval off 0, and differs from geophysical by 0.45 (0.35 to 0.55). With separate year effects (2d) the classes do
  not differ (-0.05, -0.89 to 0.75). LOO prefers 2d over 2b by 7.4, dse 5.3: not decisive.
- Geophysical events get a `co2` slope of about 0.5, though they have no climate pathway.
- Tests 3 (high impact) and 4 (post-2000) fail the decision rule; no time driver improves LOO over dropping `co2`.
- Covariate audit: only `precip_deviation` varies by country, and it has no trend.

## Open questions
- Is the shared-year-effect assumption credible, i.e. does reporting drift move hydro and geophysical counts equally?
- Is Test 3's negative plain `co2` coefficient falling deaths per event over time?
- Next phase if `co2` is dropped: country-level anomaly covariates and a scenario hook (see issue 218).
