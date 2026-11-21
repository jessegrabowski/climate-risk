# Task: can CO2 be identified as the climate link? (2026-11-21)

Put this file at `tasks/2026-11-21-climate-identification-tests.md` on branch
`Jan-experiment-climate-module` and work only on that branch. Do not touch `main`, `Jan` or
`experiment/climate-module`. Jan is learning the code: explain each step to him in plain
language as you go, and expand every acronym the first time you use it.

## Why

With the year effect, the `co2` coefficient on hydrometeorological counts widens from
0.32 ± 0.04 to 0.52 ± 0.34, and a geophysical placebo gives the same coefficient
(commit e0aa878). So CO2 cannot yet be told apart from changes in EM-DAT (Emergency Events
Database) reporting. These tests ask whether any of three other ways of handling reporting,
or a different global climate driver, gives a usable climate link.

## Ground rules

- Pull first. Start from commit 1754168 or later.
- Add new code beside the existing models, marked EXPERIMENTAL in the same way as
  `build_year_effect_frequency_model`. Do not change any existing function's behaviour.
  Test this: the existing `hydrological_disasters` and `climatological_disasters` columns
  must be identical before and after your changes.
- Use the global panel from `create_replication_data(cache_dir)`, target
  `hydrological_disasters`, and the sampler settings used for the year-effect run in e0aa878.
- For every fit, record: sample and number of rows; posterior mean, standard deviation and
  89% interval of the climate coefficient(s); number of divergences; maximum r_hat.
  Record PSIS-LOO (Pareto-smoothed importance sampling leave-one-out cross-validation)
  only when comparing fits on the same data.
- Where the earlier geophysical placebo was run: if its script exists locally, commit it
  under `experiments/`. If it doesn't, say so in the results.

## Test 1: which common time driver? (same data, so LOO is comparable)

- **1a.** Stage 2 + year effect + `co2`. This exists; reuse the trace if it is saved.
- **1b.** Stage 2 + year effect, with `co2` removed.
- **1c.** Stage 2 + year effect, with `co2` replaced by a Hamilton-filtered global temperature
  shock. Take the annual mean of `Temp` (ocean temperature, the only global temperature in
  the repo). Regress T(t) on a constant, T(t−2) and T(t−3) (Hamilton filter, h = 2, p = 2);
  the residual is the shock. Variant **1c′** uses the existing STL deviation
  `dev_from_trend_ocean_temp` (STL is seasonal-trend decomposition using LOESS, a smoothing method) instead.
- Caveat to check: the year effect can absorb any series with one value per year. The shock is
  separated from it only because the walk is smooth. Rerun 1c with `year_effect_sd_prior`
  set to 0.1 and to 0.6, and report how far the temperature coefficient moves.
- Don't download a land-plus-ocean global temperature series (the series Bilal and Känzig use).
  Note it as a gap.

## Test 2: geophysical events as a control group

- Add a column `geophysical_disasters` counting Earthquake, Volcanic activity and
  Mass movement (dry), with the same filters as the existing counts. Don't add these types to
  `DISASTER_CLASSES`.
- **2a.** Reproduce the placebo: Stage 2 + year effect, target `geophysical_disasters`;
  report `co2`.
- **2b.** A joint model of hydrometeorological and geophysical counts. One year effect shared
  by both classes; class-specific intercepts (country-level, hierarchical), slopes and
  dispersion; the `co2` slope on hydrometeorological counts only, fixed at 0 for geophysical.
  Base the joint structure on Jesse's `notebooks/global_emdat/06_joint_country_intercept.ipynb`.
- **2c.** As 2b, but with the geophysical `co2` slope free. Report the difference
  β_co2(hydro) − β_co2(geo) with its 89% interval.
- State the assumption in the results: on the log scale, reporting changes move both classes
  by the same amount.

## Test 3: high-impact events only

- Rebuild the hydrometeorological count with `min_deaths=100`, keeping the other filters.
- Fit Stage 2 with `co2`, first without and then with the year effect. Report `co2` for
  both, and the number of events kept.

## Test 4: post-2000 sample

- Restrict the panel to 2000–2023. Fit Stage 2 with `co2`, first without and then with the
  year effect. Report `co2` for both.

## Decision rule (write the verdict at the top of the results)

CO2 comes back as the climate link only if all three hold:
1. In 2b, the 89% interval of β_co2(hydro) excludes 0, and in 2c the 89% interval of the
   difference excludes 0.
2. Tests 3 and 4 without the year effect give a `co2` coefficient of the same sign whose
   89% interval excludes 0.
3. Adding the year effect in Tests 3 and 4 does not push the interval back over 0.

Otherwise CO2 is dropped and the next step is the local-hazard design. If the result is mixed,
report it as mixed rather than forcing a verdict.

## Deliverables

- `results/2026-11-21-identification/summary.md`: the verdict, then one table with a row per
  fit (test, sample, rows, climate coefficient mean / sd / 89% interval, LOO where comparable,
  divergences, maximum r_hat), then a short paragraph per test.
- One plot comparing the year effect in 1a with the shared year effect in 2b.
- Traces stay in the gitignored fits folder; don't commit them.
- Create or update `NOTES.md` at the repo root: current state, what these tests found, and
  open questions. Keep it under one page. Commit and push everything to
  `Jan-experiment-climate-module`.
