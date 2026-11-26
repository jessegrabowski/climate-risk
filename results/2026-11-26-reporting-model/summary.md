# A global temperature term with modelled reporting (task 26 Nov, part 4)

**Verdict: FAIL.** The pass rule holds on all years but not on 1990 onward. There the geophysical temperature slope is more than a third of the hydro slope (its interval does include 0). So no pathway table was made and nothing here should drive scenarios.

## Temperature slopes
Global mean surface temperature (GMST) is HadCRUT5 annual global anomaly, raw. Slopes are on the model's linear scale, per standard deviation (0.26 °C all years, 0.21 °C from 1990). 89% intervals.

| Model | Years | Hydro | Geophysical | Geo includes 0 | Geo under a third of hydro | Hydro excludes 0 |
|---|---|---|---|---|---|---|
| 4a | all | 0.38 (0.30 to 0.46) | 0.01 (-0.10 to 0.12) | yes | yes | yes |
| 4c (+ hazards) | all | 0.31 (0.23 to 0.39) | -0.01 (-0.12 to 0.10) | yes | yes | yes |
| 4a | 1990+ | 0.19 (0.12 to 0.27) | -0.07 (-0.17 to 0.04) | yes | **no** (0.07, limit 0.06) | yes |
| 4c (+ hazards) | 1990+ | 0.11 (0.03 to 0.19) | -0.08 (-0.19 to 0.02) | yes | **no** (0.08, limit 0.04) | yes |

The hydro slope also shrinks a lot from 1990: per °C, 1.46 to 0.91 (4a) and 1.18 to 0.53 (4c). It depends on the years used.

## Prediction against the year-effect model (same rows)
LOO difference, reporting model minus year-effect model (negative = predicts worse): 4a all -31 ± 12; 4c all -31 ± 11; 4a 1990+ -23 ± 9; 4c 1990+ -21 ± 8.

## Reporting terms did not behave like detection
Internet use and mobile subscriptions have **negative** slopes (-0.08 to -0.22 per standard deviation; one interval, internet in 4c 1990+, includes 0). More connectivity should mean more events recorded. Only the year-2000 step is positive (0.44 to 0.68). These covariates are probably standing in for other differences, so reporting drift is not convincingly modelled.

## Other results in 4c
rx5day: hydro 0.45 to 0.47, geophysical -0.14 to -0.15 (interval excludes 0; same unexplained placebo result as before). Cyclone share: hydro 0.35 to 0.37, geophysical includes 0.

## Temperature product: a choice you should know about
The task said to use Test 1's temperature source. That series is NOAA **ocean heat content**, not surface temperature, so I used HadCRUT5 as the main run. With ocean heat in 4a the rule holds in both samples (1990+: hydro 0.24, geophysical -0.04). 4c was not run with ocean heat, so that is not a full pass, and I have not switched the verdict to it.

## Data and checks
- All years: 6,789 of 7,710 rows used (hydro 9,032 of 9,530 events). 1990+: 5,476 of 6,365 rows. Losses come from blank internet (814 rows) and mobile (225) values; 532 lost rows are 1991-1995, where the World Bank gives a 1990 value and then blanks, left blank as the task specified.
- 10 fits. No divergences except 1 in one year-effect comparison. Highest r_hat 1.05 (population coefficient, 4a all), so that coefficient is imprecise; temperature slopes r_hat 1.01 or less, effective samples 315 or more in the four fits checked. 2 to 5 rows per fit have Pareto k above 0.7.
- Tried: 4a, 4c, the year-effect comparisons, and the ocean heat check. No priors or covariates were tuned. Optional 4d (storm anchor) was not built.

Tables: `slopes.csv`, `verdict.csv`, `loo_vs_year_effect.csv`, `models.csv`, `rows.csv`. Script: `experiments/reporting_model/fit.py`.
