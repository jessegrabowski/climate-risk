# Is CO2 identified in the Stage 2 frequency model? Results of the four identification tests

Branch `Jan-experiment-climate-module`. All fits: 4 chains, 2000 draws, 1000 tuning, seed 1, `target_accept` 0.95.
Coefficients are per standard deviation of the covariate; intervals are 89% highest-density intervals (HDI).
Source table: `fits.csv`; LOO tables: `loo_time_driver.csv`, `loo_geophysical_control.csv`.

## Verdict

**Mixed.** Under the task's decision rule (all three conditions required) CO2 is not shown to be identified.

| Condition | Result |
|---|---|
| 1. Test 2: interval for hydro `co2` (2b) and for hydro minus geophysical (2c) excludes 0 | **Pass under one shared year effect** (2b, 2c). **Fails with separate year effects** (2d). |
| 2. Tests 3 and 4 without a year effect: same sign, interval excludes 0 | **Fail.** Test 3 is negative (excludes 0), Test 4 is about 0. |
| 3. Adding the year effect in Tests 3 and 4 keeps the interval off 0 | **Fail.** Both intervals cover 0. |

The strongest evidence for a CO2 effect is Test 2 under a shared year effect (2b, 2c). It depends on two assumptions:
that geophysical events have no CO2 pathway, and that EM-DAT reporting drift moves hydro and geophysical counts by the
same amount. The separate-year-effect fit (2d), which drops the second assumption, finds no difference between classes.
The two readings are given side by side below. LOO does not separate them.

## Test 2 `test2_geophysical_control`: does CO2 act on hydro more than on geophysical events?

Geophysical events (earthquakes, volcanic activity, dry mass movements) share EM-DAT's reporting history but have no
climate pathway. 7,710 country-years in every fit; 9,530 hydro and 1,243 geophysical events.

| Fit | Coefficient | Mean ± sd | 89% HDI | Divergences | Max r_hat |
|---|---|---|---|---|---|
| **2b** joint, shared year effect, `co2` on hydro only | hydro `co2` | 0.47 ± 0.06 | 0.37 to 0.57 | 0 | 1.016 |
| **2c** joint, shared year effect, `co2` free | hydro `co2` | 0.99 ± 0.31 | 0.53 to 1.54 | 0 | 1.029 |
| 2c | geophysical `co2` | 0.54 ± 0.32 | 0.05 to 1.07 | | |
| 2c | hydro minus geophysical | 0.45 ± 0.06 | 0.35 to 0.55 | | |
| **2d** joint, separate year effects, `co2` free | hydro `co2` | 0.45 ± 0.34 | -0.13 to 0.98 | 0 | 1.015 |
| 2d | geophysical `co2` | 0.50 ± 0.39 | -0.14 to 1.07 | | |
| 2d | hydro minus geophysical | -0.05 ± 0.52 | -0.89 to 0.75 | | |
| 2a geophysical alone (reused earlier fit) | geophysical `co2` | 0.55 ± 0.40 | -0.10 to 1.14 | 9 | 1.009 |

PSIS-LOO (same 7,710 rows, both classes): 2d -10324.9 (best), 2c -10331.3 (diff -6.4, dse 5.5),
2b -10332.3 (diff -7.4, dse 5.3). Five rows with Pareto k above 0.7 in each fit. The differences are within about
1.4 standard errors, so LOO does not decide between the shared and separate year effects.

**Reading with one shared year effect (2b, 2c).** Hydro `co2` is positive and its interval excludes 0 (2b: 0.37 to
0.57; 2c: 0.53 to 1.54), near the published Stage 2 value of 0.32. The hydro-minus-geophysical difference is 0.45 and
excludes 0. The shared year effect absorbs reporting drift common to both classes, leaving the class difference, which is
why the difference is estimated tightly. 2b's narrow interval also owes to fixing the geophysical slope at 0.

**Reading with separate year effects (2d).** Each class carries its own reporting history. Hydro and geophysical slopes
are about 0.5 each and the difference is centred on 0 (-0.89 to 0.75). Geophysical events, which have no CO2 pathway,
get a slope of about 0.5 in 2a, 2c and 2d. So either hydro and geophysical reporting differ over time (and the shared
walk of 2b and 2c is too restrictive) or the ridge between `co2` and the year effect is carrying the slope.

Plot: `year_effect_1a_vs_2b.png`. The two year effects have the same hump (rising through 2005, falling after 2010);
the joint model's is much tighter.

## Test 1 `test1_time_driver`: does any common time driver improve prediction once a year effect is in?

Same 7,710 rows in every fit. Year-effect model, 9,530 events.

| Fit | Climate term | Mean ± sd | 89% HDI | LOO elpd | diff (dse) |
|---|---|---|---|---|---|
| 1c Hamilton shock (step prior 0.3) | shock | -0.067 ± 0.042 | -0.135 to 0.000 | -8152.7 | 0 |
| 1b no `co2` | none | n/a | n/a | -8153.1 | -0.4 (1.2) |
| 1a `co2` | `co2` | 0.52 ± 0.34 | -0.01 to 1.07 | -8153.5 | -0.8 (1.2) |
| 1c' STL deviation | deviation | -0.043 ± 0.034 | -0.095 to 0.013 | -8153.9 | -1.3 (0.7) |
| 1c, step prior 0.1 | shock | -0.066 ± 0.039 | -0.127 to -0.004 | not compared | |
| 1c, step prior 0.6 | shock | -0.068 ± 0.042 | -0.141 to -0.005 | not compared | |

Removing `co2` costs nothing in prediction, and no time driver beats the others by more than the noise. The Hamilton
temperature shock is small and negative, stable across the year-effect prior (moves by about 0.003). Divergences 0;
max r_hat 1.026.

## Test 3 `test3_high_impact`: only events with at least 100 deaths (686 events, 7,710 rows)

| Fit | `co2` mean ± sd | 89% HDI |
|---|---|---|
| plain (no year effect) | -0.27 ± 0.07 | -0.38 to -0.16 |
| with year effect | 0.58 ± 0.50 | -0.27 to 1.29 |

The plain coefficient is negative, the opposite sign to the published one. I have not checked whether this is falling
deaths per event over time. The year-effect fit does not identify it.

## Test 4 `test4_post_2000`: years 2000 onward only (4,608 rows, 6,764 events)

| Fit | `co2` mean ± sd | 89% HDI |
|---|---|---|
| plain (no year effect) | -0.01 ± 0.04 | -0.06 to 0.05 |
| with year effect | 0.06 ± 0.32 | -0.46 to 0.52 |

Without the earlier, steeper reporting growth there is no `co2` signal either way.

## Caveats

- Test 2's positive result rests on assumptions the data only partly test (above). LOO prefers separate year effects
  (not decisively).
- The geophysical `co2` slope of about 0.5 in the placebo says a time trend is being picked up by `co2` even where no
  climate pathway exists.
- Tests 3 and 4 shift both the sample and the target at once, so a sign change between them is not purely a time-driver effect.
- arviz raised "invalid value in scalar divide" warnings while computing r_hat; the r_hat figures are maxima over the
  parameters that computed.
- The first run of Test 2 reported wrong joint-model coefficients (a labelling problem in `az.summary` for the
  two-dimensional `beta`); they are now computed straight from the draws. Single-coefficient fits were cross-checked
  against the raw posterior and match.
