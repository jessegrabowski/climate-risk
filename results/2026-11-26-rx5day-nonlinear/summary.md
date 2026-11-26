# Does the rx5day response flatten? (task 26 Nov, part 1)

**Answer: no flattening that the data can show.** Neither curved version predicts better than the straight line.
Same set-up as A0 (7,710 country-years, hydro counts, ERA5 rx5day, population-weighted, z-scored within country).
Script: `experiments/hazard_frequency/nonlinear.py`. Plot: `response.png`.

**rx5day z observed:** minimum -3.05, 1st percentile -1.79, median -0.17, 99th percentile 2.83, maximum 4.97. Above 2: 309 rows.

| Fit | LOO against linear (± standard error) | Contribution at z = 2.8 | at z = 5.0 | Divergences | Max r_hat |
|---|---|---|---|---|---|
| Linear | - | 1.05 | 1.85 | 0 | 1.018 |
| Quadratic | -1.3 ± 1.1 | 0.97 | 1.53 | 0 | 1.035 |
| Piecewise (knots 0, 1, 2) | -3.1 ± 1.1 | 0.97 | 1.57 | 0 | 1.013 |

The curved fits sit below the line above z = 2.8; their 89% bands include it. Linear is 2.8 standard errors ahead of
piecewise but only 1.1 ahead of quadratic, so by the task's rule linear and piecewise are shown side by side.
LOO has 1 to 3 observations with Pareto k above 0.7 per fit.

## Illustrative multipliers, 2081-2100, rebuilt on the model's own link
Mean over a country's years of rate(z + dz) / rate(z), rate = softplus as fitted. Median (10th to 90th percentile) of 27 CMIP6 models. Everything else held fixed.

| Country | Scenario | dz (median) | Linear | Piecewise | 24 Nov value (wrong link) | Models with dz above 2.83 |
|---|---|---|---|---|---|---|
| LAO | SSP2-4.5 | 0.75 | 1.20 (1.02 to 1.43) | 1.21 (1.02 to 1.44) | 1.31 | 0% |
| LAO | SSP5-8.5 | 1.61 | 1.46 (1.14 to 1.78) | 1.47 (1.15 to 1.75) | 1.79 | 7% |
| NPL | SSP2-4.5 | 0.72 | 1.12 (1.01 to 1.29) | 1.13 (1.01 to 1.29) | 1.30 | 0% |
| NPL | SSP5-8.5 | 1.28 | 1.23 (1.12 to 1.73) | 1.23 (1.12 to 1.64) | 1.59 | 26% |
| ZMB | SSP2-4.5 | 0.73 | 1.23 (1.03 to 1.49) | 1.24 (1.03 to 1.50) | 1.30 | 0% |
| ZMB | SSP5-8.5 | 1.40 | 1.48 (1.06 to 1.91) | 1.49 (1.06 to 1.88) | 1.66 | 4% |
| BFA | SSP2-4.5 | 0.86 | 1.28 (0.96 to 1.52) | 1.29 (0.96 to 1.53) | 1.36 | 0% |
| BFA | SSP5-8.5 | 1.33 | 1.46 (0.88 to 3.17) | 1.47 (0.87 to 2.86) | 1.61 | 26% |
| CRI | SSP2-4.5 | 0.35 | 1.09 (0.93 to 1.29) | 1.09 (0.93 to 1.30) | 1.13 | 0% |
| CRI | SSP5-8.5 | 0.25 | 1.06 (0.78 to 1.33) | 1.07 (0.78 to 1.34) | 1.09 | 0% |
| MAR | SSP2-4.5 | -0.41 | 0.89 (0.83 to 1.03) | 0.89 (0.82 to 1.03) | 0.86 | 0% |
| MAR | SSP5-8.5 | -0.85 | 0.79 (0.70 to 0.98) | 0.78 (0.70 to 0.98) | 0.74 | 0% |

**The 24 Nov multipliers were wrong:** they used exp(0.36 x dz), but the model's link is softplus. For these six countries the new values are closer to 1. Use these instead.

**Beyond the observed range (no multiplier):** 12 of 1,592 country-scenario-windows, in 8 countries (CAF, COD, ERI, ETH, IND, PNG, SSD, TCD), all 2081-2100 under SSP3-7.0 or SSP5-8.5.
The flag uses the median model. Where some models exceed the limit (last column), the upper end of the range rests on extrapolation.

All countries: `multipliers_linear.csv`, `multipliers_piecewise.csv`, `multipliers_quadratic.csv` (with posterior 89% intervals).
dz comes from 24 Nov (ERA5 1991-2020 SD); the model's z uses each country's SD over its panel years.
