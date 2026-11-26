# Robustness of the rx5day frequency result (task 24b, part A)

Base set-up (H1): published covariates without `co2`, plus the precipitation anomaly and rx5day (ERA5, population-weighted), z-scored within country, shared year effect. Hydro counts unless marked. All rows use the same 7,710 country-years (CHIRPS rows: 7,281). 4 chains x 2000 draws.
"Gain" is the PSIS-LOO elpd gain over a fit with the same rows and every term except rx5day (± its standard error). Table: `robustness.csv`; script: `experiments/hazard_frequency/robustness.py`.

| Check | rx5day variable | Slope (89% interval) | Gain | Divergences |
|---|---|---|---|---|
| A0 H1 as before | popmean | 0.36 (0.32 to 0.41) | 69 ± 12 | 0 |
| A1 + linear trend per country | popmean | 0.35 (0.30 to 0.40) | 58 ± 11 | 0 |
| A2 detrended within country | popmean | 0.34 (0.29 to 0.39) | 64 ± 12 | 0 |
| A3 next year's rx5day (falsification) | popmean, lead | -0.01 (-0.05 to 0.03) | -0.1 ± 0.6 | 0 |
| A4 ERA5, CHIRPS rows | popmean | 0.36 (0.31 to 0.41) | 63 ± 11 | 0 |
| A4 ERA5, CHIRPS rows | area mean | 0.40 (0.35 to 0.45) | 81 ± 14 | 0 |
| A4 CHIRPS v2.0, same rows | area mean | 0.22 (0.18 to 0.27) | 29 ± 8 | 0 |
| A5 ERA5 | area mean | 0.40 (0.35 to 0.45) | 89 ± 15 | 0 |
| A5 ERA5 | country maximum | 0.20 (0.16 to 0.24) | 30 ± 8 | 0 |
| A6 geophysical placebo, A0 | popmean | -0.11 (-0.21 to -0.02) | 1.2 ± 1.8 | 7 |
| A6 placebo, detrended | detrended | -0.12 (-0.21 to -0.03) | -0.1 ± 2.2 | 3 |
| A6 placebo, country trends | popmean | -0.12 (-0.21 to -0.02) | 0.5 ± 1.9 | 0 |

Max r_hat 1.004 to 1.030.

## Plain reading
- **The slope holds** with country trends (0.35) and detrending (0.34), against 0.36. So it is not two series trending together.
- **Next year's rain predicts nothing** (-0.01, interval includes 0), as it should.
- **The size depends on the data and the aggregation.** CHIRPS gives 0.22 where ERA5 gives 0.40 on the same rows (area means); the intervals do not overlap. The country maximum gives 0.20. All are above 0. The paper should name the product and aggregation with the number. The area mean fits slightly better than the population-weighted mean (89 against 69); I did not test whether that difference is larger than its noise.
- **Placebo:** the small negative slope on geophysical counts (about -0.11) survives detrending and country trends, and its interval stays just below 0. It adds no predictive skill (gains within 1 standard error of 0). I have no explanation for it. Its size is about a third of the hydro slope, with the opposite sign.
- The gain of 69 is smaller than the 124 reported on 24 Nov because this base already contains the precipitation anomaly.
