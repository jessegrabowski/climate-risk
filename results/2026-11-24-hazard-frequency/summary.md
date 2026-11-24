# Frequency model with local hazard covariates instead of CO2 (24 Nov 2026)

Branch `Jan-experiment-climate-module`. Base = the year-effect model without `co2` (fit 1b); same priors, 4 chains x 2000 draws, seed 1, `target_accept` 0.95.
Covariates are z-scored within each country (so a slope means "one standard deviation wetter than that country's usual year") and joined from `global_country_year.parquet` on `ISO` + `date`.
Numbers are posterior mean with the 89% interval. Tables: `coefficients.csv`, `loo_vs_base.csv`, `row_counts.json`. Script: `experiments/hazard_frequency/run.py`.

## Rows
Base panel 7,710 country-years (199 countries). Soil moisture (ERA5-Land) is missing for 346 of them (small islands, e.g. MHL, MDV, TUV, KIR), so H2, H3, the placebo and post-2000 fits use 7,364 (H3-post2000: 4,385). Each hazard fit is compared with a base fit refitted on exactly its rows.

## Hydro counts
| Fit | Slope | Mean (89% interval) |
|---|---|---|
| H1 | precip anomaly (popmean) | 0.083 (0.028 to 0.143) |
| H1 | rx5day (popmean) | 0.363 (0.315 to 0.414) |
| H2 | precip anomaly | 0.144 (0.058 to 0.236) |
| H2 | rx5day | 0.360 (0.309 to 0.411) |
| H2 | soil moisture | -0.066 (-0.139 to 0.011) |
| H3 | rx5day | 0.354 (0.300 to 0.404) |
| H3 | RONI peak, regional slopes: Africa / Americas / Asia / Europe / Oceania | -0.151 (-0.241 to -0.059) / -0.046 (-0.143 to 0.050) / 0.117 (0.015 to 0.223) / 0.011 (-0.095 to 0.106) / 0.370 (0.184 to 0.556) |
| H3-post2000 | rx5day | 0.369 (0.301 to 0.437) |
| H3-post2000 | precip anomaly / soil moisture | 0.186 (0.069 to 0.309) / 0.018 (-0.079 to 0.120) |
| H3-post2000 | RONI regional slopes: all five | every interval includes 0 (e.g. Oceania 0.161, -0.053 to 0.378) |

H3 also has precip anomaly 0.141 (0.053 to 0.229) and soil moisture -0.063 (-0.135 to 0.015).

## Fit against the base (PSIS-LOO, same rows; higher elpd is better)
| Fit | elpd gain vs base | dse |
|---|---|---|
| H1 vs 1b | 124.4 | 16.3 |
| H2 vs base | 123.8 | 16.6 |
| H3 vs base | 137.5 | 17.4 |
| H3-post2000 vs base | 105.4 | 15.1 |
| Placebo vs base | 0.09 | 3.2 |

H2 and H3 are on the same rows, so H3 minus H2 is 13.8 (I did not compute its dse). Divergences: 0 in H1, H2, H3, H3-post2000. Max r_hat 1.007 to 1.021. Pareto k above 0.7: 1 to 3 rows per fit.

## Placebo (H3 on geophysical counts, 1,238 events)
Hazard slopes should be near 0. Precip anomaly -0.037 (-0.199 to 0.132) and soil moisture 0.022 (-0.120 to 0.152) are. rx5day is -0.115 (-0.214 to -0.007), a small negative whose interval just excludes 0. RONI slopes are 0.07 to 0.12 in every region; for Americas (0.117, 0.013 to 0.225) and Asia (0.109, 0.003 to 0.208) the interval excludes 0. **5 divergences**, so treat this fit as less reliable than the others.

## Plain reading
- Daily-rain extremes carry the signal: the rx5day slope is 0.36 and its interval is about 0.1 wide, in every hydro fit including post-2000, and is negative (-0.115) on the placebo. The hydro-vs-placebo gap in rx5day is far larger than the placebo's own interval.
- Soil moisture adds nothing visible: its interval includes 0 in H2, H3 and post-2000.
- RONI: `roni_peak` is the same number for every country in a year, so a shared RONI slope cannot be told apart from the year effect (shared mean 0.062, -0.144 to 0.282). Only differences between regions are identified. Africa is negative and Oceania positive in H3, but the placebo gives Asia the same slope as hydro (0.109 vs 0.117), the post-2000 fit shows no regional slope, and the placebo has divergences. I would not read the regional RONI pattern as a hydro effect yet.
- LOO gains of 105 to 138 against about 16 standard errors, versus 0.09 on the placebo, say the local rain covariates predict hydro counts and not geophysical ones. This does not by itself show that the gain is climate rather than, say, population-weighted rain tracking reporting.

## Not done
H4 (`q_days_over_rp5_popmean`) waits for GloFAS. Slopes are per within-country SD before a second, global standardisation inside the model, so they are close to but not exactly per SD.
