# Spatial model with field, offsets and hazard split, Southeast Asia (task 26 Nov, part 3)

**rx5day keeps a positive slope; the river slope does not return to negative; the geophysical placebo is not clean.** Treat the spatial results as indicative.

## Set-up
District (ADM2) by year, 1990-2023, 9 countries. 24 November model (M2) plus a smooth spatial field (HSGP, notebook 12 settings) and ADM1 and ADM2 offsets; country offsets kept. rx5day is population-weighted and z-scored within each **district** over time. Cyclone share is scaled over all district-years (one standard deviation = a share of 0.38). Slopes are on the log event rate; 89% intervals.

## 3a: all weather, water and climate events, with placebo
| | M2 (24 Nov) | M4 (field + offsets) | M4 on geophysical events (placebo) |
|---|---|---|---|
| Events | 1,294 | 1,294 | 203 |
| rx5day slope | 0.22 (0.17 to 0.27) | **0.21 (0.16 to 0.26)** | **-0.19 (-0.37 to -0.02)**, expected near 0 |
| River distance slope | 0.14 (0.06 to 0.21) | **0.30 (0.12 to 0.50)** | 0.61 (0.24 to 0.98) |
| Divergences (of 4,000) | 6 | 0 | 0 |

- **LOO, M4 against M2 on the same 1,260 records:** M4 better by 333 ± 30 (record terms only, not the total count). 2 to 4 M4 records have Pareto k above 0.7 (the count varies slightly between calculations).
- **River:** positive, so farther from major rivers means more recorded events. Geophysical events show it too, so it probably reflects something other than flood risk (not tested).
- **Placebo:** the rx5day interval excludes 0 on the negative side. The country-level placebo was also negative. Unexplained.

## 3b: floods (with wet mass movements) and storms, one joint model
795 floods, 441 storms; shared detection; each hazard has its own other terms.

| Slope | Flood rate | Storm rate |
|---|---|---|
| rx5day | 0.19 (0.12 to 0.25) | 0.20 (0.11 to 0.29) |
| Cyclone share (34 knots or more) | 0.10 (-0.03 to 0.22) | **0.46 (0.36 to 0.57)** |
| River distance | 0.27 (0.05 to 0.47) | 0.38 (0.05 to 0.70) |

Cyclone exposure goes with more storm records, not clearly more flood records, matching the country-level result. rx5day goes with both. I fitted only the version with all four rainfall and cyclone terms, not one with just the two assigned terms. 1 divergence in 4,000.

## Does a finer field change the results? (added after the first runs)
The first settings (M4 and M5: boundary factor 4, 50 basis functions per axis) resolve variation down to about 720 km, coarser than the flood field fitted in M5 (about 390 km). Two reruns:

| | M4 | M4 finer | M5 floods | Flood-only finer |
|---|---|---|---|---|
| Resolves down to about | 720 km | 270 km | 720 km | 110 km |
| Field lengthscale, km | 510 (190 to 820) | 570 (200 to 910) | 390 (180 to 570) | 350 (180 to 500) |
| rx5day slope | 0.21 (0.16 to 0.26) | 0.21 (0.16 to 0.26) | 0.19 (0.12 to 0.25) | 0.19 (0.12 to 0.25) |
| River distance slope | 0.30 (0.12 to 0.50) | 0.31 (0.13 to 0.50) | 0.27 (0.05 to 0.47) | 0.28 (0.08 to 0.48) |
| Cyclone share slope | not in model | not in model | 0.10 (-0.03 to 0.22) | 0.10 (-0.02 to 0.23) |
| Divergences | 0 | 0 | 1 (joint) | 0 |
| LOO difference, finer minus original | | -1.4 ± 0.9 | | +10 ± 5 |

- **Slopes do not change.** In the flood-only fit the field lengthscale is well inside what the settings resolve.
- The flood prediction gain is small (about 2 standard errors) and also mixes two changes: a finer field, and dropping storms from the shared reporting terms.
- **A finer field is not finer data.** Half the flood records name districts (typically about 70 km across), nearly half name provinces (about 230 km). This does not map river valleys or flood footprints.
- The finer flood fit (r_hat at most 1.01, fewest effective draws 229) took 3 h 17 min; the 270 km run took 57 min.

## Caveats
- **Mixing (first runs):** rainfall, cyclone and river slopes are good (r_hat 1.01 or less). Some M5 background terms are weaker (highest r_hat 1.04, fewest effective draws 124, flood intercept); 1,323 of 23,393 checked quantities have r_hat above 1.01.
- Not checked: posterior predictive checks, sensitivity to the detection prior, and any link between the placebo result and reporting.

Run times: M4 56 min, placebo 40 min, joint model 2 h 3 min. Scripts: `experiments/spatial_hazard/space_time_v2.py`, `compare_v2.py`. Numbers: `M4.json`, `M4_placebo.json`, `M5.json`, `M4_c15.json`, `M5_flood_c12_m100.json`, `loo_comparisons.csv`.
