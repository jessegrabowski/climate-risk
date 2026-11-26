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

## 2026-11-24 — global covariates (Jan-experiment-climate-module)
- Built global yearly covariate grids 1981-2025 (ERA5 daily rx5day/total/anomaly, ERA5-Land soil moisture/temperature/PET, CHIRPS v2.0 IRI 0.25 deg, RONI) and country and admin-1 year tables
  (`data/processed/covariates/global_country_year.parquet`, mean/max/popmean). Details and caveats: `results/2026-11-23-global-covariates/summary.md`.
- GloFAS v5.0 global (q_annual_max, q_days_over_rp5) is still running, about 4 days; then re-run `global_country_year.py` and `finalize_global.py`.
- Point table not built: synthetic non-disaster points no longer exist in the repo; extractor `point_extract.py` is ready.
- CHIRPS is v2.0 as a stand-in, not v3.

## 2026-11-24 (later) — hazard covariates in the frequency model, and the resampler
- `climate_risk/data/resample.py` (`atom_year_table`): yearly global grids onto any shapes (10 km lattice cells or GADM units), area-weighted and population-weighted means. On the Laos 10 km lattice the area-weighted mean reproduces the LAO country row to 0.014% (limit 1%). ADM2 tables built for LAO, NPL, CRI, ZMB, MAR, BFA (`data/processed/covariates/adm2_year_<ISO>.parquet`, not in git). The `sea` region config exists (`climate_risk/config/regions/sea.toml`).
- Frequency model with within-country z-scored hazard covariates in place of `co2`: `experiments/hazard_frequency/`, results in `results/2026-11-24-hazard-frequency/summary.md`. rx5day slope 0.36 (89% interval 0.31 to 0.41); LOO gain over the matching base 124 to 138 (dse about 17) on hydro, 0.09 (dse 3.2) on the geophysical placebo. Regional RONI slopes are not reliable (placebo has 5 divergences). H4 (GloFAS) still to add.
- The synthetic points stay retired (`global_points.parquet` dropped from the plan); `point_extract.py` kept, not built on.
- Fixed 13 mypy errors in the 01d2487 data modules (types only, one `cdsapi` stub override in `pyproject.toml`). Before this `pixi run typecheck` did not pass.
- GloFAS download now runs as two workers (ascending and descending, with claim files); the per-country ERA5 jobs were cancelled.

## 2026-11-26 — rx5day robustness, CMIP6 link, space-time areal model (task 24b)
- Robustness (`results/2026-11-24-rx5day-robustness/summary.md`): rx5day slope 0.36 holds with country trends (0.35) and detrending (0.34); next year's rx5day -0.01; CHIRPS 0.22 against ERA5 0.40 (area means, same rows); geophysical placebo -0.11 survives, no predictive gain.
- CMIP6 (`results/2026-11-24-cmip6-rx5day/summary.md`): country changes in annual rx5day, 27 models, and illustrative rate multipliers exp(0.36 x dz). Large dz (3 or more) is extrapolation.
- Space-time areal model (`results/2026-11-24-spatial-hazard/summary.md`): notebook 12's Laos river slope reproduces (-1.01). On ADM2 x year, Southeast Asia: rx5day slope 0.22 (0.17 to 0.27), LOO +18 (dse 6) over the same model without it. River slope differs in sign from the notebook; unexplained.
- The robustness script holds one saved fit in memory at a time; holding all of them ran this 12 GB WSL machine out of memory.
- GloFAS downloads: `experiments/covariates/glofas_workers.sh` restarts the two workers after a reboot; `glofas_fetch_finished.py` collects requests that finished while no worker was waiting. H4 still to do.
