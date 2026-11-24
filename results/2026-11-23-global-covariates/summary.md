# Global covariates for estimation (2026-11-23/24) — INTERIM: GloFAS not finished

Branch `Jan-experiment-climate-module`. No model code changed, nothing refitted. Data is git-ignored and not committed.

## What exists now
Yearly global grids (native grid of each source, 1981-2025) in `data/processed/covariates/global/`, then:
- `data/processed/covariates/global_country_year.parquet` — 263 GADM countries x 45 years (all 202 panel countries present). Keys `ISO` and `date` (1 Jan) as in the panel, plus `year`.
- `data/processed/covariates/global_admin1_year.parquet` — 3,691 admin-1 units x 45 years.
- `data/processed/covariates/manifest.json` — one row per source (dataset, version, product, URL, access date 2026-11-24, licence).

Every statistic is given as `_mean` (area-weighted over valid pixels), `_max` (largest pixel in the country) and, for ERA5 and ERA5-Land fields, `_popmean` (GHS-POP weighted, nearest epoch; years before 1990 use the 1990 grid).

| Column | Source | Notes |
|---|---|---|
| `rx5day`, `precip_era5`, `precip_anom_std` | ERA5 daily total precipitation (`derived-era5-single-levels-daily-statistics`, daily_sum, UTC) | rx5day = largest 5-day sum inside the year; anomaly = z-score of the annual total against 1991-2020 per pixel |
| `chirps_rx5day`, `chirps_precip` | CHIRPS **v2.0** (IRI mirror), 50S-50N | box-averaged to 0.25 deg on the IRI server, so rx5day is coarser than at 0.05 deg |
| `soilm_mean`, `soilm_max_month`, `t2m_mean` (deg C), `precip_land`, `pet_annual` | ERA5-Land monthly means | monthly means: max is the wettest month, not a daily maximum. `pet_annual` positive mm |
| `roni_annual_mean`, `roni_peak` | CPC RONI, file of 2026-11-22 | by year, same for every country |
| `q_annual_max`, `q_days_over_rp5` | GloFAS v5.0 | **NOT YET BUILT**, see below |

## GloFAS — pending
Feasibility passed (one global year: 2.5 GB, 112 min including queue). The full run is 46 single-year requests in one queue, about 2 h or more each, so about 4 days. Running as `experiments/covariates/global_annual_stats.py glofas`; the year-1980 request had been queued 8 h at the time of writing. Raw GloFAS years are KEPT (about 115 GB) until all years exist, because the 5-year return level needs every annual maximum before days over it can be counted; `glofas-days` then fits the level and deletes the raw files. Method: Gumbel distribution by L-moments per river pixel (beta = L2/ln 2, mu = L1 - gamma*beta, level = mu - beta*ln(-ln(1-1/5))), fitted on the 1980-2025 annual maxima, pixels with fewer than 10 valid years left empty. Then re-run `global_country_year.py` and `finalize_global.py`.
Nepal's per-country GloFAS (1980-1982 only) is superseded and was stopped.

## Checks (`checks.txt`, `map_rx5day_mean.png`)
- Nulls: ERA5 columns 0% by country; popmean 3.4% (countries with no population in a pixel); CHIRPS 10.6%; ERA5-Land 12.2% (13.7% popmean). Null share is identical in every year, so these are coverage gaps, not missing years.
- Countries with no value in a source: CHIRPS 28 (north of 50N: BLR, DNK, EST, FIN, FRO, GRL, IRL, ISL, LTU, LVA, NLD, NOR, SWE, ... plus uninhabited territories); ERA5-Land 32 (small islands without a land pixel: e.g. MDV, MHL, KIR, TUV, SYC, TON, BMU). All 202 panel countries are in the table; some of those panel countries have empty ERA5-Land or CHIRPS columns as listed in `checks.txt`.
- CHIRPS v2 vs ERA5 annual totals (country means): pooled country-years r = 0.8376 (n = 10,575); across country averages r = 0.8491 (n = 235). Reported, not judged. Single-country values differ a lot (2010: CRI 4,898 vs 3,408 mm, NPL 2,137 vs 1,408 mm).
- ERA5-Land `precip_land` agrees with ERA5 `precip_era5` to within 1% for CRI, LAO, MAR, NPL, ZMB (2010), which is a unit check on the monthly-mean conversion.
- `pet_annual` looks high in the tropics (LAO 5,437 mm, ZMB 4,654 mm in 2010) although the global land mean is plausible; ERA5-Land potential evaporation is known to run high. Treat as an input to SPEI-type indices only after rescaling or a bias check.
- Map: global mean rx5day from ERA5 looks right (ITCZ, monsoon Asia, Andes/Himalaya, Pacific North-West coast bands).

## Not done / differences from the task
- **Point table (`global_points.parquet`) not built.** CLAUDE.md says the synthetic non-disaster points were deleted from the repo, and Geo-Disasters holds admin-unit polygons, not points (GAUL licence is non-commercial). There is nothing to reuse, and I was told not to regenerate. `climate_risk/data/point_extract.py` (tested) reads every statistic at any (longitude, latitude, year) table once points exist.
- Population weighting used GHS-POP (in the repo); outlines are GADM simplified by about 200 m. Pixel overlap uses exactextract coverage fractions times cos(latitude).
- Not committed: raw or processed data. Not pushed.
- Disk: raw about 4.2 GB now (global 2.4 GB incl. the 1990 GloFAS year; country downloads NPL 0.3, ZMB 0.8, LAO 0.5, MAR 0.2, CRI 0.1 GB); processed 3.0 GB.

## CHIRPS v2 vs v3 (labelled v2.0, IRI mirror, everywhere)
v2.0 is a stand-in. Differences from the target v3 daily "rnl": v3 applies a gauge undercatch adjustment, and its daily values are pentad totals split into days with ERA5, so sub-pentad timing in v3 comes from ERA5. v2.0 daily improved is a different product; its totals will differ. The UCSB v3 host was throttled (HTTP 429) and then unreachable; it was not retried. The CHIRPS loader is separate from the aggregation code (`climate_risk/data/chirps.py`, `global_stats.chirps_global_stats`), so v3 can replace it by swapping that one function.

## Country grids (NPL, ZMB, CRI, LAO, MAR) — raw downloads only for application later
CHIRPS v2 (0.05 deg, box clips), ERA5-Land monthly and RONI are on disk for all five; ERA5 monthly for NPL and ZMB; CRI/LAO/MAR ERA5 was still queued. Per-country aggregation onto the model grid (`build_cell_year_table.py`) exists and was run for all five without GloFAS; **Laos used a World Bank outline only to cut its download box** (the HDX boundary file returned 404); the Laos grid itself comes from the place config. BFA (Burkina Faso) has no raw country downloads yet. No country is missing a place config among the five.
Tropical cyclones are skipped (IBTrACS/CLIMADA); **Laos and Costa Rica will need them later**.

## Scripts in plain words (`experiments/covariates/`)
- `global_annual_stats.py` — the main downloader. For each year it asks Copernicus (or the IRI for CHIRPS) for a whole-world file, boils it down to a few yearly maps (heaviest 5-day rain, annual total, soil moisture, ...), saves those small maps, and deletes the big file. It can be stopped and restarted; finished years are skipped.
- `global_boundaries.py` — cuts the GADM file into one shape per country and one per first-level region.
- `global_country_year.py` — lays those shapes over each yearly map and writes the average and maximum (and population-weighted average) per country and year.
- `finalize_global.py` — adds RONI, the `ISO` and `date` keys and writes the manifest.
- `check_global.py` — null counts, gaps, the CHIRPS-vs-ERA5 correlation and the map.
- `feasibility_global.py` — the one-year test requests.
- `download_covariates.py`, `build_cell_year_table.py`, `check_cell_year_table.py` — the earlier five-country versions (box downloads, then the model-grid table).
- Library code in `climate_risk/data/`: `copernicus.py` (requests), `chirps.py`, `roni.py`, `global_stats.py` (yearly maps), `global_population.py` (population weights), `covariate_table.py` (area weights, Gumbel fit), `point_extract.py`.
