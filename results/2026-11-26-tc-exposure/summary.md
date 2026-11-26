# Tropical-cyclone wind exposure (task 26 Nov, part 2)

**The expected pattern holds: cyclone exposure predicts storm counts, not flood counts.** rx5day predicts both.

## Data built
- Tracks: IBTrACS v04r01, all basins (NetCDF, accessed 26 Nov 2026; may be used and redistributed without restriction).
- Winds: CLIMADA 6.2.0 (own environment `climada_env`), Holland 2008 model, 1-minute winds, hourly track steps, up to 300 km from the track and 61 degrees latitude. Missing pressure or wind filled by CLIMADA. 22 storms of 1980-2025 dropped (no valid values or one time step).
- Grid: ERA5 0.25 degrees, the 161,278 cells with population. Each year's value is the cell's strongest wind; a storm counts in the year it starts.
- Per country-year and per Southeast Asia district-year, GHS-POP weighted: mean annual maximum wind, share of people at 17.5 m/s (34 knots) or more, share at 33 m/s (64 knots) or more.
- Not checked against observed station winds or another wind model.

## Frequency fits
A0 set-up (7,710 country-years, year effect) plus the share of people at 34 knots or more. Not z-scored within country. 1,387 rows and 95 of 199 countries have any exposure. Slopes are per standard deviation (a standard deviation is a share of 0.19). 89% intervals.

| Counts (events) | rx5day slope without cyclones | rx5day slope with | Cyclone share slope | LOO gain from cyclones (± s.e.) | Divergences |
|---|---|---|---|---|---|
| Hydro (9,530) | 0.36 (0.32 to 0.41) | 0.32 (0.27 to 0.37) | 0.40 (0.35 to 0.45) | 72 ± 13 | 0 |
| Flood + wet mass movement (5,810) | 0.28 (0.23 to 0.34) | 0.29 (0.24 to 0.34) | -0.09 (-0.16 to -0.01) | 1.9 ± 2.3 | 0 |
| Storm (3,720) | 0.26 (0.20 to 0.32) | 0.18 (0.11 to 0.24) | 0.50 (0.44 to 0.55) | 112 ± 15 | 0 |
| Geophysical, placebo (1,243) | -0.11 (-0.21 to -0.02) | -0.11 (-0.20 to -0.01) | -0.11 (-0.22 to 0.02) | -0.6 ± 2.4 | 3 |

Max r_hat 1.02. Pareto k above 0.7: 4 rows in the storm and geophysical fits, none in the others.

## Plain reading
- **Storms:** adding exposure improves prediction by 112 (± 15). About a third of the storm rx5day slope (0.26 to 0.18) was cyclone rain.
- **Floods:** the small negative cyclone slope adds no predictive skill (1.9 ± 2.3).
- **Placebo:** the cyclone slope includes 0. The negative rx5day slope on geophysical counts is still there (-0.11), as on 24 Nov, and still unexplained.

Tables: `coefficients.csv`, `loo.csv`, `exposure_description.csv`. Scripts: `experiments/tropical_cyclones/` (`cells.py`, `windfields.py`, `exposure.py`) and `experiments/hazard_frequency/cyclones.py`.
