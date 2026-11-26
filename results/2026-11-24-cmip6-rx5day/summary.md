# From warming to rx5day: CMIP6 country changes (task 24b, part B)

Source: Copernicus CDS `sis-extreme-indices-cmip6`, annual rx5day (`maximum_5_day_precipitation`), one member per model, newest dataset version. 27 models (26 SSP1-2.6, 25 SSP2-4.5, 21 SSP3-7.0, 27 SSP5-8.5). 202 panel countries, population-weighted (GHS-POP 2020) after linear interpolation to the ERA5 0.25° lattice. Changes are 2041-2060 and 2081-2100 against 1995-2014.
Files: `country_deltas.parquet` (not in git: per model, country, scenario, window: change in mean and SD), `multipliers.csv`. Scripts: `experiments/cmip6/`.

## Step to the frequency model
dz = (ERA5 1991-2020 mean rx5day × model's relative change in the mean) / ERA5 1991-2020 SD, per country. The change in SD is reported in the files but not used in dz.

## ILLUSTRATIVE event-rate multiplier, exp(0.36 × dz), everything else held fixed
Median across models (10th to 90th percentile). It is not a forecast: population, income, reporting, adaptation and the year effect are frozen, the slope's own uncertainty (0.315 to 0.414) is not included, and the model's rate link is softplus, so exp() is only the approximation the task asked for.

| | SSP2-4.5, 2041-2060 | SSP2-4.5, 2081-2100 | SSP5-8.5, 2081-2100 |
|---|---|---|---|
| Laos | 1.23 (1.00-1.37) | 1.31 (1.04-1.72) | 1.79 (1.22-2.49) |
| Costa Rica | 1.07 (0.92-1.34) | 1.13 (0.90-1.45) | 1.09 (0.71-1.52) |
| Burkina Faso | 1.29 (0.96-1.61) | 1.36 (0.96-1.70) | 1.61 (0.85-4.83) |
| Nepal | 1.20 (0.95-1.47) | 1.30 (1.03-1.79) | 1.59 (1.28-4.04) |
| Zambia | 1.16 (0.96-1.36) | 1.30 (1.04-1.67) | 1.66 (1.07-2.36) |
| Morocco | 0.94 (0.84-1.21) | 0.86 (0.78-1.04) | 0.74 (0.64-0.97) |
| Global, population-weighted mean of country multipliers | 1.34 (1.13-1.74) | 1.51 (1.28-2.46) | 2.44 (1.90-8.02) |
| Global, from the population-weighted mean dz | 1.31 (1.12-1.61) | 1.45 (1.25-2.10) | 2.05 (1.71-4.43) |

## Reading
- Median rx5day rises in Laos, Nepal, Zambia and Burkina Faso, changes little in Costa Rica (10-90% range spans 1 in every window shown) and falls in Morocco. In SSP2-4.5, 2041-2060, the median dz is above 0 in 87% of countries.
- The global figures depend on how they are averaged, because exp() stretches large values. In SSP5-8.5, 2081-2100, India (median multiplier 3.95) and China (2.72) contribute most; Ethiopia's median dz is 6.2.
- **Extrapolation.** A dz of 3 or more is outside what the frequency model was fitted on (within-country z-scores). Country-average rx5day has a small year-to-year SD in some large or inland countries (Ethiopia 7 mm on a mean of 119 mm), so a moderate percentage change becomes a large dz. Treat multipliers above about 2, and the upper ends of the SSP5-8.5 ranges, as out of range.
- Model sets differ between scenarios (21 to 27 models), so small differences between scenarios, such as global SSP3-7.0 mid-century (1.24) sitting below SSP2-4.5 (1.34), may come from which models are included. I did not test that.
- CMIP6 levels differ from ERA5 (historical median: Nepal 167 mm against 279 mm in ERA5, Zambia 130 against 98). That is why only relative changes are carried over.
