# Which loader answers which question

Every function below takes `cache_dir`, but not always first: a loader scoped to one country or
one epoch takes that first, as in `load_shapefile("world", cache_dir)` and
`load_place_points("LAO", cache_dir)`. Check the signature rather than assuming. The three builders
compose the single-source loaders, so reach for a builder first and drop to a loader only when the
extra columns are unwanted.

## The three builders

```python
import climate_risk as cr

panel = cr.build_country_year_panel(cache_dir)
series = cr.build_time_series(cache_dir)
precipitation = cr.annual_precipitation(cache_dir)
```

`build_country_year_panel` is the modeling frame: disaster counts and damages from EM-DAT, World
Bank development indicators, and annual precipitation, on one row per country and year keyed on
`ISO` and `Start_Year`. A country earns a row only when it has both a disaster record and
development indicators. **Needs EM-DAT**, so it is the call that first raises for a cache without it.

`build_time_series` merges CO2, ocean heat and worldwide precipitation onto one row per year. It
downloads GPCC, which is gigabytes on a cold cache.

`annual_precipitation` totals GPCC to `ISO`, `year`, `precip`. It covers the whole GPCC record,
which reaches further back than the EM-DAT panel does.

## The event table

```python
from climate_risk.data_functions.emdat_processing import event_filter, load_emdat_events

events = load_emdat_events(cache_dir)
```

One row per recorded event keyed by `DisNo.`, carrying `disaster_class` and the renamed damage
columns, with nothing filtered out. Narrow it with `event_filter(filters)`, which returns a polars
expression, or with any other predicate.

`event_filter` selects on severity and window only. It does not select a country. That is an
ordinary predicate the caller adds:

```python
import polars as pl

from climate_risk.config.registry import load_place

place = load_place("zmb")
mine = events.filter(event_filter(place.events) & (pl.col("ISO") == place.iso3))
```

## Single sources

`load_co2_data`, `load_ocean_heat_data`, `load_hadcrut_data`, `load_gpcc_data`, `load_wb_data`,
`load_rivers_data`, `load_shapefile`, `process_ipcc_scenarios` are exported from the package root.
The first call downloads, every later one reads the cache. Most take a keyword-only
`force_reload` to re-fetch. `load_rivers_data` does not, and neither do the three builders or
`load_emdat_events`. Delete the cached file for those.

Tabular data comes back as polars. Geospatial data comes back as geopandas. A few loaders return
pandas where an upstream reader produces it and converting would buy nothing.

## Places

```python
from climate_risk.config.registry import load_place, resolve_isos

lao = load_place("lao")
resolve_isos(lao)
```

`load_place` takes the file stem and returns a `CountryConfig` or `RegionConfig`. The stem is a
lower-case ISO alpha-3 code for a country, a short name for a region. `resolve_isos` gives the
codes either covers, which is one for a country. Nothing caches, so a file edited on disk is read fresh.
