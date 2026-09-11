# Adding a country

A country is one TOML file in `climate_risk/config/places/`, named after its ISO 3166-1 alpha-3
code in lower case. Zambia needs two lines:

```toml
iso3 = "ZMB"
name = "Zambia"
```

That is the whole minimum. The geometry comes from slicing the World Bank world shapefile by
`iso3`, and everything else falls back to the project-wide defaults in `climate_risk/config/schema.py`.
What those defaults are, and what the objects a file parses into hold, is in
[places and regions](places.md).

Nothing else has to change. If you find yourself editing a loader to make a country work, the
loader is wrong, not the config.

## What you get without asking

The directory is walked, so a new file is picked up with no registration step. Its ISO codes are
checked against the World Bank country table, and the file has to parse and match the schema. A
typo in a key is an error rather than a silently ignored default.

Downstream, `load_place("zmb")` gives you a `CountryConfig` that the loaders accept directly:

```python
from climate_risk.config.registry import load_place
from climate_risk.data_functions.shapefiles_data_loader import load_place_boundary

boundary = load_place_boundary(load_place("zmb"), cache_dir)
```

## What you can override

Every block below is optional.

**`[geometry]`**: the geographic and projected CRS.

**`[events]`**: `start_year`, `end_year`, `min_total_affected`, `min_deaths`. The defaults describe
the window the published panel uses. Lower them for a country with few recorded disasters, but
check first: all three shipped countries clear the defaults comfortably, and an empty event frame
is easy to mistake for a working pipeline.

**`[boundary]`**: a country-specific admin boundary archive, when the world shapefile is too coarse.
It needs `member` naming the layer to read inside the zip, alongside the usual source fields:

```toml
[boundary]
member = "lao_admin2.shp"
url = "https://.../lao_admin_boundaries.shp.zip"
filename = "lao_admin_boundaries.shp.zip"
license = "CC BY 3.0 IGO"
citation = "National Geographic Department (NGD), via the Humanitarian Data Exchange."
retrieved = "2026-08-08"
```

A boundary declared here is fetched, cached and reachability-checked exactly like a source declared
in code. `license` and `citation` are not decorative. Fill them in from the publisher's own page.

**`[event_location_overrides]`**: longitude and latitude forced onto EM-DAT records whose published
position is wrong, keyed by event id. Two countries may not both claim the same event.

## Reading a country's events

`[events]` says which records are severe enough and recent enough to count; it does not select a
country. Filtering is an ordinary polars predicate, so narrow it yourself:

```python
import polars as pl

from climate_risk.config.registry import load_place
from climate_risk.data_functions.emdat_processing import event_filter, load_emdat_events

place = load_place("zmb")
events = load_emdat_events(cache_dir).filter(event_filter(place.events) & (pl.col("ISO") == place.iso3))
```

## What is still global

One thing does not vary by country, and will surprise you if you assume it does.

The **river-damage frames** cover every country at once. `create_floods_rivers_damage` returns
world-wide flood events; the coordinate overrides are applied across all of them, which works
because event ids are country-specific.

## Before you trust it

The end-to-end tests run against synthetic fixtures. Every shipped country is checked to resolve to
geometry from its config alone, and to land where its ISO code says, but that is a weaker claim than
a real run.

The one test that uses real data is marked `requires_emdat` and skips wherever the licensed
workbook is absent, which includes CI. Run it once on a machine that has the workbook:

```
pixi run pytest -m requires_emdat
```

EM-DAT has to be obtained by hand. [Getting the data](../get_started/data.rst) says where from and
where to put it.

It checks that your country actually has events clearing its own filters. A country that does not
produces an empty panel, and every other test will pass on it.
