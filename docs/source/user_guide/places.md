# Places and regions

A place is a country or a group of countries, described by a TOML file and read into a frozen
dataclass. Nothing in `climate_risk/` names a country outside `climate_risk/config/places/`, and
that is the property the configuration exists to hold.

For the practical recipe, meaning what to put in a file and what you get for free, see
[adding a country](adding-a-country.md). This page is what the objects are and how they are read.

## The two shapes

`CountryConfig` is one country. It requires `iso3` and `name` and defaults everything else:

- `island`, recorded as a feature on the model's point data.
- `boundary`, a `ShapefileArchive` for a country-specific admin file, or `None` to slice the country
  out of the world shapefile by its ISO code.
- `geometry`, a `GeometrySpec`.
- `events`, an `EventFilters`.
- `event_location_overrides`, longitude and latitude forced onto named EM-DAT event ids.

`RegionConfig` is several countries analyzed together. It requires `key`, `name` and `members`, and
shares `geometry` and `events` with the country shape. A region with no members is refused, and so
is one repeating a member, which would double-count it.

`Place` is the union of the two. Code that does not care which it has takes a `Place` and calls
`resolve_isos`, which returns the members of a region or the single code of a country:

```python
from climate_risk.config.registry import load_place, resolve_isos

resolve_isos(load_place("lao"))  # ('LAO',)
resolve_isos(load_place("sea"))  # ('MMR', 'THA', 'LAO', ...)
```

## The shared blocks

`GeometrySpec` carries two coordinate reference systems: `geographic_crs`, which points are built
in, and `projected_crs`, which distances are measured in and which must be metric. The defaults are
`EPSG:4326` and `EPSG:3395`, and a place overrides them only where the project-wide pair distorts it.

`EventFilters` says which EM-DAT records a place's panel counts: `start_year` and `end_year` bound
the window, `min_total_affected` and `min_deaths` set severity floors. The defaults describe the
window the published panel uses.

It says nothing about *which country's* events. Selecting a country is an ordinary polars predicate
on the frame rather than a property of the filter, as in `event_filter(place.events) &
(pl.col("ISO") == place.iso3)`. Keeping the two apart is what lets the same filters apply to a region.

An `EventFilters` whose window ends before it starts is refused at construction.

## Reading a file

The directory a file sits in decides its schema: `places/` builds a `CountryConfig`, `regions/`
builds a `RegionConfig`. There is no registration step and no list to keep in sync. The directory
is globbed, so a new file is picked up by existing it.

`load_place(key)` takes the file stem: a lower-case ISO alpha-3 code for a country, a short name for
a region. It searches `places/` and then `regions/`, and if neither has the key it raises listing
every key that does exist. `read_place(path)` is the same reader against a path you supply, which is
how a place outside the shipped configuration is loaded.

Neither caches. A file edited on disk is read fresh on the next call.

## How TOML becomes objects

The sub-tables map onto the nested dataclasses by name: `[geometry]` builds a `GeometrySpec`,
`[events]` an `EventFilters`, `[boundary]` a `ShapefileArchive`. Everything else is passed through.
Two conversions are not one-to-one. A TOML array of members becomes a tuple, because the schema
promises immutability, and each override's coordinate pair becomes floats.

`[boundary]` is split rather than mapped. Its `member` names the layer to read inside the archive
and the remaining keys build a `DataSource`, so a boundary declared in a place file is fetched,
cached and reachability-checked exactly as one declared in Python is.

Because the dataclass constructor is called with the TOML keys as keyword arguments, **the key names
are the field names**. A misspelled key is a `TypeError` wrapped into a message naming the file and
the schema it failed. That is the intended behavior: a silently ignored key is a country configured
differently from how its file reads.

## Overrides across countries

`all_event_location_overrides()` collects the corrections every shipped country declares into one
mapping, keyed by EM-DAT event id. It refuses a duplicate: two countries claiming the same event
means one of them is wrong, and resolving it by file order would make the answer depend on the
alphabet.

The collected form exists because the river-damage frames are worldwide rather than per country. The
corrections have to be applied across all of them at once, which works because an event id already
carries the country it belongs to.
