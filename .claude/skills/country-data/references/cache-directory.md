# The cache directory

Every function in `climate_risk` that reads or writes data takes a `cache_dir` argument. There is no
default value, no environment variable, and no search for a project root. The package reads no
environment variables at all.

```python
from pathlib import Path

import climate_risk as cr

cache_dir = Path("/data/climate-risk")

co2 = cr.load_co2_data(cache_dir)
panel = cr.build_country_year_panel(cache_dir)
```

The directory need not exist. Loaders create it on the first write.

## Why it is an argument

An ambient root, discovered by walking up from `__file__` or read from an environment variable,
makes every function's behavior depend on state the caller cannot see in the call. Two runs of the
same script then differ for reasons the script does not record, and a test that forgets to isolate
the variable writes into the developer's real cache. That cache runs to tens of gigabytes, so the
failure is expensive rather than merely confusing.

Making the argument required has consequences. One process can work against several caches, so
comparing a run against a frozen snapshot means passing a different path. Nothing writes outside
the path it was handed. And the path is resolved once, at the edge: a script decides where the
cache lives on its first line and threads the value down, because library code has nowhere to look.

**Adding a default is not an improvement.** If a caller finds threading the argument tedious, bind it
once with `functools.partial` or a small wrapper in their own code.

## What lives under it

Raw downloads keep their publisher's filename at the top of the cache. Sources that unpack into many
files, or that are fetched per country or per epoch, get a subdirectory: `gadm/`, `geo_disasters/`,
`geonames/`, `ghsl/`, `gpcc/`, `osm/`, `rivers/` and `shapefiles/`.

Derived frames are parquet, named from what was asked for. `cache_key` builds the stem, so `points`
requested with a grid size of 400 over Southeast Asia becomes:

```
points__grid_size=400__region=sea.parquet
```

Parameters are sorted, so two calls asking for the same thing hit the same entry however the caller
ordered them, and two calls asking for different things cannot collide.

## Refreshing

Loaders that download take `force_reload`:

```python
co2 = cr.load_co2_data(cache_dir, force_reload=True)
```

Deleting the file works equally well. The cache holds no index, so its state is exactly what is on
disk.

An entry whose *builder* changed turns over by itself. Several loaders key their cached frame on
`builder_fingerprint`, a digest of the building function's source and the rules it reads, so editing
a transformation writes a new entry instead of reading back a stale one. Editing a comment inside
the builder turns it over too, because the fingerprint is over the source text. That is the cost of
the guarantee.

## Size

A full cache holds every source, worldwide, at every GHS-POP epoch, and runs to tens of gigabytes.
GPCC and GHS-POP are most of that. Leaving those two out brings it to a few hundred megabytes.
Nothing prunes it, so the directory only grows.
