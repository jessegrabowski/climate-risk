---
name: country-data
description: >-
  Load climate and disaster data for a country with the climate_risk package, and get past the two
  walls that stop every newcomer: every loader takes an explicit cache_dir with no default, and four
  of the upstream sources are license-walled and cannot be downloaded by code at all. Use this
  whenever someone is loading, fetching or caching data with climate_risk, is choosing between
  load_emdat_events, build_country_year_panel, build_time_series or a single-source loader, is
  adding a new country or region, or asks "where do I put the data", "why can't it download EM-DAT",
  "what is cache_dir", "NotImplementedError No emdat.xlsx was found", "how do I add a country", or
  "why is it downloading again". Also use it before writing any downloader, default cache path, or
  environment-variable lookup against this package, because all three are deliberate absences rather
  than gaps. Prefer this skill over reading the source.
---

# Getting country data out of climate_risk

`climate_risk` builds disaster-frequency and damage panels from a set of upstream sources. Most
of them download themselves. Four cannot, and the package holds no default location to put any
of them. Those two facts cause nearly every failure a newcomer hits, so start there rather than at
the API.

Route by what the user is doing.

## Start here, always

- [references/cache-directory.md](references/cache-directory.md): `cache_dir` is an argument to
  every loader. There is no default, no environment variable, and no project-root search. Read this
  before writing any call, and before "fixing" the absence of a default.
- [references/obtaining-data.md](references/obtaining-data.md): which sources download themselves
  and which four must be placed by hand, with the exact filename and path each loader looks for.
  Read this the moment a `NotImplementedError` names a missing file.

## Loading something

- [references/loading.md](references/loading.md): which function answers which question. The
  country-year panel, the worldwide annual series, the raw event table, and the single-source
  loaders, with what each returns and which sources it pulls.
- [concepts/data-layer.md](concepts/data-layer.md): how a loader is built: `DataSource` versus
  `ManualSource`, how `fetch` and `cached` compose, and why a warm cache never touches the network.
  Read when changing a loader rather than calling one.

## Adding a country or region

- [concepts/adding-a-country.md](concepts/adding-a-country.md): the recipe. A country is one TOML
  file and no Python.
- [concepts/places.md](concepts/places.md): what `CountryConfig`, `RegionConfig`, `GeometrySpec`
  and `EventFilters` hold, and how TOML keys map onto them.

## Placing events on a map

- [concepts/locating-events.md](concepts/locating-events.md): how EM-DAT's free-text `Location`
  becomes GADM units, and what to run to reproduce it.

## Before changing anything in the package

- [concepts/design-notes.md](concepts/design-notes.md): four things that look like oversights and
  are decisions. Read before removing one.
- [concepts/layering.md](concepts/layering.md): what may not import what. Nothing enforces these
  rules, so they are broken by accident.

## What not to add

**Do not write a downloader for a license-walled source.** EM-DAT, GADM, Geo-Disasters and the Penn
World Table forbid automated download. The loader raising `NotImplementedError` is the designed
behavior, and its message already names the license, the homepage and the destination path.

**Do not invent a default `cache_dir`.** Not a module constant, not `Path.cwd()`, not an environment
variable, not a walk up from `__file__`. The argument is required on purpose.

**Do not correct `disaster_class` values against a dictionary.** `Hydrometeorological` and
`Climatological` are this project's own two-class vocabulary, built from EM-DAT's `Disaster Type` by
`DISASTER_CLASSES`. They are not an EM-DAT partition, and a type outside the mapping becomes null
rather than a guess.
