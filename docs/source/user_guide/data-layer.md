# The data layer

Every upstream file in this project is described by an object that sits next to the loader reading
it, and every derived frame is written through one caching function. Two ideas, and most of
`climate_risk.data` follows from them.

## Declaring a source

A source is a frozen dataclass carrying where the data comes from, what it is stored as, and the
terms it is published under. There are three kinds, and which one a publisher gets is decided by
what it is possible to do with them.

`DataSource` is for a file the library can fetch: it has a `url`, a bare `filename`, a `license`, a
`citation`, and a `retrieved` date recording when the declaration was last checked against the
publisher. NOAA's CO2 record is one line of it:

```python
CO2 = DataSource(
    url="https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.csv",
    filename="co2_annmean_mlo.csv",
    license="public domain (U.S. Government work, 17 U.S.C. 105)",
    citation="...",
    retrieved="2026-08-05",
)
```

`ManualSource` is for a file the user has to place by hand. It carries a `homepage` instead of a
`url`, because there is nothing an unattended client is permitted to request. Its `require` method
is the whole interface: it returns the path, or raises with the license, the homepage and the exact
path the file is missing from. The four licensed sources are listed in the getting-started guide.

`ApiSource` is for a service answering a query rather than serving a file. It has neither a
`filename` nor anything to fetch, so it carries only the documented entry point and the terms.

The declarations validate themselves on construction. A `url` that is not http(s) is refused, and so
is a `filename` carrying a directory separator. A source able to write outside the cache directory
would break the one guarantee the cache makes.

A zipped shapefile needs one more thing: which layer inside the archive to read. `ShapefileArchive`
pairs a `DataSource` with a `member` path, and the case of that path matters, because a
case-insensitive filesystem hides a mismatch that fails on Linux.

## Fetching

`fetch(source, cache_dir)` returns the path to the file, downloading it if it is not already there.
Nothing else in the package downloads anything.

It writes to a `.part` file beside the destination and moves it into place only when the transfer
completes, so an interrupted download can never be read back as a cache hit. A dropped connection is
retried up to five times, each attempt asking for the bytes the last one stopped at, which is what
makes the multi-hundred-megabyte rasters obtainable at all. A `.part` file left by an earlier call
is deleted rather than resumed: nothing proves it is a prefix of the file being fetched now, and
appending to bytes from somewhere else writes a corrupt archive that looks complete.

Downloads carry a browser-shaped user agent. Some hosts, the World Bank boundaries archive among
them, answer 403 to `Python-urllib` and 200 to the same URL otherwise.

## Caching derived frames

Downloading is only half the cost. Reading a decade of gridded NetCDF and totalling it to countries
takes minutes, and doing it on every call would make the library unusable. `cached` is the one
implementation of check, build, write:

```python
return cached(cache_dir, "co2", build, polars_parquet(), force=force_reload)
```

`build` is called only on a miss. On a hit the artifact is read straight back, which is why a warm
cache never touches the network.

The format argument is a `CacheFormat`: a matched reader and writer declared together, so the two
cannot drift apart. Three are shipped: `pandas_parquet`, `polars_parquet` and `geo_parquet`.
Parquet is used throughout because it carries dtypes, the index, and in the geospatial case the
CRS, so nothing has to be restored by hand on the way back in.

### Keys

An artifact requested with different arguments is a different artifact. `cache_key` builds the
filename stem from a logical name and the parameters that distinguish the entry:

```python
cache_key("points", {"grid_size": 400, "region": "sea"})
# 'points__grid_size=400__region=sea'
```

Parameters are sorted, so ordering at the call site cannot produce two entries for one request. A
value carrying `/`, `\`, `.`, `__` or `=` is refused: the first two would write outside the cache,
and the rest would let two different parameter sets collide on a single filename.

### Fingerprints

A key records what was asked for. It does not record how the answer was produced, so an entry built
under rules that have since changed reads back as a hit and quietly poisons everything downstream.

`builder_fingerprint` closes that: it digests the builder function's source together with any values
the builder reads that its source does not show, and the digest goes into the key as a parameter.
Editing the transformation therefore writes a new entry instead of returning the old one. Editing a
comment inside the builder does too, because the fingerprint is over the source text, so a
formatting change turns the cache over. That is the cost of the guarantee, and it is deliberate.

Rules passed as fingerprint inputs must `repr` identically in every process. One that does not moves
the digest between runs, and nothing ever reads back.

## Loaders

A loader is the composition of the pieces above, and they all have the same shape:

```python
def load_co2_data(cache_dir: Path, *, force_reload: bool = False) -> pl.DataFrame:
    def build():
        raw = fetch(CO2, cache_dir, force=force_reload)
        return transform_co2(pl.read_csv(raw, skip_rows=CO2_HEADER_ROW))

    return cached(cache_dir, "co2", build, polars_parquet(), force=force_reload)
```

The transformation is a separate, pure function taking a frame and returning a frame. It never
touches the filesystem, which is what makes it testable without a cache and reusable on data from
somewhere else.

`force_reload` reaches both layers: it re-downloads the source and rebuilds what was derived from it.

## Frames

Tabular data is polars. Geospatial data is geopandas. A few loaders return pandas where an upstream
reader produces it and converting would buy nothing.

The layers are kept apart on purpose, and the rules are in [layering](layering.md).
