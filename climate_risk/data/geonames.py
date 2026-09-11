from pathlib import Path
from zipfile import ZipFile

import polars as pl

from climate_risk.data.cache import builder_fingerprint, cached, polars_parquet
from climate_risk.data.fetch import fetch
from climate_risk.data.geocoding import Geocoder
from climate_risk.data.place_names import keying_fingerprint, match_key
from climate_risk.data.source import DataSource

GEONAMES_URL = "https://download.geonames.org/export/dump"
GEONAMES_SUBDIRECTORY = "geonames"
GEONAMES_LICENSE = "CC BY 4.0"
GEONAMES_CITATION = "GeoNames geographical database, https://www.geonames.org, licensed CC BY 4.0."
GEONAMES_RETRIEVED = "2026-08-19"

# The dump is headerless and positional; these are the fields a name-to-point lookup reads, keyed
# on where they sit in a row.
DUMP_FIELDS = {1: "name", 2: "ascii_name", 3: "alternates", 4: "lat", 5: "lon", 14: "population"}

COUNTRY_INFO = DataSource(
    url=f"{GEONAMES_URL}/countryInfo.txt",
    filename="countryInfo.txt",
    license=GEONAMES_LICENSE,
    citation=GEONAMES_CITATION,
    retrieved=GEONAMES_RETRIEVED,
)


def geonames_dir(cache_dir: Path) -> Path:
    """Return the directory the GeoNames country dumps live in, inside ``cache_dir``."""
    return cache_dir / GEONAMES_SUBDIRECTORY


def country_dump(alpha2: str) -> DataSource:
    """Declare the dump for one country, which GeoNames files under its two-letter code."""
    return DataSource(
        url=f"{GEONAMES_URL}/{alpha2}.zip",
        filename=f"{alpha2}.zip",
        license=GEONAMES_LICENSE,
        citation=GEONAMES_CITATION,
        retrieved=GEONAMES_RETRIEVED,
    )


def read_country_codes(cache_dir: Path, *, force_reload: bool = False) -> dict[str, str]:
    """
    Map each country's alpha-3 code to the alpha-2 code GeoNames files its dump under.

    Parameters
    ----------
    cache_dir : Path
        Directory the caches live under.
    force_reload : bool, optional
        Re-download even when the file is already there. Default False.

    Returns
    -------
    codes : dict mapping str to str
        Alpha-2 code, keyed by alpha-3.
    """
    path = fetch(COUNTRY_INFO, geonames_dir(cache_dir), force=force_reload)
    codes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) > 1 and fields[0] and fields[1]:
            codes[fields[1]] = fields[0]

    return codes


def load_place_points(iso: str, cache_dir: Path, *, force_reload: bool = False) -> pl.DataFrame:
    """
    Read one country's GeoNames places, keyed for matching against written mentions.

    Every name a place is published under becomes a row: its own, its ASCII form and each
    alternate. A mention therefore reaches a point whichever spelling it used. Where a name is shared, the
    most populous place keeps it, which is what makes a written ``Manila`` the city rather than one
    of the hamlets sharing the name.

    Parameters
    ----------
    iso : str
        ISO 3166-1 alpha-3 code of the country to read.
    cache_dir : Path
        Directory the caches live under.
    force_reload : bool, optional
        Rebuild even when the cache is warm. Default False.

    Returns
    -------
    points : DataFrame
        Columns ``key``, ``lon`` and ``lat``, one row per distinct name.
    """
    directory = geonames_dir(cache_dir)

    def build() -> pl.DataFrame:
        alpha2 = read_country_codes(cache_dir)[iso]
        archive = fetch(country_dump(alpha2), directory)

        with ZipFile(archive) as zipped, zipped.open(f"{alpha2}.txt") as dump:
            # Names carry unescaped quotes, which a quoting reader swallows whole lines on.
            rows = pl.read_csv(
                dump.read(),
                separator="\t",
                has_header=False,
                quote_char=None,
                columns=list(DUMP_FIELDS),
                new_columns=list(DUMP_FIELDS.values()),
                schema_overrides={"lat": pl.Float64, "lon": pl.Float64, "population": pl.Int64},
                infer_schema_length=0,
            )

        spellings = rows.with_columns(
            pl.concat_list(
                pl.col("name"),
                pl.col("ascii_name"),
                pl.col("alternates").fill_null("").str.split(","),
            ).alias("written")
        ).explode("written", empty_as_null=False)

        keyed = spellings.with_columns(
            pl.col("written").map_elements(match_key, return_dtype=pl.String).alias("key")
        ).filter(pl.col("key") != "")

        return keyed.sort("population", descending=True).unique(subset="key", keep="first").select("key", "lon", "lat")

    return cached(
        directory,
        "places",
        build,
        polars_parquet(),
        # `keying` covers the rules that turn a name into a key, `reading` the rules that decide
        # which spelling of a place keeps one.
        params={"iso": iso, "keying": keying_fingerprint(), "reading": builder_fingerprint(build, DUMP_FIELDS)},
        force=force_reload,
    )


def geonames_geocoder(iso: str, cache_dir: Path, *, force_reload: bool = False) -> Geocoder:
    """
    Build a geocoder answering from one country's GeoNames places.

    Parameters
    ----------
    iso : str
        ISO 3166-1 alpha-3 code of the country to answer for.
    cache_dir : Path
        Directory the caches live under.
    force_reload : bool, optional
        Rebuild the places table even when it is cached. Default False.

    Returns
    -------
    geocoder : callable
        Takes an ISO code and a written name, and returns longitude and latitude or None.
    """
    points = load_place_points(iso, cache_dir, force_reload=force_reload)
    located = dict(zip(points["key"], zip(points["lon"], points["lat"], strict=True), strict=True))

    def locate(_: str, name: str) -> tuple[float, float] | None:
        return located.get(match_key(name))

    return locate
