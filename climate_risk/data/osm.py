import json
import logging
import time
import urllib.parse

from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import polars as pl
import requests

from climate_risk.data.cache import cache_key, cached, polars_parquet
from climate_risk.data.fetch import TIMEOUT_SECONDS, USER_AGENT
from climate_risk.data.geocoding import Geocoder
from climate_risk.data.place_names import match_key
from climate_risk.data.source import ApiSource
from climate_risk.exceptions import UpstreamUnavailableError

_log = logging.getLogger(__name__)

OSM_SUBDIRECTORY = "osm"

NOMINATIM = ApiSource(
    url="https://nominatim.openstreetmap.org/search",
    license="ODbL 1.0",
    citation="OpenStreetMap contributors, https://www.openstreetmap.org/copyright, licensed ODbL 1.0.",
    retrieved="2026-09-05",
)

# The public instance allows one request a second for a job of this length, and requires results to
# be cached rather than re-requested. Both are conditions of use, not tuning.
RATE_LIMIT_SECONDS = 1.1

# What Nominatim answers with is whatever carries the name, so a village and the pharmacy named
# after it score alike. Only these two categories describe somewhere an event can happen; the rest
# are buildings, roads and amenities that share a name with the place around them.
PLACE_CATEGORIES = ("place", "boundary")

# One row per name asked about, whether or not it was found. A miss is a result: without recording
# it, every rebuild asks the whole question again. `kind` is Nominatim's `type`, renamed off the
# builtin. The name is stored as written and keyed at read time, so a change to the keying rules
# costs a lookup rather than another crawl.
LOOKUP_COLUMNS = {
    "written": pl.String,
    "lon": pl.Float64,
    "lat": pl.Float64,
    "category": pl.String,
    "kind": pl.String,
}


class NominatimAnswer(NamedTuple):
    """
    One place Nominatim reports, reduced to what the cache keeps.

    Parameters
    ----------
    lon : float
        Longitude of the point.
    lat : float
        Latitude of the point.
    category : str
        What kind of thing carries the name.
    kind : str
        The narrower type within that category, as Nominatim names it.
    """

    lon: float
    lat: float
    category: str
    kind: str


def osm_dir(cache_dir: Path) -> Path:
    """Return the directory the recorded OSM lookups live in, inside ``cache_dir``."""
    return cache_dir / OSM_SUBDIRECTORY


def search_nominatim(
    name: str, alpha2: str | None = None, *, session: requests.Session | None = None
) -> NominatimAnswer | None:
    """
    Ask Nominatim where one written name is, waiting out the rate limit before returning.

    The wait is here rather than in the caller because it is a condition of using the public
    instance: a loop that forgets it is one that gets the project blocked.

    A name Nominatim has no place for and a name it was never asked about are different answers.
    Only the first is None. A request that failed raises, because recording it as a miss would cache
    a permanent absence that nothing afterwards could tell apart from a real one.

    Parameters
    ----------
    name : str
        The place as written.
    alpha2 : str, optional
        ISO 3166-1 alpha-2 code to confine the search to. Default None, which searches everywhere.
    session : requests.Session, optional
        Session to make the request on, so a crawl reuses one connection. Default None, which makes
        a request of its own.

    Returns
    -------
    answer : NominatimAnswer or None
        The best answer, or None where Nominatim has no place of that name.
    """
    query = {"q": name, "format": "jsonv2", "limit": 1}
    if alpha2:
        query["countrycodes"] = alpha2.lower()

    get = session.get if session else requests.get
    try:
        response = get(
            f"{NOMINATIM.url}?{urllib.parse.urlencode(query)}",
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        answers = response.json()
    except (requests.RequestException, json.JSONDecodeError) as error:
        raise UpstreamUnavailableError(f"Nominatim could not be asked about {name!r}: {error}") from error
    finally:
        time.sleep(RATE_LIMIT_SECONDS)

    if not answers:
        return None

    best = answers[0]

    return NominatimAnswer(
        float(best["lon"]), float(best["lat"]), str(best.get("category", "")), str(best.get("type", ""))
    )


def read_osm_places(iso: str, cache_dir: Path) -> pl.DataFrame:
    """
    Read the answers Nominatim has already given for one country.

    Reads only what is cached. Filling the cache is the job of ``tools/fetch_osm_places.py``, which
    keeps the crawl out of every code path that merely wants to look a name up.

    Parameters
    ----------
    iso : str
        ISO 3166-1 alpha-3 code of the country to read.
    cache_dir : Path
        Directory the caches live under.

    Returns
    -------
    places : DataFrame
        Columns ``written``, ``lon``, ``lat``, ``category`` and ``kind``, one row per name asked
        about. Empty where nothing has been asked for this country yet.
    """
    stored = polars_parquet()
    path = (osm_dir(cache_dir) / cache_key("lookups", {"iso": iso})).with_suffix(stored.suffix)

    return stored.read(path) if path.exists() else pl.DataFrame(schema=LOOKUP_COLUMNS)


def osm_geocoder(iso: str, cache_dir: Path) -> Geocoder:
    """
    Build a geocoder answering from the OpenStreetMap points cached for one country.

    Only answers Nominatim described as a place or a boundary are offered. Filtering on the name
    alone accepts a pharmacy for the village it is named after, which lands a point somewhere
    plausible and wrong.

    Parameters
    ----------
    iso : str
        ISO 3166-1 alpha-3 code of the country to answer for.
    cache_dir : Path
        Directory the caches live under.

    Returns
    -------
    geocoder : callable
        Takes an ISO code and a written name, and returns longitude and latitude or None.
    """
    found = read_osm_places(iso, cache_dir).filter(
        pl.col("lon").is_not_null() & pl.col("category").is_in(PLACE_CATEGORIES)
    )
    points = zip(found["lon"], found["lat"], strict=True)
    located = {match_key(written): point for written, point in zip(found["written"], points, strict=True)}

    def locate(_: str, name: str) -> tuple[float, float] | None:
        return located.get(match_key(name))

    return locate


def record_lookups(iso: str, cache_dir: Path, answers: Mapping[str, NominatimAnswer | None]) -> pl.DataFrame:
    """
    Add answers to a country's cache, keeping whatever is already there.

    Parameters
    ----------
    iso : str
        ISO 3166-1 alpha-3 code the names belong to.
    cache_dir : Path
        Directory the caches live under.
    answers : mapping of str to NominatimAnswer or None
        What Nominatim said about each written name, None where it said nothing.

    Returns
    -------
    places : DataFrame
        The country's whole cache, as it now stands on disk.
    """
    fresh = pl.DataFrame(
        [
            {
                "written": written,
                "lon": answer.lon if answer else None,
                "lat": answer.lat if answer else None,
                "category": answer.category if answer else None,
                "kind": answer.kind if answer else None,
            }
            for written, answer in answers.items()
        ],
        schema=LOOKUP_COLUMNS,
        orient="row",
    )

    def merged() -> pl.DataFrame:
        return pl.concat([read_osm_places(iso, cache_dir), fresh]).unique(subset="written", keep="last")

    return cached(osm_dir(cache_dir), "lookups", merged, polars_parquet(), params={"iso": iso}, force=True)
