import logging

from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path

import geopandas as gpd
import pandas as pd

from anyascii import anyascii

from climate_risk.data.cache import cached, pandas_parquet
from climate_risk.data.gadm import load_units_in_country
from climate_risk.data.source import ManualSource
from climate_risk.exceptions import DataValidationError

_log = logging.getLogger(__name__)

# The geometries are a GAUL 2015 derivative and non-commercial, so the archive is placed by hand
# and never appears in the fetchable registry. The attribute table is CC-BY-4.0.
GEO_DISASTERS = ManualSource(
    filename="disaster_subnational_90_23.gpkg",
    homepage="https://doi.org/10.5281/zenodo.15487667",
    license=(
        "Spatial geometries are © FAO 2015 under the GAUL 2015 Data License, non-commercial with "
        "attribution required. All non-spatial attributes are CC-BY-4.0."
    ),
    citation=(
        "Teber, K., Weynants, M., Gans, F., & Mahecha, M. D. (2025). Geo-Disasters v1.0.0: geocoded "
        "EM-DAT climate-disaster footprints (1990-2023). https://doi.org/10.5281/zenodo.15487667"
    ),
    retrieved="2026-08-09",
)

GEO_DISASTERS_LAYER = "disaster_subnational_90_23"

# What a location is keyed and named by, per admin level.
NAME_COLUMNS = {1: "ADM1_NAME", 2: "ADM2_NAME"}

LOCATION_COLUMNS = ["DisNo.", "ISO", "admin_level", "geocoding_q", "ADM1_NAME", "ADM2_NAME"]

# Attribution required by the GAUL 2015 Data License, clause 2(a), verbatim.
GAUL_ATTRIBUTION = (
    "Source of Administrative boundaries: The Global Administrative Unit Layers (GAUL) dataset, "
    "implemented by FAO within the CountrySTAT and Agricultural Market Information System (AMIS) "
    "projects"
)


def geo_disasters_dir(cache_dir: Path) -> Path:
    """Return the directory the Geo-Disasters GeoPackage lives in, inside ``cache_dir``."""
    return cache_dir / "geo_disasters"


def geo_disasters_path(cache_dir: Path) -> Path:
    """
    Return the path to the Geo-Disasters GeoPackage, raising if it has not been placed in the cache.

    Parameters
    ----------
    cache_dir : Path
        Directory the caches live under.

    Returns
    -------
    path : Path
        Location of ``disaster_subnational_90_23.gpkg``.
    """
    return GEO_DISASTERS.require(geo_disasters_dir(cache_dir))


def load_event_locations(cache_dir: Path, *, iso: str | None = None, layer: str = GEO_DISASTERS_LAYER) -> pd.DataFrame:
    """
    Read the geocoded locations Geo-Disasters records, one row per affected administrative unit.

    Geometry is left on disk. The polygons carry the non-commercial GAUL license while the attribute
    table is CC-BY-4.0, and the attributes are what a comparison against EM-DAT needs.

    Parameters
    ----------
    cache_dir : Path
        Directory the caches live under.
    iso : str, optional
        Restrict to one ISO 3166-1 alpha-3 country code. Default None, meaning every country.
    layer : str, optional
        Layer to read inside the GeoPackage. Default ``GEO_DISASTERS_LAYER``.

    Returns
    -------
    locations : DataFrame
        Columns ``DisNo.``, ``ISO``, ``admin_level``, ``geocoding_q``, ``ADM1_NAME`` and
        ``ADM2_NAME``, one row per location.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.geo_disasters import load_event_locations

        locations = load_event_locations(Path("data"), iso="LAO")
    """
    path = geo_disasters_path(cache_dir)
    where = f"ISO = '{iso}'" if iso is not None else None

    locations = gpd.read_file(path, layer=layer, columns=LOCATION_COLUMNS, where=where, ignore_geometry=True)

    return pd.DataFrame(locations).reset_index(drop=True)


def unit_names(locations: pd.DataFrame) -> pd.Series:
    """
    Name each location at the level it was geocoded to.

    Parameters
    ----------
    locations : DataFrame
        Rows as :func:`load_event_locations` returns them.

    Returns
    -------
    names : Series
        One name per row, taken from the column its ``admin_level`` points at.
    """
    named = pd.Series(pd.NA, index=locations.index, dtype="object")
    for level, column in NAME_COLUMNS.items():
        at_level = locations["admin_level"] == level
        named[at_level] = locations.loc[at_level, column]

    return named


def load_event_footprints(cache_dir: Path, *, iso: str, layer: str = GEO_DISASTERS_LAYER) -> gpd.GeoDataFrame:
    """
    Read one country's geocoded locations with the polygons attached.

    These polygons are the non-commercial half of the license, and :data:`GAUL_ATTRIBUTION` has to
    appear on anything derived from them. :func:`load_event_locations` reads the same rows without
    geometry where the attributes alone will do.

    Parameters
    ----------
    cache_dir : Path
        Directory the caches live under.
    iso : str
        ISO 3166-1 alpha-3 country code to read.
    layer : str, optional
        Layer to read inside the GeoPackage. Default ``GEO_DISASTERS_LAYER``.

    Returns
    -------
    footprints : GeoDataFrame
        The columns of :func:`load_event_locations` and the footprint of each location.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.geo_disasters import load_event_footprints

        footprints = load_event_footprints(Path("data"), iso="LAO")
    """
    footprints = gpd.read_file(geo_disasters_path(cache_dir), layer=layer, where=f"ISO = '{iso}'")

    return footprints.reset_index(drop=True)


EQUAL_AREA_CRS = "ESRI:54009"

RESOLVED_DTYPES = {
    "DisNo.": "string",
    "ISO": "string",
    "geometry_source": "string",
    "gid": "string",
    "name": "string",
    "admin_level": "Int64",
    "geocoding_q": "Int64",
    "overlap": "float64",
}

RESOLVED_COLUMNS = list(RESOLVED_DTYPES)


# A unit belongs to an event when most of the unit lies inside the event's footprint.
MIN_CONTAINMENT = 0.5


def _units_within_events(footprints: gpd.GeoDataFrame, units: gpd.GeoDataFrame) -> pd.DataFrame:
    """Pair each event with every unit lying mostly inside its footprint, dropping those with none."""
    # Repaired before the union: a footprint crossing the antimeridian is valid in lat/lon and
    # self-intersecting once projected, and unioning one raises rather than returning a bad answer.
    repaired = footprints.assign(geometry=footprints.geometry.make_valid())

    # One geometry per event, so a unit straddling two of its footprints is measured against the
    # whole and cannot arrive twice. The quality flag takes the worse of the footprints it came
    # from, which is how Geo-Disasters propagates it.
    merged = repaired.dissolve(by="DisNo.", aggfunc={"ISO": "first", "geocoding_q": "max"}).reset_index()
    measured = units.assign(unit_area=units.geometry.area)
    pieces = gpd.overlay(merged, measured, how="intersection", keep_geom_type=False)
    pieces = pieces.assign(covered=pieces.geometry.area)

    # Membership is the share of the unit inside the footprint, which means the same thing whatever
    # the unit's size; the share of the footprint the unit covers does not, and is the weight below.
    inside = pieces[pieces["covered"] / pieces["unit_area"] > MIN_CONTAINMENT]
    if inside.empty:
        return pd.DataFrame(columns=RESOLVED_COLUMNS)

    footprint_area = merged.set_index("DisNo.").geometry.area
    shares = inside["covered"].to_numpy() / footprint_area.reindex(inside["DisNo."]).to_numpy()

    return inside.assign(overlap=shares).drop(columns=["geometry", "covered", "unit_area"])


def resolve_to_gadm(footprints: gpd.GeoDataFrame, cache_dir: Path) -> pd.DataFrame:
    """
    Name each Geo-Disasters footprint as the GADM unit it covers, so it can join EM-DAT's own units.

    The two gazetteers share no identifier and spell the same unit differently, so a footprint is
    placed by where it is rather than what it is called. Every GADM unit more than
    ``MIN_CONTAINMENT`` of the way inside the footprint is named, because one GAUL unit routinely
    covers several GADM ones. Where GAUL's admin-1 is a region and GADM's is a province, one
    footprint holds five or six.

    An event's footprints are dissolved per administrative level before the overlay, so a unit
    straddling two of them is measured against the whole and named once.

    An event holding no unit at all is dropped and logged.

    Parameters
    ----------
    footprints : GeoDataFrame
        Locations for one country, as :func:`load_event_footprints` returns them.
    cache_dir : Path
        Directory the caches live under.

    Returns
    -------
    resolved : DataFrame
        ``DisNo.``, ``ISO``, ``geometry_source``, a GADM unit, the location's ``geocoding_q``, and
        ``overlap``, the share of the event's footprint that unit covers. One row per event and
        unit, so an event spanning several units contributes several.
    """
    if footprints.empty:
        return pd.DataFrame(columns=RESOLVED_COLUMNS).astype(RESOLVED_DTYPES)

    countries = set(footprints["ISO"])
    if len(countries) > 1:
        raise DataValidationError(
            f"resolve_to_gadm reads one country's units at a time, but these footprints span "
            f"{sorted(countries)}. Call it once per country."
        )

    iso = countries.pop()
    matched = []

    for level in sorted(set(footprints["admin_level"]) & set(NAME_COLUMNS)):
        at_level = footprints[footprints["admin_level"] == level].to_crs(EQUAL_AREA_CRS)
        units = load_units_in_country(iso, level, cache_dir).to_crs(EQUAL_AREA_CRS)
        if units.empty:
            _log.warning("GADM holds no level-%d unit for %s, so %d footprints go unplaced", level, iso, len(at_level))
            continue

        matched.append(_units_within_events(at_level, units))

    resolved = pd.concat(matched) if matched else pd.DataFrame(columns=RESOLVED_COLUMNS)
    # Counted on events rather than rows: one event names as many units as its footprint holds.
    events = footprints["DisNo."].nunique()
    placed = resolved["DisNo."].nunique()
    if events > placed:
        _log.warning("%d of %d %s events hold no GADM unit and were dropped", events - placed, events, iso)

    return (
        pd.DataFrame(resolved)
        .assign(geometry_source="geo_disasters")[RESOLVED_COLUMNS]
        .sort_values(["DisNo.", "gid"])
        .reset_index(drop=True)
        .astype(RESOLVED_DTYPES)
    )


def _placement_rule() -> str:
    """Name the procedure that placed a cached entry, so the entry cannot outlive it.

    Rename this whenever the procedure changes. The threshold follows :data:`MIN_CONTAINMENT`.
    """
    # A cache key may not carry a dot, so the threshold is written without one.
    return f"dissolved-containment{MIN_CONTAINMENT}".replace(".", "-")


def _resolve_country(iso: str, cache_dir: Path) -> pd.DataFrame:
    return resolve_to_gadm(load_event_footprints(cache_dir, iso=iso), cache_dir)


def load_resolved_units(isos: Sequence[str], cache_dir: Path, *, force_reload: bool = False) -> pd.DataFrame:
    """
    Read the GADM units Geo-Disasters places, for several countries at once.

    Resolving is a polygon overlay against every GADM unit in the country, so it is cached, one
    entry per country.

    Parameters
    ----------
    isos : sequence of str
        ISO 3166-1 alpha-3 codes to read. Duplicates and ordering do not matter. The result is
        ordered by code.
    cache_dir : Path
        Directory the caches live under.
    force_reload : bool, optional
        Resolve again and overwrite the cached entries. Default False.

    Returns
    -------
    resolved : DataFrame
        The columns of :func:`resolve_to_gadm`, one row per event and unit, across every country
        asked for.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.geo_disasters import load_resolved_units

        resolved = load_resolved_units(["LAO", "THA"], Path("data"))
    """
    frames = [
        cached(
            cache_dir,
            "geo_disasters_units",
            partial(_resolve_country, iso, cache_dir),
            pandas_parquet(),
            params={"iso": iso, "placement": _placement_rule()},
            force=force_reload,
        )
        for iso in sorted(set(isos))
    ]
    if not frames:
        return pd.DataFrame(columns=RESOLVED_COLUMNS).astype(RESOLVED_DTYPES)

    return pd.concat(frames, ignore_index=True)


def normalize_unit_name(name: str) -> str:
    """
    Reduce an administrative unit's name to what two gazetteers can be expected to agree on.

    GADM and GAUL romanize the same unit differently, and accents, hyphens and casing all drift, so
    a literal comparison reports spelling as disagreement. The name is transliterated rather than
    decomposed, because a letter like ``Đ`` or ``ł`` carries no combining mark to strip.

    Parameters
    ----------
    name : str
        A unit name as either source publishes it.

    Returns
    -------
    normalized : str
        Casefolded, transliterated to ASCII, and stripped of everything that is not a letter or digit.
    """
    return "".join(character for character in anyascii(name) if character.isalnum()).casefold()


def event_unit_ids(units: pd.DataFrame) -> dict[str, set[str]]:
    """
    Collect the GADM units an event was placed in, keyed on the EM-DAT event id.

    Both sides of the comparison arrive in this shape: EM-DAT's own unit table, and the resolved
    footprints :func:`resolve_to_gadm` returns.

    Parameters
    ----------
    units : DataFrame
        Any frame carrying ``DisNo.`` and ``gid``, one row per event-unit.

    Returns
    -------
    unit_ids : dict mapping str to set of str
        One entry per event, holding every GADM identifier it was placed in.
    """
    return {str(disno): set(group.dropna()) for disno, group in units.groupby("DisNo.")["gid"]}


AGREEMENT_LEVELS = ("exact", "partial", "disjoint", "gained", "unmatched")

AGREEMENT_COLUMNS = ["DisNo.", "agreement", "em_dat_units", "geo_disasters_units", "shared_units"]


def _classify(em_dat: set[str], geo_disasters: set[str]) -> str:
    if not em_dat:
        return "gained"
    if not geo_disasters:
        return "unmatched"
    if em_dat == geo_disasters:
        return "exact"

    return "partial" if em_dat & geo_disasters else "disjoint"


def compare_event_units(*, em_dat: Mapping[str, set[str]], geo_disasters: Mapping[str, set[str]]) -> pd.DataFrame:
    """
    Report how EM-DAT's own geocoding and Geo-Disasters' agree, event by event.

    Both sides are GADM identifiers, matched literally. The signature cannot tell an identifier from
    a unit name and the two gazetteers' names do not compare, so Geo-Disasters' side comes through
    :func:`resolve_to_gadm` first.

    An identifier names a level as well as a place, so a province and a district inside it never
    match, and an event the two sources describe at different resolutions reads as ``disjoint``.

    Each event is classified as ``exact`` when both name the same units, ``partial`` when they
    overlap, ``disjoint`` when both name units and none is shared, ``gained`` when only
    Geo-Disasters has any, and ``unmatched`` when only EM-DAT does. An event neither has is absent.

    Parameters
    ----------
    em_dat : mapping of str to set of str
        GADM identifiers EM-DAT records, keyed on event id.
    geo_disasters : mapping of str to set of str
        The same from Geo-Disasters, as :func:`event_unit_ids` returns it.

    Returns
    -------
    comparison : DataFrame
        Columns ``DisNo.``, ``agreement`` and the three counts, one row per event either source
        geocoded, ordered by event id.
    """
    rows = []

    for disno in sorted(em_dat.keys() | geo_disasters.keys()):
        recorded = {gid for gid in em_dat.get(disno, ()) if gid}
        published = {gid for gid in geo_disasters.get(disno, ()) if gid}
        if not recorded and not published:
            continue

        rows.append(
            {
                "DisNo.": disno,
                "agreement": _classify(recorded, published),
                "em_dat_units": len(recorded),
                "geo_disasters_units": len(published),
                "shared_units": len(recorded & published),
            }
        )

    return pd.DataFrame(rows, columns=AGREEMENT_COLUMNS)
