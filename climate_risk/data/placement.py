from collections.abc import Iterator
from pathlib import Path

import polars as pl

from climate_risk.data.gadm import GADM_LAYER
from climate_risk.data.geocoding import Geocoder, units_from_geocoders
from climate_risk.data.geonames import geonames_geocoder
from climate_risk.data.osm import osm_geocoder
from climate_risk.data.place_names import NAMES_NO_UNIT, read_gazetteer, resolve_event_places
from climate_risk.data_functions.emdat_processing import (
    TEXT_RESOLUTION_SCHEMA,
    TEXT_UNIT_SCHEMA,
    events_missing_units,
)


def available_geocoders(iso: str, cache_dir: Path) -> Iterator[Geocoder]:
    """
    Yield every point source that can answer for one country, most trusted first.

    GeoNames comes first: it is a gazetteer of populated places, so a name it knows is a settlement
    rather than whatever object happened to carry the name. OpenStreetMap answers second, reaching
    the villages and statistical regions GeoNames has no row for. A source with nothing cached for
    this country is skipped rather than raised over, because most countries have only some of them.

    Parameters
    ----------
    iso : str
        ISO 3166-1 alpha-3 code of the country to answer for.
    cache_dir : Path
        Directory the caches live under.

    Yields
    ------
    callable
        Takes an ISO code and a written name, and returns longitude and latitude or None.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.placement import available_geocoders

        for geocoder in available_geocoders("LAO", Path("data")):
            print(geocoder("LAO", "Pakse"))
    """
    try:
        yield geonames_geocoder(iso, cache_dir)
    except (KeyError, OSError):
        pass

    yield osm_geocoder(iso, cache_dir)


def resolve_location_text(
    events: pl.DataFrame, cache_dir: Path, *, layer: str = GADM_LAYER, verbose: bool = False
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Read the location text of every event EM-DAT left uncoded into GADM units.

    One country's gazetteer and point sources are loaded once and used for all of its events, so
    the cost is per country rather than per event. A country with no gazetteer is skipped.

    Parameters
    ----------
    events : DataFrame
        The workbook, from :func:`~climate_risk.data_functions.emdat_processing.load_emdat_events`.
        Filter it first to resolve one place rather than the world.
    cache_dir : Path
        Directory the caches live under.
    layer : str, optional
        Layer to read inside the GADM GeoPackage. Default ``GADM_LAYER``.
    verbose : bool, optional
        Log each gazetteer and unit table read. Default False: one country alone reads eight.

    Returns
    -------
    from_text : DataFrame
        One row per event and unit its prose reached, with the columns of ``TEXT_UNIT_SCHEMA``.
    text_resolution : DataFrame
        One row per event read, with the columns of ``TEXT_RESOLUTION_SCHEMA``. A name that can
        only be a feature, such as a sea, is counted in neither column, so an event writing nothing
        else reads as zero names written rather than as an event whose text went unread.
    """
    uncoded: dict[str, list] = {}
    for event in events_missing_units(events):
        uncoded.setdefault(event.iso, []).append(event)

    units: list[dict[str, object]] = []
    resolution: list[dict[str, object]] = []

    for iso, group in sorted(uncoded.items()):
        gazetteer = read_gazetteer(iso, cache_dir, layer=layer, verbose=verbose)
        if not gazetteer.names:
            continue

        level_of = {unit.gid: unit.level for found in gazetteer.names.values() for unit in found}
        wanted = {place.name for event in group for place in event.places}
        located = units_from_geocoders(wanted, iso, cache_dir, available_geocoders(iso, cache_dir))

        for event in group:
            written = [(place.name, place.parent) for place in event.places]
            placeable = [
                placement
                for placement in resolve_event_places(written, gazetteer, located=located)
                if placement.how != NAMES_NO_UNIT
            ]

            resolution.append(
                {
                    "DisNo.": event.disno,
                    "names_written": len(placeable),
                    "names_reached": sum(1 for placement in placeable if placement.gids),
                }
            )
            units.extend(
                {"DisNo.": event.disno, "gid": gid, "name": None, "admin_level": level_of.get(gid)}
                for placement in placeable
                for gid in sorted(placement.gids)
            )

    return (
        pl.DataFrame(units, schema=TEXT_UNIT_SCHEMA),
        pl.DataFrame(resolution, schema=TEXT_RESOLUTION_SCHEMA),
    )
