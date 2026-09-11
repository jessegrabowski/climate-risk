from collections.abc import Iterator
from pathlib import Path

from climate_risk.data.geocoding import Geocoder
from climate_risk.data.geonames import geonames_geocoder
from climate_risk.data.osm import osm_geocoder


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
