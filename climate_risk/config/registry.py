import tomllib

from pathlib import Path
from typing import Any

from climate_risk.config.schema import CountryConfig, EventFilters, GeometrySpec, Place, RegionConfig
from climate_risk.data.source import DataSource, ShapefileArchive

CONFIG_ROOT = Path(__file__).parent

COUNTRY_SUBDIRECTORY = "places"
REGION_SUBDIRECTORY = "regions"


def read_place(path: Path) -> Place:
    """
    Read one place from its TOML file.

    A file under ``places/`` is a country and one under ``regions/`` is a region, so the directory
    decides which schema applies.

    Parameters
    ----------
    path : Path
        The TOML file to read.

    Returns
    -------
    place : CountryConfig or RegionConfig
        The place the file describes.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.config.registry import read_place

        laos = read_place(Path("climate_risk/config/places/lao.toml"))
    """
    try:
        table = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path} is not valid TOML: {error}") from error

    build = RegionConfig if path.parent.name == REGION_SUBDIRECTORY else CountryConfig
    try:
        return build(**_nested(table))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} does not match the {build.__name__} schema: {error}") from error


def load_place(key: str, *, root: Path = CONFIG_ROOT) -> Place:
    """
    Read the country or region filed under ``key``.

    Parameters
    ----------
    key : str
        The file's stem: an ISO 3166-1 alpha-3 code in lower case for a country, a short name for a
        region.
    root : Path, optional
        Directory holding ``places/`` and ``regions/``. Default the shipped configuration.

    Returns
    -------
    place : CountryConfig or RegionConfig
        The place, read fresh from disk.

    Examples
    --------
    .. code-block:: python

        from climate_risk.config.registry import load_place

        laos = load_place("lao")
    """
    for subdirectory in (COUNTRY_SUBDIRECTORY, REGION_SUBDIRECTORY):
        path = root / subdirectory / f"{key}.toml"
        if path.is_file():
            return read_place(path)

    known = ", ".join(_place_keys(root)) or "nothing"
    raise ValueError(f"No place is filed under {key!r}. Known places: {known}.")


def resolve_isos(place: Place) -> tuple[str, ...]:
    """
    Return the ISO 3166-1 alpha-3 codes a place covers, which is one for a country.

    Examples
    --------
    A region resolves to its members, a country to itself:

    .. code-block:: python

        from climate_risk.config.registry import load_place, resolve_isos

        print(resolve_isos(load_place("sea")))
    """
    return (place.iso3,) if isinstance(place, CountryConfig) else place.members


def _nested(table: dict[str, Any]) -> dict[str, Any]:
    """Turn a place's sub-tables into the objects the schema expects, leaving the rest alone."""
    fields = dict(table)

    for key, build in (("geometry", GeometrySpec), ("events", EventFilters)):
        if key in fields:
            fields[key] = build(**fields[key])

    if "boundary" in fields:
        fields["boundary"] = _boundary(fields["boundary"])

    if "members" in fields:
        # TOML arrays parse as lists, and the schema promises a tuple.
        fields["members"] = tuple(fields["members"])

    if "event_location_overrides" in fields:
        fields["event_location_overrides"] = {
            event: (float(lon), float(lat)) for event, (lon, lat) in fields["event_location_overrides"].items()
        }

    return fields


def _boundary(table: dict[str, Any]) -> ShapefileArchive:
    """Split a ``[boundary]`` table into the archive to fetch and the layer to read from it."""
    fields = dict(table)
    try:
        member = fields.pop("member")
    except KeyError:
        raise ValueError("a [boundary] must set member, naming the layer to read inside its archive") from None

    return ShapefileArchive(DataSource(**fields), member)


def _place_keys(root: Path) -> list[str]:
    return sorted(
        path.stem
        for subdirectory in (COUNTRY_SUBDIRECTORY, REGION_SUBDIRECTORY)
        for path in (root / subdirectory).glob("*.toml")
    )


def all_event_location_overrides(*, root: Path = CONFIG_ROOT) -> dict[str, tuple[float, float]]:
    """
    Collect the coordinate corrections every shipped country declares.

    Parameters
    ----------
    root : Path, optional
        Directory holding ``places/``. Default the shipped configuration.

    Returns
    -------
    overrides : dict mapping str to tuple of float
        Longitude and latitude, keyed by EM-DAT event id.
    """
    overrides: dict[str, tuple[float, float]] = {}
    for path in sorted((root / COUNTRY_SUBDIRECTORY).glob("*.toml")):
        place = read_place(path)
        if not isinstance(place, CountryConfig):
            continue

        claimed = overrides.keys() & place.event_location_overrides.keys()
        if claimed:
            raise ValueError(f"{path} re-declares {sorted(claimed)}, which another country already corrects")

        overrides.update(place.event_location_overrides)

    return overrides
