from collections.abc import Mapping
from dataclasses import dataclass, field

from climate_risk.data.source import ShapefileArchive
from climate_risk.geo.crs import GEOGRAPHIC_CRS, PROJECTED_CRS

# An ISO 3166-1 alpha-3 code, which is what every frame in the project keys countries on.
ISO3_LENGTH = 3


def _validate_iso3(code: str, described_as: str) -> None:
    if len(code) != ISO3_LENGTH or not code.isalpha() or not code.isupper():
        raise ValueError(f"{described_as} must be an ISO 3166-1 alpha-3 code, got {code!r}")


@dataclass(frozen=True, slots=True)
class GeometrySpec:
    """
    How a place is projected.

    Parameters
    ----------
    geographic_crs : str
        The lat/lon CRS points are built in. Default ``EPSG:4326``.
    projected_crs : str
        The CRS distances are measured in, which must be metric. Default ``EPSG:3395``.
    """

    geographic_crs: str = GEOGRAPHIC_CRS
    projected_crs: str = PROJECTED_CRS


@dataclass(frozen=True, slots=True)
class EventFilters:
    """
    Which EM-DAT records a place's panel counts.

    Parameters
    ----------
    start_year : int
        First year of the study window, included. Default 1981.
    end_year : int, optional
        Last year, included. Default None, meaning the newest event in the workbook.
    min_total_affected : int, optional
        An event must affect more than this many people to count. Default None, which counts every
        event, including one whose reach was never recorded.
    min_deaths : int, optional
        An event must kill more than this many people to count. Default None, which counts an event
        on its reach alone.
    """

    start_year: int = 1981
    end_year: int | None = None
    min_total_affected: int | None = None
    min_deaths: int | None = None

    def __post_init__(self) -> None:
        if self.end_year is not None and self.end_year < self.start_year:
            raise ValueError(f"the window {self.start_year}-{self.end_year} ends before it starts")


@dataclass(frozen=True, slots=True)
class CountryConfig:
    """
    One country, and everything about it the library would otherwise hardcode.

    Parameters
    ----------
    iso3 : str
        ISO 3166-1 alpha-3 code, which every frame keys on.
    name : str
        Human-readable name, used in figures and messages.
    island : bool
        Whether the country is an island, which the point features record. Default False.
    boundary : ShapefileArchive, optional
        A country-specific boundary archive, and the layer within it to read. Default None, meaning
        slice the world shapefile by ``iso3``.
    geometry : GeometrySpec
        Projection settings. Defaults to the project-wide ones.
    events : EventFilters
        Which EM-DAT records count. Defaults to the project-wide thresholds.
    event_location_overrides : mapping of str to tuple of float
        Longitude and latitude to force onto specific EM-DAT event ids, for records whose published
        coordinates are wrong. Default empty.
    """

    iso3: str
    name: str
    island: bool = False
    boundary: ShapefileArchive | None = None
    geometry: GeometrySpec = field(default_factory=GeometrySpec)
    events: EventFilters = field(default_factory=EventFilters)
    event_location_overrides: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_iso3(self.iso3, "iso3")


@dataclass(frozen=True, slots=True)
class RegionConfig:
    """
    A group of countries analyzed together.

    Parameters
    ----------
    key : str
        Short identifier the config is filed and cached under.
    name : str
        Human-readable name.
    members : tuple of str
        The ISO 3166-1 alpha-3 codes belonging to the region.
    geometry : GeometrySpec
        Projection settings. Defaults to the project-wide ones.
    events : EventFilters
        Which EM-DAT records count. Defaults to the project-wide thresholds.
    """

    key: str
    name: str
    members: tuple[str, ...]
    geometry: GeometrySpec = field(default_factory=GeometrySpec)
    events: EventFilters = field(default_factory=EventFilters)

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError(f"region {self.key!r} has no members")

        for code in self.members:
            _validate_iso3(code, f"region {self.key!r} member")

        repeated = sorted({code for code in self.members if self.members.count(code) > 1})
        if repeated:
            raise ValueError(f"region {self.key!r} repeats {repeated}, which would double-count them")


Place = CountryConfig | RegionConfig
