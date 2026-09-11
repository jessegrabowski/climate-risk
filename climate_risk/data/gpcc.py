import gzip
import logging
import os
import shutil

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from climate_risk.data.cache import builder_fingerprint, cached, pandas_parquet
from climate_risk.data.fetch import fetch
from climate_risk.data.source import DataSource
from climate_risk.data_functions.shapefiles_data_loader import load_shapefile
from climate_risk.geo.raster import cell_areas_km2, cell_coverage

_log = logging.getLogger(__name__)

GPCC_URL = "https://opendata.dwd.de/climate_environment/GPCC"

# The names every grid carries once read, whatever the product it came from calls them.
PRECIPITATION = "precip"
GAUGES = "gauges"

# Each is averaged onto countries over the same land weights, so each needs a running total of its
# own. Naming them here keeps the two functions that pass them between each other in step.
WEIGHTED = {PRECIPITATION: "weighted_precip", GAUGES: "weighted_gauges"}

# DWD writes the monitoring dates as YYYYMMDD floats under this unit string. CF does not define it,
# so xarray hands the axis back undecoded and it has to be read here.
DWD_DATE_UNITS = "day as %Y%m%d.%f"

FULL_DATA_START = 1891
FULL_DATA_END = 2020

# The near-real-time product carries on where the gauge analysis stops. Complete calendar years
# only: the annual totals downstream would read a part-year as a drought.
MONITORING_END = 2025
MONITORING_YEARS = tuple(range(FULL_DATA_END + 1, MONITORING_END + 1))

FULL_DATA_DECADES = tuple(f"{start}_{start + 9}" for start in range(FULL_DATA_START, FULL_DATA_END, 10))

FULL_DATA_CITATION = (
    "Schneider, U., Hänsel, S., Finger, P., Rustemeier, E., Ziese, M. (2022): GPCC Full Data "
    "Monthly Product Version 2022 at 1.0 degrees. "
    "https://doi.org/10.5676/DWD_GPCC/FD_M_V2022_100"
)

MONITORING_CITATION = (
    "Schneider, U., Becker, A., Finger, P., Rustemeier, E., Ziese, M. (2022): GPCC Monitoring "
    "Product: Near Real-Time Monthly Land-Surface Precipitation from Rain-Gauges based on SYNOP and "
    "CLIMAT data, at 1.0 degrees. https://doi.org/10.5676/DWD_GPCC/MP_M_V2022_100"
)


@dataclass(frozen=True, slots=True)
class GriddedProduct:
    """
    A GPCC product and the archives it is published in.

    Parameters
    ----------
    variable : str
        Name of the precipitation grid inside each archive.
    gauges : str
        Name of the station-count grid inside each archive, which the products spell differently.
    first_year, last_year : int
        The span the archives cover, both years included.
    sources : tuple of DataSource
        The archives making up the product, in time order.
    """

    variable: str
    gauges: str
    first_year: int
    last_year: int
    sources: tuple[DataSource, ...]


def coverage_of(products: Iterable[GriddedProduct]) -> str:
    """
    Return the span a set of products covers, as the string their cache entry is keyed on.

    Examples
    --------
    .. code-block:: python

        from climate_risk.data.gpcc import GPCC_PRODUCTS, coverage_of

        print(coverage_of(GPCC_PRODUCTS))
    """
    covered = tuple(products)

    return f"{min(product.first_year for product in covered)}-{max(product.last_year for product in covered)}"


def _full_data_archive(decade: str) -> DataSource:
    name = f"full_data_monthly_v2022_{decade}_10.nc.gz"

    return DataSource(
        url=f"{GPCC_URL}/full_data_monthly_v2022/10/{name}",
        filename=name,
        license="CC BY 4.0",
        citation=FULL_DATA_CITATION,
        retrieved="2026-08-05",
    )


def _monitoring_archive(year: int, month: int) -> DataSource:
    name = f"monitoring_v2022_10_{year}_{month:02d}.nc.gz"

    return DataSource(
        url=f"{GPCC_URL}/monitoring_v2022/{year}/{name}",
        filename=name,
        license="CC BY 4.0",
        citation=MONITORING_CITATION,
        retrieved="2026-08-07",
    )


# The reanalyzed gauge record, published a decade to an archive.
FULL_DATA = GriddedProduct(
    variable="precip",
    gauges="numgauge",
    first_year=FULL_DATA_START,
    last_year=FULL_DATA_END,
    sources=tuple(_full_data_archive(decade) for decade in FULL_DATA_DECADES),
)

# The near-real-time continuation, published a month to an archive and built from fewer stations.
MONITORING = GriddedProduct(
    variable="p",
    gauges="s",
    first_year=MONITORING_YEARS[0],
    last_year=MONITORING_YEARS[-1],
    sources=tuple(_monitoring_archive(year, month) for year in MONITORING_YEARS for month in range(1, 13)),
)

GPCC_PRODUCTS = (FULL_DATA, MONITORING)

# The archives are large enough to keep out of the top-level cache listing.
GPCC_SUBDIRECTORY = "gpcc"

WORLD_COLUMNS = {
    "ISO_A3": "country_code",
    "FORMAL_EN": "country",
    "CONTINENT": "continent",
    "REGION_UN": "region",
}


# Used when a grid holds a single cell along both axes and so states no spacing of its own. Every
# country then draws on one cell, whose size scales its weight alone and divides back out.
LONE_CELL_DEGREES = 1.0


def _axis_steps(latitudes: np.ndarray, longitudes: np.ndarray) -> tuple[float, float]:
    """
    Return the latitude and longitude spacing of an evenly spaced grid.

    Each axis takes the other's spacing where it holds a single cell, which leaves every weight on
    that axis scaled alike and so cancels out of the mean.

    Parameters
    ----------
    latitudes : ndarray
        The grid's distinct latitudes, descending.
    longitudes : ndarray
        The grid's distinct longitudes, ascending.

    Returns
    -------
    latitude_step : float
        Angular cell height, in degrees.
    longitude_step : float
        Angular cell width, in degrees.
    """

    def spacing(axis: np.ndarray, name: str) -> float | None:
        if len(axis) < 2:
            return None

        gaps = np.abs(np.diff(axis))
        # The cell a weight is measured over is placed by index, so an uneven axis would put it
        # somewhere the reading never was.
        if not np.allclose(gaps, gaps[0]):
            raise ValueError(f"The {name} axis is spaced {gaps.min()} to {gaps.max()} degrees, which is not a grid.")

        return float(gaps[0])

    latitude_step = spacing(latitudes, "latitude")
    longitude_step = spacing(longitudes, "longitude")

    if latitude_step is None:
        latitude_step = longitude_step if longitude_step is not None else LONE_CELL_DEGREES
    if longitude_step is None:
        longitude_step = latitude_step

    return latitude_step, longitude_step


def _cell_weights(cells: pd.DataFrame, countries: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Return the land area each country holds in each grid cell.

    A weight is the cell's area times the share of it lying inside the country, so a country
    smaller than one cell still gets one and a border cell is credited only with the part inside.

    Parameters
    ----------
    cells : DataFrame
        The grid's distinct cell centers, with ``lat`` and ``lon``.
    countries : GeoDataFrame
        Country boundaries carrying ``country_code``.

    Returns
    -------
    weights : DataFrame
        One row per country and cell it touches, with ``country_code``, ``lat``, ``lon`` and
        ``weight`` in square kilometers.
    """
    latitudes = np.sort(cells["lat"].unique())[::-1]
    longitudes = np.sort(cells["lon"].unique())
    latitude_step, longitude_step = _axis_steps(latitudes, longitudes)
    edges = (
        float(longitudes[0]) - longitude_step / 2,
        float(latitudes[-1]) - latitude_step / 2,
        float(longitudes[-1]) + longitude_step / 2,
        float(latitudes[0]) + latitude_step / 2,
    )

    # cell_id indexes the lattice north row first, which is the order the latitudes are sorted into.
    overlaps = cell_coverage((len(latitudes), len(longitudes)), edges, countries, "country_code")
    rows, columns = np.divmod(overlaps["cell_id"].to_numpy(), len(longitudes))
    located = overlaps.assign(lat=latitudes[rows], lon=longitudes[columns])

    area = cell_areas_km2(located["lat"].to_numpy(), longitude_step, latitude_step)

    return located.assign(weight=located["coverage"].to_numpy() * area)[["country_code", "lat", "lon", "weight"]]


def _weighted_by_country(gridded: pd.DataFrame, countries: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Reduce one grid to the weighted precipitation total and the weight behind it.

    Parameters
    ----------
    gridded : DataFrame
        One grid as read, with ``lat``, ``lon``, ``time`` and ``precip``.
    countries : GeoDataFrame
        Country boundaries carrying ``country_code``.

    Returns
    -------
    totals : DataFrame
        Indexed by ``country_code`` and ``time``, with ``weighted`` and ``weight``. The ratio is
        left to the caller, because a country's cells are spread across archives and a mean cannot
        be summed.
    """
    weights = _cell_weights(gridded[["lat", "lon"]].drop_duplicates(), countries)

    # A cell the product did not measure carries no weight either, or the mean is biased toward zero.
    reported = gridded.dropna(subset=[PRECIPITATION]).merge(weights, on=["lat", "lon"], how="inner")
    scaled = reported.assign(
        **{weighted: reported[name].astype("float64") * reported["weight"] for name, weighted in WEIGHTED.items()}
    )

    return scaled.groupby(["country_code", "time"], observed=True)[[*WEIGHTED.values(), "weight"]].sum()


def transform_gpcc(grids: Iterable[pd.DataFrame], world: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Average gridded precipitation onto countries, weighting each cell by the land it contributes.

    Every month is attributed to the boundaries ``world`` carries, which are current ones. The
    record opens in 1891, so a long series describes rainfall over a country's present-day footprint
    rather than over the country as it was. The Soviet Union, Yugoslavia and pre-split Sudan are all
    absent from the years they existed in.

    Parameters
    ----------
    grids : iterable of DataFrame
        Gridded precipitation, one frame per archive, with ``lat``, ``lon``, ``time`` and ``precip``.
    world : GeoDataFrame
        Country boundaries, carrying the columns named in ``WORLD_COLUMNS``.

    Returns
    -------
    precipitation : DataFrame
        One row per country and month, indexed by ``country_code`` and ``time``.
    """
    countries = world.rename(columns=WORLD_COLUMNS)

    # Each grid is reduced before the next is read, so the whole record is never held.
    # Totals are accumulated in double: the archives are float32, and both this mean and the annual
    # totals downstream add partial results in an order float32 has too little precision to absorb.
    totals = pd.concat([_weighted_by_country(gridded, countries) for gridded in grids], axis=0)
    summed = totals.groupby(level=["country_code", "time"], observed=True).sum()

    return pd.DataFrame(
        {name: summed[weighted] / summed["weight"] for name, weighted in WEIGHTED.items()},
        index=summed.index,
    )


def _reading_fingerprint() -> str:
    """
    Digest the rules that turn grids into country readings.

    Editing any of them changes every value in the panel, so the entry they produced has to turn
    over rather than be read back.

    Returns
    -------
    fingerprint : str
        A short digest of the transform and the weighting it applies.
    """
    return builder_fingerprint(transform_gpcc, _cell_weights, _axis_steps)


def _as_timestamps(time: xr.DataArray) -> np.ndarray:
    """
    Return an archive's time axis as timestamps, whichever way the product encoded it.

    Parameters
    ----------
    time : DataArray
        The ``time`` coordinate as read from the archive.

    Returns
    -------
    timestamps : ndarray
        The axis as ``datetime64``.
    """
    if np.issubdtype(time.dtype, np.datetime64):
        return time.values

    units = time.attrs.get("units")
    if units != DWD_DATE_UNITS:
        raise ValueError(
            f"The time axis is {time.dtype} under units {units!r}, which is neither a decoded datetime nor "
            f"the {DWD_DATE_UNITS!r} convention. Reading it as a number would date every row to 1970."
        )

    return pd.to_datetime(time.values.astype("int64").astype(str), format="%Y%m%d").values


def _extract(archive: Path) -> Path:
    """Decompress the archive beside itself if needed, and return the plain file."""
    extracted = archive.with_suffix("")

    if not extracted.exists():
        _log.info(f"Extracting {archive.name}")
        # Decompress beside the target and move, so an interrupted run leaves no truncated archive.
        partial = extracted.with_suffix(extracted.suffix + ".part")
        with gzip.open(archive, "rb") as compressed, partial.open("wb") as plain:
            shutil.copyfileobj(compressed, plain)
        os.replace(partial, extracted)

    return extracted


def _read_archive(archive: Path, variable: str, gauges: str) -> pd.DataFrame:
    """Read one archive's precipitation and station counts, renamed and with its dates decoded."""
    with xr.open_dataset(_extract(archive)) as dataset:
        grid = dataset[[variable, gauges]].rename({variable: PRECIPITATION, gauges: GAUGES})
        dated = grid.assign_coords(time=_as_timestamps(grid["time"]))

        return dated.to_dataframe().reset_index()


def load_gpcc_data(
    cache_dir: Path,
    *,
    products: tuple[GriddedProduct, ...] = GPCC_PRODUCTS,
    force_reload: bool = False,
    repair_ISO_codes: bool = True,
) -> pd.DataFrame:
    """
    Load GPCC gridded precipitation, averaged onto countries.

    GPCC publishes the record as more than one product, and every product named in ``products`` is
    read and combined into a single frame.

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under.
    products : tuple of GriddedProduct, optional
        Which published products to read. Defaults to the full and monitoring pair.
    force_reload : bool, optional
        Download again and rebuild the cache rather than reading it. Default False.
    repair_ISO_codes : bool, optional
        Correct the ISO codes on the world boundaries before the join. Default True.

    Returns
    -------
    precipitation : DataFrame
        One row per country and month, indexed by ``country_code`` and ``time``.

    Examples
    --------
    Load the record, and report the span the default products cover:

    .. code-block:: python

        from pathlib import Path

        from climate_risk import load_gpcc_data
        from climate_risk.data.gpcc import GPCC_PRODUCTS, coverage_of

        precipitation = load_gpcc_data(Path("data"))
        print(coverage_of(GPCC_PRODUCTS))
    """

    def build() -> pd.DataFrame:
        archives = [
            (fetch(source, cache_dir / GPCC_SUBDIRECTORY, force=force_reload), product.variable, product.gauges)
            for product in products
            for source in product.sources
        ]
        world = load_shapefile("world", cache_dir, repair_ISO_codes=repair_ISO_codes)

        return transform_gpcc(
            (_read_archive(archive, variable, gauges) for archive, variable, gauges in archives), world
        )

    return cached(
        cache_dir,
        "gpcc",
        build,
        pandas_parquet(),
        params={
            "repaired_iso": repair_ISO_codes,
            "coverage": coverage_of(products),
            "precision": "float64",
            "reading": _reading_fingerprint(),
        },
        force=force_reload,
    )
