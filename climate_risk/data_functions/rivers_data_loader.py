import logging

from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd

from climate_risk.data.cache import cached, geo_parquet
from climate_risk.data.fetch import fetch
from climate_risk.data.source import DataSource

_log = logging.getLogger(__name__)

RIVERS = DataSource(
    url="https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_shp.zip",
    filename="HydroRIVERS_v10_shp.zip",
    license="HydroSHEDS License Agreement: free for scientific, educational and commercial use",
    citation=(
        "Lehner, B., Grill, G. (2013): Global river hydrography and network routing: baseline data "
        "and new approaches to study the world's large river systems. Hydrological Processes, "
        "27(15), 2171-2186. https://doi.org/10.1002/hyp.9740"
    ),
    retrieved="2026-08-08",
)

# The archive unpacks into a directory named for itself.
RIVERS_MEMBER = "HydroRIVERS_v10_shp/HydroRIVERS_v10.shp"

RIVERS_SUBDIRECTORY = "rivers"

# ORD_FLOW is a discharge class, not a topological order: class 1 is every reach at or above
# 100,000 m3/s and each class down is a factor of ten, so a lower cutoff keeps fewer, bigger rivers.
# The panel uses the major rivers alone; the wider set exists for sensitivity checks.
BIG_RIVER_ORDER = 5
MEDIUM_RIVER_ORDER = 6


def transform_rivers(rivers: gpd.GeoDataFrame, stream_order_cutoff: int) -> gpd.GeoDataFrame:
    """
    Keep the rivers whose discharge class is below the cutoff.

    Parameters
    ----------
    rivers : GeoDataFrame
        The HydroRIVERS network, carrying an ``ORD_FLOW`` discharge class.
    stream_order_cutoff : int
        The exclusive upper bound on ``ORD_FLOW``. The class counts down from the largest discharge,
        so a lower cutoff keeps fewer and bigger rivers.

    Returns
    -------
    kept : GeoDataFrame
        The rivers that clear the cutoff.
    """
    return rivers.query(f"ORD_FLOW < {stream_order_cutoff}")


def _extract_rivers(cache_dir: Path) -> Path:
    """Unpack the archive unless the network is already on disk, and return the shapefile."""
    directory = cache_dir / RIVERS_SUBDIRECTORY
    extracted = directory / RIVERS_MEMBER

    if not extracted.exists():
        _log.info(f"Extracting {RIVERS.filename}")
        with ZipFile(fetch(RIVERS, directory)) as archive:
            archive.extractall(path=directory)

    return extracted


def load_rivers_data(cache_dir: Path, *, include_medium: bool = False) -> gpd.GeoDataFrame:
    """
    Load HydroRIVERS, keeping the larger rivers.

    Rivers are filtered on ``ORD_FLOW``, the long-term average discharge class, so the result holds
    the major channels rather than every mapped tributary. Strahler order is a separate column and
    is not what this reads.

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under.
    include_medium : bool, optional
        Widen the cutoff to keep reaches down to 10 cubic meters per second as well, rather than
        stopping at 100. Default False.

    Returns
    -------
    rivers : GeoDataFrame
        One row per river reach, with its geometry.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk import load_rivers_data

        rivers = load_rivers_data(Path("data"), include_medium=True)
    """
    cutoff = MEDIUM_RIVER_ORDER if include_medium else BIG_RIVER_ORDER

    def build() -> gpd.GeoDataFrame:
        return transform_rivers(gpd.read_file(_extract_rivers(cache_dir)), cutoff)

    return cached(
        cache_dir / RIVERS_SUBDIRECTORY,
        "rivers",
        build,
        geo_parquet(),
        params={"stream_order_below": cutoff},
    )
