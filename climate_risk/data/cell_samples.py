import hashlib

from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

from climate_risk.data.cache import builder_fingerprint, cached, pandas_parquet
from climate_risk.geo.crs import PROJECTED_CRS
from climate_risk.geo.distance import MIN_DISTANCE_METRES, get_distance_to
from climate_risk.geo.raster import CellGrid, assign_cells_to_units, cell_coverage

CELL_SAMPLES_SUBDIRECTORY = "cell_samples"


def cell_samples_dir(cache_dir: Path) -> Path:
    """Directory the per-grid overlaps and distances are cached under."""
    return cache_dir / CELL_SAMPLES_SUBDIRECTORY


def features_fingerprint(features: gpd.GeoDataFrame | gpd.GeoSeries, columns: Sequence[str] = ()) -> str:
    """
    Fingerprint a set of features by their CRS, their geometries and the names and values of the columns a result
    carries from them.

    Parameters
    ----------
    features : GeoDataFrame or GeoSeries
        The features a grid is measured against.
    columns : sequence of str, optional
        Columns whose values reach the result. Default empty, for a result that carries geometry only.

    Returns
    -------
    fingerprint : str
        A short hex digest, the same for the same features in any process.
    """
    digest = hashlib.sha256(str(features.crs).encode())
    digest.update(b"".join(shapely.to_wkb(features.geometry.to_numpy())))
    carried = pd.DataFrame(features)
    for column in columns:
        digest.update(str(column).encode())
        digest.update(pd.util.hash_pandas_object(carried[column], index=False).to_numpy().tobytes())

    return digest.hexdigest()[:12]


def load_cell_unit_overlaps(
    grid: CellGrid, units: gpd.GeoDataFrame, cache_dir: Path, *, unit_column: str = "gid"
) -> pd.DataFrame:
    """
    Overlap between grid cells and administrative units, cached against the grid and the units.

    Parameters
    ----------
    grid : CellGrid
        The lattice and its cells.
    units : GeoDataFrame
        Administrative units, carrying ``unit_column``.
    cache_dir : Path
        Directory the caches live under.
    unit_column : str, optional
        Column naming each unit. Default 'gid'.

    Returns
    -------
    assignments : DataFrame
        One row per overlapping cell and unit, as :func:`~climate_risk.geo.raster.assign_cells_to_units` returns.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.cell_samples import load_cell_unit_overlaps
        from climate_risk.data.gadm import load_units_in_country

        districts = load_units_in_country("LAO", 2, Path("data"))
        assignments = load_cell_unit_overlaps(grid, districts, Path("data"))
    """

    def build() -> pd.DataFrame:
        return assign_cells_to_units(grid, units, unit_column=unit_column)

    return cached(
        cell_samples_dir(cache_dir),
        "unit_overlaps",
        build,
        pandas_parquet(),
        params={
            "grid": grid.fingerprint,
            "units": features_fingerprint(units, [unit_column]),
            "reading": builder_fingerprint(build, assign_cells_to_units, cell_coverage),
        },
    )


def load_cell_distances_to(
    grid: CellGrid, features: gpd.GeoDataFrame | gpd.GeoSeries, cache_dir: Path, *, return_columns: Sequence[str] = ()
) -> pd.DataFrame:
    """
    Distance from every grid cell's center to the nearest feature, cached against the grid and the features.

    Parameters
    ----------
    grid : CellGrid
        The lattice and its cells. Distances are measured from ``grid.cells``.
    features : GeoDataFrame or GeoSeries
        Features to measure to. Pass ``.boundary`` to measure to the edge of a polygon.
    cache_dir : Path
        Directory the caches live under.
    return_columns : sequence of str, optional
        Columns of ``features`` to carry across from whichever feature was nearest. Default empty.

    Returns
    -------
    distances : DataFrame
        ``distance_to_closest`` in meters plus one column per entry in ``return_columns``, in the order of ``grid.cells``.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.cell_samples import load_cell_distances_to
        from climate_risk.data_functions.rivers_data_loader import load_rivers_data

        rivers = load_rivers_data(Path("data"))
        distances = load_cell_distances_to(grid, rivers, Path("data"), return_columns=["DIS_AV_CMS"])
    """
    carried = list(return_columns)

    def build() -> pd.DataFrame:
        return get_distance_to(features, grid.cells, return_columns=carried or None).reset_index(drop=True)

    return cached(
        cell_samples_dir(cache_dir),
        "distances",
        build,
        pandas_parquet(),
        params={
            "grid": grid.fingerprint,
            "features": features_fingerprint(features, carried),
            "reading": builder_fingerprint(build, get_distance_to, PROJECTED_CRS, MIN_DISTANCE_METRES),
        },
    )
