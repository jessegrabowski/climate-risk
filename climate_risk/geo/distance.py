import geopandas as gpd
import pandas as pd

from climate_risk.exceptions import DataValidationError
from climate_risk.geo.crs import PROJECTED_CRS

# Distances come back in meters, and a point sitting on a feature measures zero, which has no log.
MIN_DISTANCE_METRES = 1.0


def get_distance_to(
    gdf: gpd.GeoDataFrame | gpd.GeoSeries,
    points: gpd.GeoDataFrame,
    return_columns: list[str] | None = None,
    crs: str = PROJECTED_CRS,
) -> pd.DataFrame:
    """
    Measure the distance from every point to the nearest feature in ``gdf``.

    Parameters
    ----------
    gdf : GeoDataFrame or GeoSeries
        Features to measure to. Pass ``.boundary`` to measure to the edge of a polygon rather than
        into its interior.
    points : GeoDataFrame
        Points to measure from. The result carries this index.
    return_columns : list of str, optional
        Columns of ``gdf`` to carry across from whichever feature was nearest. Default None, which
        returns the distance alone.
    crs : str, optional
        Projected CRS to measure in, which fixes the units of the result. Default
        ``PROJECTED_CRS``, giving meters.

    Returns
    -------
    distances : DataFrame
        ``distance_to_closest`` plus one column per entry in ``return_columns``, indexed like
        ``points``.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk import load_rivers_data
        from climate_risk.geo.distance import get_distance_to

        rivers = load_rivers_data(Path("data"))
        distances = get_distance_to(rivers, grid_points)
    """
    carried = list(return_columns or [])

    if gdf.empty:
        raise DataValidationError(
            "There are no features to measure to, so every distance would come back as NaN and poison whatever "
            "it is joined onto."
        )

    wanted = gdf if isinstance(gdf, gpd.GeoSeries) else gdf[[*carried, gdf.geometry.name]]
    features = gpd.GeoDataFrame(geometry=wanted) if isinstance(wanted, gpd.GeoSeries) else wanted
    features = features.to_crs(crs).reset_index(drop=True)

    # Only the geometry crosses the join, so a column shared with the features cannot collide, and
    # the index is positional so that a repeated label in `points` is not mistaken for a tie below.
    origins = gpd.GeoDataFrame(geometry=points.geometry.to_numpy(), crs=points.crs).to_crs(crs)

    matched = origins.sjoin_nearest(features, how="left", distance_col="distance_to_closest")
    # A point equidistant from two features matches both; the first is as good as either.
    nearest = matched[~matched.index.duplicated(keep="first")]

    return pd.DataFrame(
        {column: nearest[column].to_numpy() for column in ("distance_to_closest", *carried)}, index=points.index
    )
