import geopandas as gpd
import numpy as np
import pandas as pd

from climate_risk.geo.crs import GEOGRAPHIC_CRS
from climate_risk.geo.distance import MIN_DISTANCE_METRES, get_distance_to


def create_grid_from_shape(
    shapefile: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    coastline: gpd.GeoDataFrame,
    grid_size: int = 100,
    *,
    iso3: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Lay a regular grid over the bounds of ``shapefile`` and measure each point to river and coast.

    Parameters
    ----------
    shapefile : GeoDataFrame
        Region to cover. Points falling outside its geometry are dropped, and an ``ISO_A3`` column,
        if present, labels the survivors.
    rivers : GeoDataFrame
        River geometries, carrying ``ORD_FLOW`` and ``HYRIV_ID``.
    coastline : GeoDataFrame
        Coastline geometries. Distance is measured to the boundary.
    grid_size : int, optional
        Points per axis, so the grid holds ``grid_size ** 2`` before clipping. Default 100.
    iso3 : str, optional
        Code to label every point with when ``shapefile`` carries no ``ISO_A3`` column. Default
        None, which requires the geometry to label itself.

    Returns
    -------
    grid : GeoDataFrame
        One row per surviving point, with ``lon``, ``lat``, distances to the nearest river and
        coastline, their logs, and ``ISO``.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk import load_rivers_data, load_shapefile
        from climate_risk.geo.grid import create_grid_from_shape

        cache_dir = Path("data")
        world = load_shapefile("world", cache_dir)
        coastline = load_shapefile("coastline", cache_dir)
        rivers = load_rivers_data(cache_dir)

        grid = create_grid_from_shape(world[world["ISO_A3"] == "LAO"], rivers, coastline)
    """
    labeled = "ISO_A3" in shapefile.columns
    if not labeled and iso3 is None:
        raise ValueError(
            "The geometry carries no ISO_A3 column, so pass iso3 to say which country its points belong to."
        )

    long_min, lat_min, long_max, lat_max = shapefile.dissolve().bounds.values.ravel()
    long_grid = np.linspace(long_min, long_max, grid_size)
    lat_grid = np.linspace(lat_min, lat_max, grid_size)

    mesh = np.column_stack([axis.ravel() for axis in np.meshgrid(long_grid, lat_grid)])
    grid = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*mesh.T), crs=GEOGRAPHIC_CRS)

    point_overlay = grid.overlay(shapefile, how="intersection")
    points = gpd.GeoDataFrame(
        point_overlay.geometry.to_frame().assign(lon=lambda x: x.geometry.x, lat=lambda x: x.geometry.y),
        crs=GEOGRAPHIC_CRS,
    )

    distances_to_rivers = get_distance_to(rivers, points=points, return_columns=["ORD_FLOW", "HYRIV_ID"]).rename(
        columns={"distance_to_closest": "distance_to_river"}
    )

    points = gpd.GeoDataFrame(pd.merge(points, distances_to_rivers, left_index=True, right_index=True, how="left"))

    distances_to_coastlines = get_distance_to(coastline.boundary, points=points, return_columns=None).rename(
        columns={"distance_to_closest": "distance_to_coastline"}
    )

    points = gpd.GeoDataFrame(pd.merge(points, distances_to_coastlines, left_index=True, right_index=True, how="left"))

    points = gpd.GeoDataFrame(
        points.assign(
            log_distance_to_river=lambda x: np.log(x.distance_to_river.clip(lower=MIN_DISTANCE_METRES)),
            log_distance_to_coastline=lambda x: np.log(x.distance_to_coastline.clip(lower=MIN_DISTANCE_METRES)),
        )
    )

    points["ISO"] = point_overlay["ISO_A3"] if labeled else iso3

    return points
