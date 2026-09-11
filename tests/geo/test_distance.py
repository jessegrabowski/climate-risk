import geopandas as gpd
import numpy as np
import pytest

from shapely.geometry import LineString, box

from climate_risk.exceptions import DataValidationError
from climate_risk.geo.distance import get_distance_to


def test_distance_is_measured_in_meters_of_the_projected_crs(grid_points):
    """`to_km` and the log transforms downstream only hold while the result is in meters. Measuring
    in the points' own lat/lon would return half a degree here and read as half a meter.
    """
    rivers = gpd.GeoDataFrame(
        {"ORD_FLOW": [4], "HYRIV_ID": [1], "geometry": [LineString([(0, -1), (0, 2)])]}, crs="EPSG:4326"
    )

    distances = get_distance_to(rivers, points=grid_points)["distance_to_closest"]

    # The fixture sits half a degree and two and a half degrees east of the river, on the equator,
    # where a degree of longitude is about 111.3 km in the projected CRS.
    assert distances.iloc[0] == pytest.approx(55_660, rel=0.01)
    assert distances.iloc[1] == pytest.approx(278_299, rel=0.01)


def test_requested_columns_come_from_the_nearest_feature(grid_points):
    far = LineString([(50, 50), (50, 51)])
    near = LineString([(0, -1), (0, 2)])
    rivers = gpd.GeoDataFrame({"ORD_FLOW": [9, 4], "HYRIV_ID": [99, 1], "geometry": [far, near]}, crs="EPSG:4326")

    nearest = get_distance_to(rivers, points=grid_points, return_columns=["ORD_FLOW", "HYRIV_ID"])

    assert nearest["HYRIV_ID"].tolist() == [1, 1]
    assert nearest["ORD_FLOW"].tolist() == [4, 4]


def test_a_point_equidistant_from_two_features_gets_one_row():
    """A nearest join reports every tied feature, so a point on the midline between two rivers
    matches both. One row per point in, one row out, or the caller's frame silently grows and
    stops aligning with the grid it was built from.
    """
    rivers = gpd.GeoDataFrame(
        {
            "ORD_FLOW": [4, 4],
            "HYRIV_ID": [1, 2],
            "geometry": [LineString([(-1, -1), (-1, 1)]), LineString([(1, -1), (1, 1)])],
        },
        crs="EPSG:4326",
    )
    midline = gpd.GeoDataFrame(geometry=gpd.points_from_xy([0.0, 0.0], [0.0, 0.5]), crs="EPSG:4326")

    nearest = get_distance_to(rivers, points=midline, return_columns=["HYRIV_ID"])

    assert len(nearest) == len(midline)
    assert nearest["HYRIV_ID"].isin([1, 2]).all(), "one of the two tied rivers, not a missing match"
    # Either tie is a degree away, so the distance is the same whichever was kept.
    np.testing.assert_allclose(nearest["distance_to_closest"].to_numpy(), 111_319, rtol=0.01)


def test_a_geoseries_of_boundaries_is_measured_to():
    """`create_grid_from_shape` passes `coastline.boundary`, which is a GeoSeries rather than a
    frame, so it carries no columns to join on. Measuring to the boundary rather than the polygon
    is what makes a point inside the landmass a positive distance from the coast.
    """
    land = gpd.GeoDataFrame(geometry=[box(-2, -2, 2, 2)], crs="EPSG:4326")
    inside = gpd.GeoDataFrame(geometry=gpd.points_from_xy([0.0], [0.0]), crs="EPSG:4326")

    to_coast = get_distance_to(land.boundary, points=inside)["distance_to_closest"]

    assert to_coast.iloc[0] > 100_000, "the center of the box is far from its edge"
    assert get_distance_to(land, points=inside)["distance_to_closest"].iloc[0] == 0.0


def test_the_result_is_indexed_like_the_points():
    """The caller merges this back onto the grid by index, so a reset index would misalign every
    distance by however many rows the grid had already dropped.
    """
    rivers = gpd.GeoDataFrame(
        {"ORD_FLOW": [4], "HYRIV_ID": [1], "geometry": [LineString([(0, -1), (0, 2)])]}, crs="EPSG:4326"
    )
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy([1.0, 2.0], [1.0, 1.0]), crs="EPSG:4326", index=[17, 42])

    nearest = get_distance_to(rivers, points=points, return_columns=["HYRIV_ID"])

    assert nearest.index.tolist() == [17, 42]
    assert nearest["distance_to_closest"].idxmin() == 17, "the nearer point keeps its own label"


def test_points_sharing_an_index_label_each_keep_their_own_distance():
    """The nearest join reports ties as extra rows, and the index is how they are collapsed. If
    that runs on the caller's labels rather than on position, two distinct points that happen to
    share a label look like one point matched twice, and one of them is silently discarded.
    """
    rivers = gpd.GeoDataFrame({"HYRIV_ID": [1], "geometry": [LineString([(0, -1), (0, 2)])]}, crs="EPSG:4326")
    shared_label = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([1.0, 5.0, 9.0], [1.0, 1.0, 1.0]), crs="EPSG:4326", index=[7, 7, 8]
    )

    nearest = get_distance_to(rivers, points=shared_label, return_columns=["HYRIV_ID"])

    assert len(nearest) == len(shared_label)
    assert nearest["distance_to_closest"].is_monotonic_increasing, "each point keeps its own distance"


def test_features_whose_geometry_column_is_named_something_else_are_measured_to():
    """A GeoDataFrame's active geometry need not be called `geometry`, and upstream archives are
    read as published. Selecting the column by name would drop the geometry and fail the join.
    """
    rivers = gpd.GeoDataFrame(
        {"HYRIV_ID": [1], "geometry": [LineString([(0, -1), (0, 2)])]}, crs="EPSG:4326"
    ).rename_geometry("geom")
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy([1.0], [1.0]), crs="EPSG:4326")

    nearest = get_distance_to(rivers, points=points, return_columns=["HYRIV_ID"])

    assert nearest["HYRIV_ID"].tolist() == [1]
    assert nearest["distance_to_closest"].iloc[0] > 10_000


def test_measuring_to_nothing_is_refused():
    """A place whose river or coastline query returns nothing reaches here as an empty frame. The
    nearest join answers NaN for every point, which survives the clip, becomes NaN through the log,
    and shows up as a column of missing covariates rather than as a failure.
    """
    nothing = gpd.GeoDataFrame({"HYRIV_ID": [], "geometry": []}, crs="EPSG:4326")
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy([1.0], [1.0]), crs="EPSG:4326")

    with pytest.raises(DataValidationError, match="no features to measure to"):
        get_distance_to(nothing, points=points, return_columns=["HYRIV_ID"])
