import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from shapely.geometry import LineString, box

from climate_risk.data.cell_samples import features_fingerprint, load_cell_distances_to, load_cell_unit_overlaps
from climate_risk.geo.raster import build_cell_grid, dissolve_place_boundary


@pytest.fixture
def grid():
    """A small lattice over a one-degree square."""
    boundary = dissolve_place_boundary(
        gpd.GeoDataFrame({"ISO_A3": ["AAA"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
    )

    return build_cell_grid(boundary, resolution_km=30.0)


def halves(split_at):
    """Two units meeting at longitude ``split_at``."""
    return gpd.GeoDataFrame(
        {"gid": ["west", "east"], "geometry": [box(0, 0, split_at, 1), box(split_at, 0, 1, 1)]}, crs="EPSG:4326"
    )


def refuse(*args, **kwargs):
    raise AssertionError("measured again")


def test_overlaps_are_measured_once_per_grid_and_units(grid, tmp_path, monkeypatch):
    """The coverage pass over thousands of districts takes minutes, so a repeat reads the first one,
    and units with other boundaries are measured afresh."""
    overlaps = load_cell_unit_overlaps(grid, halves(0.5), tmp_path)

    # The pass itself is refused, since the builder's own source is part of the cache key.
    monkeypatch.setattr("climate_risk.geo.raster.exact_extract", refuse)
    pd.testing.assert_frame_equal(load_cell_unit_overlaps(grid, halves(0.5), tmp_path), overlaps)
    with pytest.raises(AssertionError, match="measured again"):
        load_cell_unit_overlaps(grid, halves(0.4), tmp_path)


def test_distances_are_measured_once_per_grid_and_features(grid, tmp_path, monkeypatch):
    """A repeat reads the first measurement in the order of the grid's cells, and a river with a
    different discharge is a different entry, since the carried column reaches the result."""
    river = gpd.GeoDataFrame({"DIS_AV_CMS": [100.0], "geometry": [LineString([(0.5, 0), (0.5, 1)])]}, crs="EPSG:4326")
    distances = load_cell_distances_to(grid, river, tmp_path, return_columns=["DIS_AV_CMS"])

    monkeypatch.setattr(gpd.GeoDataFrame, "sjoin_nearest", refuse)
    pd.testing.assert_frame_equal(
        load_cell_distances_to(grid, river, tmp_path, return_columns=["DIS_AV_CMS"]), distances
    )
    with pytest.raises(AssertionError, match="measured again"):
        load_cell_distances_to(grid, river.assign(DIS_AV_CMS=200.0), tmp_path, return_columns=["DIS_AV_CMS"])

    # The cells closest to the river sit beside the meridian it follows.
    nearest = distances["distance_to_closest"].to_numpy().argsort()[:2]
    assert np.abs(grid.cells["lon"].to_numpy()[nearest] - 0.5).max() < 0.2


def test_the_digest_ignores_columns_the_result_does_not_carry():
    """A column outside the result changing must not turn the cache over, or every reload of a
    frame with a new bookkeeping column would rebuild."""
    units = halves(0.5)

    assert features_fingerprint(units.assign(note="anything"), ["gid"]) == features_fingerprint(units, ["gid"])
    assert features_fingerprint(units.assign(gid=["a", "b"]), ["gid"]) != features_fingerprint(units, ["gid"])
