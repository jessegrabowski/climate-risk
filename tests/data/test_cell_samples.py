import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from shapely.geometry import LineString, box

from climate_risk.data import cell_samples
from climate_risk.data.cell_samples import features_fingerprint, load_cell_distances_to, load_cell_unit_overlaps
from climate_risk.geo import raster
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


def test_a_unit_column_under_another_name_is_its_own_entry(grid, tmp_path):
    """The result is labeled with the column it was asked for, so two columns holding the same values are two
    entries, or the second caller is handed the first one's label."""
    units = halves(0.5).assign(GID_2=lambda frame: frame["gid"])

    load_cell_unit_overlaps(grid, units, tmp_path)
    relabeled = load_cell_unit_overlaps(grid, units, tmp_path, unit_column="GID_2")

    assert "GID_2" in relabeled.columns


def test_changing_how_coverage_is_measured_turns_the_cached_overlaps_over(grid, tmp_path, monkeypatch):
    """The overlaps are measured by `cell_coverage`, under the wrapper the builder calls, so a cache that sees only
    the wrapper keeps serving overlaps measured under the old rule."""
    measured = load_cell_unit_overlaps(grid, halves(0.5), tmp_path)

    def halved(*args, **kwargs):
        overlaps = measure(*args, **kwargs)
        return overlaps.assign(coverage=overlaps["coverage"] / 2)

    measure = raster.cell_coverage
    monkeypatch.setattr(raster, "cell_coverage", halved)
    monkeypatch.setattr(cell_samples, "cell_coverage", halved)
    remeasured = load_cell_unit_overlaps(grid, halves(0.5), tmp_path)

    assert remeasured["coverage"].sum() == pytest.approx(measured["coverage"].sum() / 2)
