import dataclasses

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import shapely

from pyproj import Geod
from rasterio.transform import from_bounds
from shapely.geometry import box

from climate_risk.exceptions import DataValidationError
from climate_risk.geo.crs import GEOGRAPHIC_CRS
from climate_risk.geo.raster import (
    KM_PER_DEGREE_LATITUDE,
    MIN_CELLS_PER_AXIS,
    assign_cells_to_units,
    build_cell_grid,
    dissolve_place_boundary,
    grid_axes,
    sample_onto_cells,
)


def tiles(bounds, *, crs="EPSG:4326", **columns):
    """A boundary split into several features, as a country's own administrative archive arrives."""
    return gpd.GeoDataFrame({**columns, "geometry": [box(*corner) for corner in bounds]}, crs=crs)


def test_both_boundary_sources_dissolve_to_the_same_geometry():
    """The whole point of dissolving. A country's archive arrives as many districts with no ISO
    column while the world slice arrives as one labeled feature, and the two paths must grid
    identically or `lao` and `sea` silently disagree about where Laos is.
    """
    archive = tiles([(0, 0, 1, 1), (1, 0, 2, 1), (0, 1, 1, 2), (1, 1, 2, 2)], adm2_name=list("abcd"))
    world_slice = tiles([(0, 0, 2, 2)], ISO_A3=["LAO"])

    from_archive = dissolve_place_boundary(archive, iso3="LAO")
    from_world = dissolve_place_boundary(world_slice)

    assert list(from_archive.columns) == list(from_world.columns) == ["ISO_A3", "geometry"]
    assert from_archive["ISO_A3"].tolist() == from_world["ISO_A3"].tolist() == ["LAO"]
    assert from_archive.geometry.iloc[0].equals(from_world.geometry.iloc[0])


def test_a_region_keeps_one_row_per_country():
    """A region spans several countries and each keeps its own row, because the grid labels cells
    by country. Dissolving the lot into one polygon would erase that.
    """
    region = tiles([(0, 0, 1, 1), (1, 0, 2, 1), (5, 5, 6, 6)], ISO_A3=["LAO", "LAO", "THA"])

    dissolved = dissolve_place_boundary(region)

    assert dissolved["ISO_A3"].tolist() == ["LAO", "THA"]
    assert dissolved.loc[dissolved["ISO_A3"] == "LAO"].geometry.iloc[0].equals(box(0, 0, 2, 1))


def test_a_fallback_code_the_boundary_disagrees_with_is_refused():
    """A geometry that labels itself keeps its own codes, so stamping the caller's over it would
    relabel every country as one. Returning the boundary unchanged instead hands back a result that
    is not the country asked for, and gridding it lays a lattice over all of them.
    """
    with pytest.raises(DataValidationError, match="iso3='LAO'"):
        dissolve_place_boundary(tiles([(0, 0, 1, 1)], ISO_A3=["THA"]), iso3="LAO")


def test_a_fallback_code_the_boundary_agrees_with_is_allowed():
    """A place resolves either to its own archive, which carries no code, or to a world slice, which
    does. A caller handling both passes iso3 either way, and it is right both times.
    """
    dissolved = dissolve_place_boundary(tiles([(0, 0, 1, 1)], ISO_A3=["LAO"]), iso3="LAO")

    assert dissolved["ISO_A3"].tolist() == ["LAO"]


def test_a_fallback_code_carried_alongside_others_is_refused():
    """Carrying the code asked for is not enough, because the rest are dissolved along with it. This
    is the world shapefile passed whole, which yields one geometry spanning every country in it.
    """
    several = tiles([(0, 0, 1, 1), (5, 5, 6, 6)], ISO_A3=["THA", "LAO"])

    with pytest.raises(DataValidationError, match="carries 2 codes"):
        dissolve_place_boundary(several, iso3="LAO")


def test_the_boundary_comes_back_in_the_geographic_crs():
    """Archives arrive in whatever CRS they were published in, and the grid is laid in degrees.
    Gridding a projected boundary would put the cells in meters and space them wrongly.
    """
    projected = tiles([(0, 0, 1, 1)], crs="EPSG:3395", ISO_A3=["LAO"])

    dissolved = dissolve_place_boundary(projected)

    # A one-meter square on the equator spans about 9e-6 degrees, so untouched coordinates would
    # come back a hundred thousand times wider than this.
    assert dissolved.crs == GEOGRAPHIC_CRS
    assert dissolved.total_bounds[2] < 1e-4


def test_a_boundary_with_no_crs_is_rejected():
    """Assuming lat/lon for an unlabeled CRS would place a projected archive off the coast of
    Africa without complaint.
    """
    unlocated = gpd.GeoDataFrame({"ISO_A3": ["LAO"], "geometry": [box(0, 0, 1, 1)]})

    with pytest.raises(DataValidationError, match="no CRS"):
        dissolve_place_boundary(unlocated)


def test_an_unlabeled_boundary_with_no_code_is_rejected():
    with pytest.raises(DataValidationError, match="pass iso3"):
        dissolve_place_boundary(tiles([(0, 0, 1, 1)]))


def test_a_square_degree_on_the_equator_grids_squarely():
    """On the equator a degree of longitude and a degree of latitude are the same distance, so the
    two axes must come back with the same count. Any factor missing from the conversion shows up
    as a difference here.
    """
    longitudes, latitudes = grid_axes((0.0, -0.5, 1.0, 0.5), resolution_km=KM_PER_DEGREE_LATITUDE / 4.0)

    assert len(longitudes) == len(latitudes) == 4
    np.testing.assert_allclose(longitudes, [0.125, 0.375, 0.625, 0.875])


def test_the_axes_are_cell_centers_that_tile_the_extent():
    """The operator sums one sample per cell, which is a midpoint rule. Returning the extent's
    edges instead would put half of the first and last cells outside the place, and the weights
    would claim ground the grid does not cover.
    """
    longitudes, _ = grid_axes((0.0, 0.0, 1.0, 1.0), resolution_km=KM_PER_DEGREE_LATITUDE / 4.0)
    step = longitudes[1] - longitudes[0]

    assert longitudes[0] == pytest.approx(step / 2.0)
    assert longitudes[-1] == pytest.approx(1.0 - step / 2.0)


def test_longitude_needs_fewer_cells_away_from_the_equator():
    """A degree of longitude at 60N is half a degree at the equator, so the same degree span needs
    about half the cells. Spacing both axes by degrees would make every high-latitude cell twice
    as wide as it is tall, and the weights would inherit that.
    """
    longitudes, latitudes = grid_axes((0.0, 59.5, 1.0, 60.5), resolution_km=10.0)

    assert len(longitudes) == pytest.approx(len(latitudes) / 2, abs=1)
    assert len(longitudes) < len(latitudes)


def test_cells_are_square_in_the_middle_of_a_tall_extent():
    """Longitude is converted at the extent's mean latitude, so a place spanning many degrees gets
    square cells at its center. Converting at an edge instead skews every cell across the whole
    grid, and the error grows with how tall the place is.
    """
    resolution = 50.0
    longitudes, latitudes = grid_axes((0.0, 50.0, 10.0, 70.0), resolution_km=resolution)

    middle = np.cos(np.radians(60.0))
    longitude_step_km = (longitudes[1] - longitudes[0]) * KM_PER_DEGREE_LATITUDE * middle
    latitude_step_km = (latitudes[1] - latitudes[0]) * KM_PER_DEGREE_LATITUDE

    assert longitude_step_km == pytest.approx(resolution, rel=0.1)
    assert latitude_step_km == pytest.approx(resolution, rel=0.1)


def test_the_cell_count_scales_with_the_resolution_asked_for():
    """Five times finer is five times as many cells along an axis. A conversion that ignored the
    argument, or applied it as a count rather than a distance, would break this ratio.
    """
    coarse, _ = grid_axes((0.0, 0.0, 10.0, 10.0), resolution_km=50.0)
    fine, _ = grid_axes((0.0, 0.0, 10.0, 10.0), resolution_km=10.0)

    assert len(fine) == pytest.approx(5 * len(coarse), rel=0.02)


def test_an_extent_smaller_than_a_cell_becomes_one_cell():
    """A place smaller than the requested resolution is one cell sampled at its middle, not zero
    cells and not a degenerate axis.
    """
    longitudes, latitudes = grid_axes((0.0, 0.0, 0.01, 0.01), resolution_km=100.0)

    assert len(longitudes) == len(latitudes) == MIN_CELLS_PER_AXIS
    assert longitudes[0] == pytest.approx(0.005)


@pytest.mark.parametrize("resolution", [0.0, -5.0])
def test_a_non_positive_resolution_is_rejected(resolution):
    with pytest.raises(DataValidationError, match="positive distance"):
        grid_axes((0.0, 0.0, 1.0, 1.0), resolution_km=resolution)


def test_inverted_bounds_are_rejected():
    """A transposed bounds tuple would otherwise give a negative span, one point per axis, and a
    grid silently covering nothing.
    """
    with pytest.raises(DataValidationError, match="inverted"):
        grid_axes((1.0, 0.0, 0.0, 1.0), resolution_km=10.0)


def test_a_cell_the_place_misses_entirely_is_dropped():
    """The grid is laid over a rectangular extent, so an L-shaped place has cells in the notch that
    belong to nobody. Keeping them would put ground in the model that the place does not cover.
    """
    boundary = dissolve_place_boundary(tiles([(0, 0, 2, 1), (0, 1, 1, 2)], ISO_A3=["LAO", "LAO"]))

    cells = build_cell_grid(boundary, resolution_km=20.0).cells
    deep_in_the_notch = (cells["lon"] > 1.2) & (cells["lat"] > 1.2)

    assert len(cells) > 0
    assert not deep_in_the_notch.any()


def test_a_cell_on_the_frontier_is_kept_with_the_part_inside():
    """Dropping a straddling cell takes its overlaps with the border units too, and a country is
    mostly border at any resolution coarse enough to model. It is kept, credited with the ground
    the place actually covers.
    """
    boundary = dissolve_place_boundary(tiles([(0, 0, 2, 1), (0, 1, 1, 2)], ISO_A3=["LAO", "LAO"]))

    cells = build_cell_grid(boundary, resolution_km=20.0).cells
    partial = cells["place_area_km2"] < cells["cell_area_km2"] - 1e-9

    assert partial.any(), "the notch and the outline should both cut cells"
    assert (cells["place_area_km2"] > 0.0).all()
    assert (cells["place_area_km2"] <= cells["cell_area_km2"] + 1e-9).all()


def test_the_covered_area_adds_up_to_the_place():
    """The reason for coverage over center-in-polygon. Summed over cells, the covered area is the
    place's own area; clipping by center loses whatever the border cells were carrying, which for
    a ragged outline is percent-scale rather than rounding.
    """
    outline = box(0, 0, 2, 1).union(box(0, 1, 1, 2))
    boundary = dissolve_place_boundary(gpd.GeoDataFrame({"ISO_A3": ["LAO"], "geometry": [outline]}, crs=GEOGRAPHIC_CRS))
    exact = abs(Geod(ellps="WGS84").geometry_area_perimeter(outline)[0]) / 1e6

    cells = build_cell_grid(boundary, resolution_km=20.0).cells

    assert cells["place_area_km2"].sum() == pytest.approx(exact, rel=1e-3)
    assert cells["cell_area_km2"].sum() > exact, "the lattice covers the bounding box, not the place"


def test_each_cell_takes_the_country_it_lands_in():
    """Cells are labeled by the polygon they fall in, not by the place as a whole, because a
    region spans several countries and the model reports by country.
    """
    boundary = dissolve_place_boundary(tiles([(0, 0, 1, 1), (3, 0, 4, 1)], ISO_A3=["LAO", "THA"]))

    cells = build_cell_grid(boundary, resolution_km=20.0).cells

    assert set(cells["ISO_A3"]) == {"LAO", "THA"}
    assert cells.loc[cells["lon"] < 2.0, "ISO_A3"].eq("LAO").all()
    assert cells.loc[cells["lon"] > 2.0, "ISO_A3"].eq("THA").all()


def test_the_cell_areas_sum_to_the_area_of_the_extent():
    """The weights are what the operator integrates with, so they have to add up to real ground.
    A lon/lat box has a closed-form area, and the band runs to 60N so that treating cells as
    equal-area would overshoot it by a fifth rather than by a rounding error.
    """
    boundary = dissolve_place_boundary(tiles([(0, 0, 1, 60)], ISO_A3=["LAO"]))
    extent = box(0.0, 0.0, 1.0, 60.0)
    exact = abs(Geod(ellps="WGS84").geometry_area_perimeter(extent)[0]) / 1e6
    flat = KM_PER_DEGREE_LATITUDE**2 * 60.0

    cells = build_cell_grid(boundary, resolution_km=25.0).cells

    assert cells["cell_area_km2"].sum() == pytest.approx(exact, rel=1e-3)
    assert exact < flat / 1.2


def test_cells_shrink_towards_the_pole():
    """Equal angular cells cover less ground at high latitude. Treating them as equal-area would
    overweight the top of a tall place in every total that spans it.
    """
    boundary = dissolve_place_boundary(tiles([(0, 0, 1, 60)], ISO_A3=["LAO"]))

    cells = build_cell_grid(boundary, resolution_km=100.0).cells
    southern = cells.loc[cells["lat"] < 5.0, "cell_area_km2"].mean()
    northern = cells.loc[cells["lat"] > 55.0, "cell_area_km2"].mean()

    assert northern < southern / 1.5


def test_a_boundary_that_was_not_dissolved_is_rejected():
    """Gridding a raw archive would work and label nothing, so the grid would carry no country."""
    with pytest.raises(DataValidationError, match="dissolve_place_boundary"):
        build_cell_grid(tiles([(0, 0, 1, 1)], adm2_name=["a"]), resolution_km=20.0)


def test_an_empty_boundary_is_rejected():
    """An ISO code matching nothing in the world file reaches here as an empty frame, and an empty
    grid is a confusing way to find out.
    """
    with pytest.raises(DataValidationError, match="nothing to grid"):
        dissolve_place_boundary(gpd.GeoDataFrame({"ISO_A3": [], "geometry": []}, crs="EPSG:4326"))


def test_the_lattice_index_survives_clipping():
    """`cell_id` indexes the full lattice, not the surviving rows, because the zonal pass runs over
    the whole raster. Renumbering after the clip would silently shift every assignment.
    """
    boundary = dissolve_place_boundary(tiles([(0, 0, 2, 1), (0, 1, 1, 2)], ISO_A3=["LAO", "LAO"]))

    grid = build_cell_grid(boundary, resolution_km=40.0)
    rows, columns = grid.shape

    # Rebuild the index from each cell's position in the lattice, independently of how it was set.
    row_of_cell = np.searchsorted(-grid.latitudes, -grid.cells["lat"].to_numpy())
    column_of_cell = np.searchsorted(grid.longitudes, grid.cells["lon"].to_numpy())

    assert len(grid.cells) < rows * columns, "the notch should have cost some cells"
    np.testing.assert_array_equal(grid.cells["cell_id"].to_numpy(), row_of_cell * columns + column_of_cell)


def test_the_lattice_rows_run_north_to_south():
    """Raster order, so `cell_id` lines up with what a zonal pass reports. Ascending latitudes
    would flip the grid vertically and put every unit's cells in the wrong hemisphere of the place.
    """
    boundary = dissolve_place_boundary(tiles([(0, 0, 1, 1)], ISO_A3=["LAO"]))

    grid = build_cell_grid(boundary, resolution_km=20.0)

    assert grid.latitudes[0] > grid.latitudes[-1]
    northernmost = grid.cells.loc[grid.cells["cell_id"].idxmin(), "lat"]
    assert northernmost == pytest.approx(grid.latitudes[0])


def test_a_cell_on_a_border_is_shared_between_units():
    """The reason for area weighting. A column of cells straddling the border belongs half to each
    unit, and the operator carries both rows. Assigning by center would give one unit the lot.
    """
    boundary = dissolve_place_boundary(tiles([(0, 0, 4, 4)], ISO_A3=["LAO"]))
    grid = build_cell_grid(boundary, resolution_km=KM_PER_DEGREE_LATITUDE)
    units = gpd.GeoDataFrame(
        {"gid": ["west", "east"], "geometry": [box(0, 0, 1.5, 4), box(1.5, 0, 4, 4)]}, crs=GEOGRAPHIC_CRS
    )

    overlaps = assign_cells_to_units(grid, units)
    shared = overlaps["cell_id"].value_counts()

    assert set(overlaps["gid"]) == {"west", "east"}
    assert (shared == 2).sum() == 4, "one straddling cell per row of the grid"
    np.testing.assert_allclose(overlaps.loc[overlaps["cell_id"] == 1, "coverage"].to_numpy(), [0.5, 0.5])


def test_each_cell_is_fully_accounted_for_by_the_units_covering_it():
    """Units that tile the place must claim every cell exactly once between them. Coverage summing
    to less would lose ground from the totals and more would double-count it.
    """
    boundary = dissolve_place_boundary(tiles([(0, 0, 4, 4)], ISO_A3=["LAO"]))
    grid = build_cell_grid(boundary, resolution_km=KM_PER_DEGREE_LATITUDE)
    units = gpd.GeoDataFrame(
        {"gid": ["west", "east"], "geometry": [box(0, 0, 1.5, 4), box(1.5, 0, 4, 4)]}, crs=GEOGRAPHIC_CRS
    )

    overlaps = assign_cells_to_units(grid, units)
    per_cell = overlaps.groupby("cell_id")["coverage"].sum()

    np.testing.assert_allclose(per_cell.to_numpy(), 1.0, atol=1e-6)
    assert overlaps["overlap_km2"].sum() == pytest.approx(grid.cells["cell_area_km2"].sum(), rel=1e-6)


def test_a_cell_outside_every_unit_gets_no_row():
    """Units rarely tile a country exactly, and a cell in the gap contributes to no total. Keeping
    it with zero coverage would put a column in the operator that nothing reads.
    """
    boundary = dissolve_place_boundary(tiles([(0, 0, 4, 4)], ISO_A3=["LAO"]))
    grid = build_cell_grid(boundary, resolution_km=KM_PER_DEGREE_LATITUDE)
    units = gpd.GeoDataFrame({"gid": ["corner"], "geometry": [box(0, 0, 1, 1)]}, crs=GEOGRAPHIC_CRS)

    overlaps = assign_cells_to_units(grid, units)

    assert len(overlaps) < len(grid.cells)
    assert overlaps["gid"].eq("corner").all()


def test_units_are_reprojected_before_they_meet_the_grid():
    """Units arrive from GADM in whatever CRS it publishes. Overlaying a projected unit on a
    lat/lon grid would find no overlap at all and return an empty operator.
    """
    # Away from the origin, so a polygon left in meters lands nowhere near the lattice rather than
    # swallowing it. At the origin the projected coordinates still cover the degree-scale grid and
    # the test would pass whether or not anything was reprojected.
    boundary = dissolve_place_boundary(tiles([(100, 13, 104, 17)], ISO_A3=["LAO"]))
    grid = build_cell_grid(boundary, resolution_km=KM_PER_DEGREE_LATITUDE)
    units = gpd.GeoDataFrame({"gid": ["all"], "geometry": [box(100, 13, 104, 17)]}, crs=GEOGRAPHIC_CRS).to_crs(
        "EPSG:3395"
    )

    overlaps = assign_cells_to_units(grid, units)

    assert len(overlaps) == len(grid.cells) > 0


def test_units_without_the_key_column_are_rejected():
    boundary = dissolve_place_boundary(tiles([(0, 0, 4, 4)], ISO_A3=["LAO"]))
    grid = build_cell_grid(boundary, resolution_km=KM_PER_DEGREE_LATITUDE)
    units = gpd.GeoDataFrame({"name": ["all"], "geometry": [box(0, 0, 4, 4)]}, crs=GEOGRAPHIC_CRS)

    with pytest.raises(DataValidationError, match="'gid'"):
        assign_cells_to_units(grid, units)


def test_a_footprint_covers_the_ground_its_area_claims():
    """The footprint and `cell_area_km2` describe the same rectangle, so measuring one must give
    the other. A half-step error in the corners would show up as a factor of four here and is
    otherwise invisible, since the centers would still look right.
    """
    boundary = dissolve_place_boundary(tiles([(0, 0, 2, 40)], ISO_A3=["LAO"]))
    grid = build_cell_grid(boundary, resolution_km=50.0)
    geodesic = Geod(ellps="WGS84")

    measured = np.array([abs(geodesic.geometry_area_perimeter(shape)[0]) / 1e6 for shape in grid.footprints])

    np.testing.assert_allclose(measured, grid.cells["cell_area_km2"].to_numpy(), rtol=1e-9)


def test_footprints_are_centerd_on_the_cells_they_belong_to():
    """Off-by-one in the index would pair each footprint with a neighbour's center, which tiles
    just as neatly and puts every covariate one cell out.
    """
    boundary = dissolve_place_boundary(tiles([(0, 0, 2, 2)], ISO_A3=["LAO"]))
    grid = build_cell_grid(boundary, resolution_km=40.0)

    assert grid.footprints.contains(grid.cells.geometry).all()


def test_a_one_cell_place_still_reports_its_geometry():
    """A place smaller than the resolution grids to a single cell, and the derived geometry has to
    survive that. Inferring the step from the gap between the first two centers has no gap to read
    and takes every downstream caller down with it, including the unit assignment.
    """
    tiny = gpd.GeoDataFrame({"ISO_A3": ["SGP"], "geometry": [box(103.6, 1.2, 104.0, 1.5)]}, crs=GEOGRAPHIC_CRS)

    grid = build_cell_grid(dissolve_place_boundary(tiny), resolution_km=200.0)

    assert grid.shape == (1, 1)
    assert grid.steps == pytest.approx((0.4, 0.3))
    assert grid.bounds == pytest.approx((103.6, 1.2, 104.0, 1.5))
    assert grid.footprints.iloc[0].bounds == pytest.approx((103.6, 1.2, 104.0, 1.5))
    assert len(assign_cells_to_units(grid, tiny.rename(columns={"ISO_A3": "gid"}))) == 1


def test_a_cell_on_an_internal_frontier_takes_the_larger_country():
    """A region's cells straddle its members' shared borders. The cell is one piece of ground:
    labeled by whichever country holds most of it, and credited with everything the place covers
    rather than only the dominant country's share.
    """
    # The majority country sorts last, so labelling by the largest share and labelling by the first
    # row the zonal pass returns give different answers.
    boundary = dissolve_place_boundary(tiles([(0, 0, 0.95, 2), (0.95, 0, 2, 2)], ISO_A3=["AAA", "ZZZ"]))

    grid = build_cell_grid(boundary, resolution_km=20.0)
    half_width = grid.longitude_step / 2.0
    straddling = grid.cells[(grid.cells["lon"] - half_width < 0.95) & (grid.cells["lon"] + half_width > 0.95)]

    assert not straddling.empty, "a column of cells should span the frontier"
    assert straddling["ISO_A3"].eq("ZZZ").all(), "the frontier sits left of center, so ZZZ holds more"
    # Coverage comes back from exactextract as float32, so two shares of one cell sum to 1 within
    # single precision rather than exactly.
    np.testing.assert_allclose(
        straddling["place_area_km2"].to_numpy(), straddling["cell_area_km2"].to_numpy(), rtol=1e-6
    )


def test_a_unit_smaller_than_a_cell_still_gets_its_share():
    """Small units are the ones with the fewest events, so losing them to the grid resolution drops
    exactly the observations that most need the field. The unit's own area comes back out of the
    cell it sits inside.
    """
    boundary = dissolve_place_boundary(tiles([(0, 0, 2, 2)], ISO_A3=["LAO"]))
    grid = build_cell_grid(boundary, resolution_km=50.0)
    speck = box(0.5, 0.5, 0.55, 0.55)
    units = gpd.GeoDataFrame({"gid": ["speck"], "geometry": [speck]}, crs=GEOGRAPHIC_CRS)
    exact = abs(Geod(ellps="WGS84").geometry_area_perimeter(speck)[0]) / 1e6

    overlaps = assign_cells_to_units(grid, units)

    assert overlaps["gid"].eq("speck").all()
    assert 0.0 < overlaps["coverage"].sum() < 1.0, "the speck is a fraction of one cell"
    assert overlaps["overlap_km2"].sum() == pytest.approx(exact, rel=1e-3)


def test_the_operator_column_is_the_position_among_surviving_cells():
    """`cell_id` numbers the lattice and the operator numbers what survived clipping, so the two
    diverge from the first dropped cell. Feeding lattice ids straight to the operator would index
    the wrong cells and still produce plausible totals.
    """
    boundary = dissolve_place_boundary(tiles([(0, 0, 2, 1), (0, 1, 1, 2)], ISO_A3=["LAO", "LAO"]))
    grid = build_cell_grid(boundary, resolution_km=40.0)

    columns = grid.column_of_cell(grid.cells["cell_id"].to_numpy())

    assert (grid.cells["cell_id"].to_numpy() != columns).any(), "clipping should have shifted the spaces apart"
    np.testing.assert_array_equal(columns, np.arange(len(grid.cells)))


def test_a_repeated_cell_maps_to_the_same_column():
    """An overlap table names a shared cell once per unit, so the translation has to be a lookup
    rather than a renumbering of whatever it was handed.
    """
    boundary = dissolve_place_boundary(tiles([(0, 0, 2, 2)], ISO_A3=["LAO"]))
    grid = build_cell_grid(boundary, resolution_km=40.0)
    shared = grid.cells["cell_id"].to_numpy()[[3, 1, 3, 1]]

    columns = grid.column_of_cell(shared)

    assert columns[0] == columns[2] == 3
    assert columns[1] == columns[3] == 1


def test_a_cell_outside_the_grid_is_named_rather_than_translated():
    """A cell id from a different lattice, or from this one before clipping, would otherwise become
    a silently wrong column.
    """
    boundary = dissolve_place_boundary(tiles([(0, 0, 1, 1)], ISO_A3=["LAO"]))
    grid = build_cell_grid(boundary, resolution_km=40.0)

    with pytest.raises(DataValidationError, match="not in the grid"):
        grid.column_of_cell(np.array([0, 10_000]))


def write_raster(path, values, bounds=(0.0, 0.0, 1.0, 1.0)):
    """A north-up GeoTIFF holding `values`, spread evenly over `bounds`."""
    rows, columns = values.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=columns,
        count=1,
        dtype="float64",
        crs=GEOGRAPHIC_CRS,
        transform=from_bounds(*bounds, columns, rows),
    ) as writing:
        writing.write(values, 1)

    return str(path)


def test_summing_conserves_the_total_the_raster_holds(tmp_path):
    """A count is extensive: every person in the raster lands in exactly one cell, so the cells add
    back to the whole. Averaging instead would give people per source pixel and lose the scale."""
    raster = write_raster(tmp_path / "people.tif", np.ones((8, 8)))
    place = tiles([(0, 0, 1, 1)], ISO_A3=["AAA"])
    grid = build_cell_grid(place, resolution_km=30.0)

    counted = sample_onto_cells(grid, raster, statistic="sum")

    assert counted.sum() == pytest.approx(64.0)
    assert len(counted) == len(grid.cells)


def test_the_statistic_is_the_callers_choice(tmp_path):
    """An elevation averages where a population sums, and nothing in the grid says which this is."""
    raster = write_raster(tmp_path / "field.tif", np.full((8, 8), 3.0))
    place = tiles([(0, 0, 1, 1)], ISO_A3=["AAA"])
    grid = build_cell_grid(place, resolution_km=30.0)

    assert sample_onto_cells(grid, raster, statistic="mean") == pytest.approx(3.0)
    assert sample_onto_cells(grid, raster, statistic="sum").sum() == pytest.approx(64.0 * 3.0)


def test_a_cell_the_raster_does_not_cover_is_an_error(tmp_path):
    """A sum over no data is zero, so an unmeasured cell is indistinguishable from an empty one and
    the exposure weight silently says nobody lives there."""
    raster = write_raster(tmp_path / "corner.tif", np.ones((4, 4)), bounds=(0.0, 0.0, 0.4, 0.4))
    place = tiles([(0, 0, 1, 1)], ISO_A3=["AAA"])
    grid = build_cell_grid(place, resolution_km=30.0)

    with pytest.raises(DataValidationError, match="not covered"):
        sample_onto_cells(grid, raster, statistic="sum")


def test_values_come_back_in_the_order_of_the_cells(tmp_path):
    """Totals are order-invariant, so nothing above this would notice the values being shuffled —
    and a permuted exposure vector hands each cell its neighbour's people."""
    rising_eastwards = np.tile(np.arange(8, dtype=float), (8, 1))
    raster = write_raster(tmp_path / "gradient.tif", rising_eastwards)
    place = tiles([(0, 0, 1, 1)], ISO_A3=["AAA"])
    grid = build_cell_grid(place, resolution_km=30.0)

    counted = sample_onto_cells(grid, raster, statistic="sum")

    eastwards = np.argsort(grid.cells["lon"].to_numpy(), kind="stable")
    assert np.all(np.diff(counted[eastwards]) >= 0.0), "cells do not rise with longitude as the raster does"


def test_the_grid_carries_the_place_it_was_cut_to():
    """Sampling clips to this rather than to a boundary passed alongside. A grid built from one
    place and sampled against another would count the neighbours and never say so."""
    # L-shaped, so the place and its bounding box are different ground and a grid keeping the box
    # instead would still look right on a rectangular country.
    place = tiles([(0, 0, 1, 1), (1, 0, 2, 1), (0, 1, 1, 2)], ISO_A3=["AAA"] * 3)

    grid = build_cell_grid(place, resolution_km=30.0)

    assert grid.place.equals(place.union_all()), "the grid clips to different ground than it was built from"


def test_a_cell_that_misses_the_place_is_an_error(tmp_path):
    """Cells are kept by one engine's coverage and clipped by another's intersection, so the two can
    disagree on a sliver. Sampling an empty geometry gets a parse error out of the raster library
    rather than a sentence about the grid."""
    raster = write_raster(tmp_path / "field.tif", np.ones((8, 8)))
    place = tiles([(0, 0, 1, 1)], ISO_A3=["AAA"])
    grid = build_cell_grid(place, resolution_km=30.0)
    detached = dataclasses.replace(grid, place=shapely.box(10.0, 10.0, 11.0, 11.0))

    with pytest.raises(DataValidationError, match="do not intersect it"):
        sample_onto_cells(detached, raster, statistic="sum")
