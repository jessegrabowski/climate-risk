import geopandas as gpd
import numpy as np
import polars as pl
import pytest

from shapely.geometry import box

from climate_risk.exceptions import DataValidationError
from climate_risk.geo.raster import assign_cells_to_units, build_cell_grid
from climate_risk.models.frame import region_aggregation, window_aggregation

RESOLUTION_KM = 30.0


def one_degree_place(iso="AAA", corner=(0.0, 0.0)):
    """A square country a degree on a side, which grids to sixteen cells at 30 km."""
    west, south = corner

    return gpd.GeoDataFrame({"ISO_A3": [iso], "geometry": [box(west, south, west + 1.0, south + 1.0)]}, crs="EPSG:4326")


def halves(iso="AAA", corner=(0.0, 0.0)):
    """The place split into a west and an east unit, plus a district inside the west one."""
    west, south = corner

    return gpd.GeoDataFrame(
        {
            "gid": [f"{iso}.1_1", f"{iso}.2_1", f"{iso}.1.1_1"],
            "geometry": [
                box(west, south, west + 0.5, south + 1.0),
                box(west + 0.5, south, west + 1.0, south + 1.0),
                box(west, south, west + 0.5, south + 0.5),
            ],
        },
        crs="EPSG:4326",
    )


def events(*rows, iso="AAA"):
    """One row per event, shaped like `event_windows` output."""
    # `event_windows` aggregates to a genuine empty list for a country-tier event. A literal `[]`
    # in the DataFrame constructor becomes null instead, which the country-tier branch would miss.
    return pl.from_records(
        [(name, iso, gids) for name, gids in rows],
        schema={"DisNo.": pl.String, "ISO": pl.String, "gids": pl.List(pl.String)},
        orient="row",
    )


def test_a_window_naming_a_unit_and_a_unit_inside_it_counts_each_cell_once():
    """Names in one event nest freely — a province and a district inside it are two names for
    overlapping ground. Adding their shares would weight the overlap twice and quietly make the
    event look like it happened harder there."""
    place = one_degree_place()
    grid = build_cell_grid(place, resolution_km=RESOLUTION_KM)
    coverage = assign_cells_to_units(grid, halves())
    weights = np.ones(len(grid.cells))

    nested = window_aggregation(events(("nested", ["AAA.1_1", "AAA.1.1_1"])), coverage, grid, cell_weights=weights)
    province_only = window_aggregation(events(("province", ["AAA.1_1"])), coverage, grid, cell_weights=weights)

    assert nested.matrix.sum() == pytest.approx(province_only.matrix.sum())
    assert nested.matrix.max() == pytest.approx(1.0), "a cell carries at most its own weight"


def test_an_event_no_source_placed_is_observed_through_its_whole_country():
    """A country-tier event happened somewhere, and the window that says so is the country. Dropping
    it would discard an eighth of the workbook for being imprecise rather than absent."""
    place = one_degree_place()
    grid = build_cell_grid(place, resolution_km=RESOLUTION_KM)
    coverage = assign_cells_to_units(grid, halves())
    weights = np.arange(1.0, len(grid.cells) + 1.0)

    aggregation = window_aggregation(events(("unplaced", [])), coverage, grid, cell_weights=weights)

    assert aggregation.units == ("unplaced",)
    assert aggregation.matrix.sum() == pytest.approx(weights.sum()), "the window is every cell of the country"


def test_a_country_window_stops_at_its_own_border():
    """A region holds several countries on one grid, and an unplaced Lao event says nothing about
    Vietnam. Covering the whole grid would spread it over the neighbours."""
    two = gpd.GeoDataFrame({"ISO_A3": ["AAA", "BBB"], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]}, crs="EPSG:4326")
    grid = build_cell_grid(two, resolution_km=RESOLUTION_KM)
    coverage = assign_cells_to_units(grid, halves())
    weights = np.ones(len(grid.cells))

    aggregation = window_aggregation(events(("unplaced", []), iso="AAA"), coverage, grid, cell_weights=weights)
    at_home = (grid.cells["ISO_A3"] == "AAA").sum()

    assert aggregation.matrix.sum() == pytest.approx(float(at_home))
    assert at_home < len(grid.cells), "the grid holds cells of the other country too"


def test_the_weights_carry_the_exposure_the_caller_supplied():
    """The operator's weight is what multiplies the intensity, so an exposure-weighted total needs
    the population in it rather than the fraction of a cell the window covers."""
    place = one_degree_place()
    grid = build_cell_grid(place, resolution_km=RESOLUTION_KM)
    coverage = assign_cells_to_units(grid, halves())
    people = np.full(len(grid.cells), 100.0)

    aggregation = window_aggregation(events(("west", ["AAA.1_1"])), coverage, grid, cell_weights=people)

    assert aggregation.matrix.sum() == pytest.approx(100.0 * len(grid.cells) / 2.0)


def test_weights_that_do_not_index_the_grid_are_an_error():
    """Cell weights arrive from a raster and the grid from a boundary. A length mismatch would index
    the wrong cells and produce a plausible operator over the wrong ground."""
    place = one_degree_place()
    grid = build_cell_grid(place, resolution_km=RESOLUTION_KM)
    coverage = assign_cells_to_units(grid, halves())

    with pytest.raises(DataValidationError, match="index the same grid"):
        window_aggregation(events(("west", ["AAA.1_1"])), coverage, grid, cell_weights=np.ones(3))

    with pytest.raises(DataValidationError, match="index the same grid"):
        region_aggregation(grid, np.ones(3))


def test_the_region_spans_every_cell_in_one_row():
    """The objective subtracts this once for the whole place, so it has to be one row and it has to
    miss nothing."""
    place = one_degree_place()
    grid = build_cell_grid(place, resolution_km=RESOLUTION_KM)
    weights = np.arange(1.0, len(grid.cells) + 1.0)

    region = region_aggregation(grid, weights)

    assert region.n_units == 1
    assert region.matrix.sum() == pytest.approx(weights.sum())


def test_an_event_whose_units_miss_the_grid_is_dropped_and_the_rest_survive():
    """A place's grid holds its own units. An event carried over from a neighbour reaches no cell,
    and the model has nowhere to put it — but it must not take the rest of the frame with it."""
    place = one_degree_place()
    grid = build_cell_grid(place, resolution_km=RESOLUTION_KM)
    coverage = assign_cells_to_units(grid, halves())
    weights = np.ones(len(grid.cells))

    aggregation = window_aggregation(
        events(("here", ["AAA.1_1"]), ("elsewhere", ["ZZZ.9_1"])), coverage, grid, cell_weights=weights
    )

    assert aggregation.units == ("here",)


def test_no_window_reaching_the_grid_at_all_is_an_error():
    """An operator with no rows fits nothing and says nothing. Far better to fail where the grid and
    the events were chosen than to hand the model an empty likelihood."""
    place = one_degree_place()
    grid = build_cell_grid(place, resolution_km=RESOLUTION_KM)
    coverage = assign_cells_to_units(grid, halves())

    with pytest.raises(DataValidationError, match="No event window reaches"):
        window_aggregation(events(("elsewhere", ["ZZZ.9_1"])), coverage, grid, cell_weights=np.ones(len(grid.cells)))


def test_the_rows_carry_a_defined_order():
    """Counts are supplied by position, so the caller has to know which row is which event. Sorted
    labels make that a lookup rather than a guess about insertion order."""
    place = one_degree_place()
    grid = build_cell_grid(place, resolution_km=RESOLUTION_KM)
    coverage = assign_cells_to_units(grid, halves())

    aggregation = window_aggregation(
        events(("zeta", ["AAA.1_1"]), ("alpha", ["AAA.2_1"])), coverage, grid, cell_weights=np.ones(len(grid.cells))
    )

    assert aggregation.units == ("alpha", "zeta")
