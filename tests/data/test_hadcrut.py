import dataclasses

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import requests
import xarray as xr

from shapely.geometry import box

from climate_risk.data import hadcrut
from climate_risk.data.hadcrut import HADCRUT, load_hadcrut_data, temperature_on_cells, transform_hadcrut
from climate_risk.geo.raster import build_cell_grid, dissolve_place_boundary
from tests.conftest import toy_world, toy_world_with_places

# The panel starts in 1960; the gridded record reaches back much further.
FIRST_PANEL_YEAR = 1959

# Every parameter but the reading digest is spelled out, so a key that lost one fails rather than
# agreeing with whatever the loader happened to write.
REPAIRED_CACHE = "hadcrut__reading=*__repaired_iso=True.parquet"
UNREPAIRED_CACHE = "hadcrut__reading=*__repaired_iso=False.parquet"


def gridded(rows) -> pd.DataFrame:
    """Gridded anomalies as xarray hands them over, one row per cell and timestamp."""
    return pd.DataFrame(rows, columns=["time", "latitude", "longitude", "tas_mean"]).assign(
        time=lambda x: pd.to_datetime(x["time"])
    )


def test_cells_are_averaged_per_country_and_year():
    """Many grid cells fall in one country, and the panel wants one number per country-year."""
    temperatures = gridded(
        [
            ("1990-01-01", 0.5, 0.5, 1.0),
            ("1990-06-01", 0.5, 0.5, 3.0),
            ("1990-01-01", 0.5, 2.5, 10.0),
        ]
    )

    annual = transform_hadcrut(temperatures, toy_world())

    assert annual.loc[("AAA", pd.Timestamp("1990-01-01")), "tas_mean"] == pytest.approx(2.0)
    assert annual.loc[("BBB", pd.Timestamp("1990-01-01")), "tas_mean"] == pytest.approx(10.0)


def test_cells_outside_every_country_are_dropped():
    """The grid covers ocean, which belongs to no country and would otherwise average in."""
    temperatures = gridded([("1990-01-01", 0.5, 0.5, 1.0), ("1990-01-01", 50.0, 50.0, 99.0)])

    annual = transform_hadcrut(temperatures, toy_world())

    assert annual["tas_mean"].tolist() == [1.0]


def test_years_before_the_panel_are_dropped():
    temperatures = gridded([("1950-01-01", 0.5, 0.5, 1.0), ("1990-01-01", 0.5, 0.5, 2.0)])

    annual = transform_hadcrut(temperatures, toy_world())

    assert pd.DatetimeIndex(annual.index.get_level_values("year")).year.tolist() == [1990]


def test_the_boundary_year_is_excluded_but_the_next_is_kept():
    """The comparison is exclusive, and off by one here adds a sparse year to the panel."""
    temperatures = gridded(
        [(f"{FIRST_PANEL_YEAR}-01-01", 0.5, 0.5, 1.0), (f"{FIRST_PANEL_YEAR + 1}-01-01", 0.5, 0.5, 2.0)]
    )

    annual = transform_hadcrut(temperatures, toy_world())

    assert pd.DatetimeIndex(annual.index.get_level_values("year")).year.tolist() == [FIRST_PANEL_YEAR + 1]


def test_repairing_iso_codes_gets_its_own_cache_entry(tmp_path, write_shapefile_cache):
    """Repairing changes which countries appear, so one entry cannot serve both settings."""
    write_gridded(tmp_path, west=np.ones(24), east=np.ones(24))
    write_shapefile_cache("world", toy_world_with_places())

    load_hadcrut_data(tmp_path, repair_ISO_codes=True)
    load_hadcrut_data(tmp_path, repair_ISO_codes=False)

    assert len(list(tmp_path.glob(REPAIRED_CACHE))) == 1
    assert len(list(tmp_path.glob(UNREPAIRED_CACHE))) == 1


def write_gridded(cache_dir, *, west, east):
    """Write a two-box HadCRUT file, monthly over 2000 and 2001, holding ``west`` and ``east`` per month."""
    time = pd.date_range("2000-01-01", periods=24, freq="MS") + pd.Timedelta(days=15)
    values = np.stack([west, east], axis=-1)[:, None, :]
    xr.Dataset(
        {
            "tas_mean": (("time", "latitude", "longitude"), values),
            "latitude_bnds": (("latitude", "bnds"), [[0.0, 5.0]]),
            "longitude_bnds": (("longitude", "bnds"), [[0.0, 5.0], [5.0, 10.0]]),
        },
        coords={"time": time, "latitude": [2.5], "longitude": [2.5, 7.5]},
    ).to_netcdf(cache_dir / HADCRUT.filename)


@pytest.fixture
def grid():
    """Cells over two 5-degree boxes side by side."""
    boundary = dissolve_place_boundary(
        gpd.GeoDataFrame({"ISO_A3": ["AAA"], "geometry": [box(0, 0, 10, 5)]}, crs="EPSG:4326")
    )

    return build_cell_grid(boundary, resolution_km=100.0)


def test_each_cell_takes_the_annual_mean_of_the_box_its_center_falls_in(tmp_path, grid):
    """Box 2.5 east holds 1 through 2000 and 3 through 2001; box 7.5 east holds 5 through both. The
    grid puts a column of centers on the shared edge at 5 east, which belongs to the eastern box."""
    write_gridded(tmp_path, west=np.repeat([1.0, 3.0], 12), east=np.full(24, 5.0))

    anomalies = temperature_on_cells(grid, tmp_path).merge(grid.cells[["cell_id", "lon"]], on="cell_id")
    western = anomalies["lon"] < 5.0

    assert anomalies.loc[western & (anomalies["year"] == 2000), "tas_mean"].unique().tolist() == [1.0]
    assert anomalies.loc[western & (anomalies["year"] == 2001), "tas_mean"].unique().tolist() == [3.0]
    assert anomalies.loc[~western, "tas_mean"].unique().tolist() == [5.0]
    assert len(anomalies) == 2 * len(grid.cells)


def test_a_year_missing_a_month_is_missing(tmp_path, grid):
    """An average over eleven months is not the year's, so the year is missing rather than wrong."""
    east = np.full(24, 5.0)
    east[13] = np.nan
    write_gridded(tmp_path, west=np.ones(24), east=east)

    anomalies = temperature_on_cells(grid, tmp_path).merge(grid.cells[["cell_id", "lon"]], on="cell_id")
    eastern_2001 = (anomalies["lon"] >= 5.0) & (anomalies["year"] == 2001)

    assert anomalies.loc[eastern_2001, "tas_mean"].isna().all()
    assert anomalies.loc[~eastern_2001, "tas_mean"].notna().all()


def test_a_warm_read_needs_neither_the_file_nor_the_network(tmp_path, grid, monkeypatch):
    write_gridded(tmp_path, west=np.ones(24), east=np.ones(24))
    first = temperature_on_cells(grid, tmp_path)
    (tmp_path / HADCRUT.filename).unlink()

    def offline(*args, **kwargs):
        raise AssertionError("reached the network on a warm cache")

    monkeypatch.setattr(requests, "get", offline)
    pd.testing.assert_frame_equal(temperature_on_cells(grid, tmp_path), first)


def test_a_new_release_of_the_record_turns_the_cached_cells_over(tmp_path, grid, monkeypatch):
    """A new release downloads to a new file, and cells read from the old release must not be served for it."""
    write_gridded(tmp_path, west=np.ones(24), east=np.ones(24))
    temperature_on_cells(grid, tmp_path)

    release = dataclasses.replace(
        HADCRUT,
        url=HADCRUT.url.replace("5.0.2.0", "5.1.0.0"),
        filename=HADCRUT.filename.replace("5.0.2.0", "5.1.0.0"),
    )
    write_gridded(tmp_path, west=np.full(24, 2.0), east=np.full(24, 2.0))
    (tmp_path / HADCRUT.filename).rename(tmp_path / release.filename)
    monkeypatch.setattr(hadcrut, "HADCRUT", release)

    assert temperature_on_cells(grid, tmp_path)["tas_mean"].unique().tolist() == [2.0]


def test_each_cell_reads_its_own_latitude_band(tmp_path):
    """Two boxes stacked north over south, so a lookup that read latitude by longitude, or by the wrong edge,
    would put both bands in one box."""
    time = pd.date_range("2000-01-01", periods=12, freq="MS") + pd.Timedelta(days=15)
    xr.Dataset(
        {
            "tas_mean": (("time", "latitude", "longitude"), np.tile([[[1.0], [5.0]]], (12, 1, 1))),
            "latitude_bnds": (("latitude", "bnds"), [[0.0, 5.0], [5.0, 10.0]]),
            "longitude_bnds": (("longitude", "bnds"), [[0.0, 5.0]]),
        },
        coords={"time": time, "latitude": [2.5, 7.5], "longitude": [2.5]},
    ).to_netcdf(tmp_path / HADCRUT.filename)
    boundary = dissolve_place_boundary(
        gpd.GeoDataFrame({"ISO_A3": ["AAA"], "geometry": [box(0, 0, 5, 10)]}, crs="EPSG:4326")
    )
    stacked = build_cell_grid(boundary, resolution_km=100.0)

    anomalies = temperature_on_cells(stacked, tmp_path).merge(stacked.cells[["cell_id", "lat"]], on="cell_id")
    southern = anomalies["lat"] < 5.0

    assert anomalies.loc[southern, "tas_mean"].unique().tolist() == [1.0]
    assert anomalies.loc[~southern, "tas_mean"].unique().tolist() == [5.0]


def test_force_reload_downloads_the_record_again_and_rereads_it(tmp_path, grid, serve_downloads):
    write_gridded(tmp_path, west=np.ones(24), east=np.ones(24))
    temperature_on_cells(grid, tmp_path)

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    write_gridded(upstream, west=np.full(24, 2.0), east=np.full(24, 2.0))
    requested = serve_downloads({HADCRUT.url: (upstream / HADCRUT.filename).read_bytes()})

    assert temperature_on_cells(grid, tmp_path, force_reload=True)["tas_mean"].unique().tolist() == [2.0]
    assert requested == [HADCRUT.url]
