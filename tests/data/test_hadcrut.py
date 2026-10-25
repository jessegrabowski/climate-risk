import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import requests
import xarray as xr

from shapely.geometry import box

from climate_risk.data.hadcrut import HADCRUT, load_hadcrut_data, temperature_on_cells, transform_hadcrut
from climate_risk.geo.raster import build_cell_grid, dissolve_place_boundary
from tests.conftest import toy_world

# The panel starts in 1960; the gridded record reaches back much further.
FIRST_PANEL_YEAR = 1959

# The cache key is stated literally, so a wrong one fails rather than agreeing with itself.
REPAIRED_CACHE = "hadcrut__repaired_iso=True.parquet"


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

    assert annual.loc[("AAA", pd.Timestamp("1990-01-01")), "surface_temperature_dev"] == pytest.approx(2.0)
    assert annual.loc[("BBB", pd.Timestamp("1990-01-01")), "surface_temperature_dev"] == pytest.approx(10.0)


def test_cells_outside_every_country_are_dropped():
    """The grid covers ocean, which belongs to no country and would otherwise average in."""
    temperatures = gridded([("1990-01-01", 0.5, 0.5, 1.0), ("1990-01-01", 50.0, 50.0, 99.0)])

    annual = transform_hadcrut(temperatures, toy_world())

    assert annual["surface_temperature_dev"].tolist() == [1.0]


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


def test_repairing_iso_codes_gets_its_own_cache_entry(tmp_path):
    """Repairing changes which countries appear, so one entry cannot serve both settings."""
    pd.DataFrame(
        {"surface_temperature_dev": [0.5]},
        index=pd.MultiIndex.from_arrays([["LAO"], pd.to_datetime(["1960-01-01"])], names=["ISO", "year"]),
    ).to_parquet(tmp_path / REPAIRED_CACHE)

    assert load_hadcrut_data(tmp_path).index.get_level_values("ISO").tolist() == ["LAO"]
    assert not (tmp_path / "hadcrut__repaired_iso=False.parquet").exists()


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
