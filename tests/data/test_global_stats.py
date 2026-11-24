import numpy as np
import pandas as pd
import xarray as xr

from climate_risk.data.global_stats import (
    daily_precip_stats,
    return_level_from_maxima,
    sliding_max_sum,
    standardise_stack,
    wrap_longitude,
)
from climate_risk.data.point_extract import values_at_points


def test_sliding_max_sum_stays_inside_the_year():
    days = np.zeros((10, 1, 1))
    days[[0, 1, 2, 7, 8, 9]] = 1.0
    assert sliding_max_sum(days)[0, 0] == 3.0


def test_daily_precip_stats_converts_units_and_masks_missing():
    rain = np.zeros((8, 2, 2), dtype="float32")
    rain[2:5, 0, 0] = 0.01  # 10 mm a day for three days
    rain[:, 1, 1] = np.nan
    data = xr.Dataset(
        {"tp": (("valid_time", "latitude", "longitude"), rain)},
        coords={"latitude": [1.0, 0.0], "longitude": [0.0, 1.0]},
    )
    out = daily_precip_stats(data, "tp", 1000.0)
    assert float(out["rx5day"].sel(latitude=1.0, longitude=0.0)) == 30.0
    assert np.isnan(float(out["rx5day"].sel(latitude=0.0, longitude=1.0)))


def test_wrap_longitude_orders_minus_180_to_180():
    data = xr.Dataset({"x": (("longitude",), [0.0, 1.0, 2.0])}, coords={"longitude": [0.0, 90.0, 270.0]})
    assert wrap_longitude(data)["longitude"].to_numpy().tolist() == [-90.0, 0.0, 90.0]


def test_return_level_needs_ten_years_and_standardise_has_unit_spread():
    rng = np.random.default_rng(0)
    years = np.arange(1981, 2026)
    stack = xr.DataArray(
        rng.gumbel(100, 20, (len(years), 2, 2)),
        dims=("year", "latitude", "longitude"),
        coords={"year": years, "latitude": [1.0, 0.0], "longitude": [0.0, 1.0]},
    )
    stack[:, 0, 0] = np.nan
    stack[:5, 1, 1] = np.nan
    stack[5:, 1, 1] = np.nan  # fewer than ten valid years
    level = return_level_from_maxima(stack)
    assert np.isnan(level.to_numpy()[0, 0]) and np.isfinite(level.to_numpy()[0, 1])
    z = standardise_stack(stack.where(stack.notnull(), 1.0))
    assert abs(float(z.sel(year=slice(1991, 2020)).isel(latitude=0, longitude=1).std("year", ddof=1)) - 1.0) < 1e-6


def test_values_at_points_reads_the_containing_pixel_and_nan_outside(tmp_path):
    grid = xr.Dataset(
        {"rx5day": (("latitude", "longitude"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={"latitude": [1.0, 0.0], "longitude": [0.0, 1.0]},
    )
    grid.to_netcdf(tmp_path / "era5_daily_2000.nc")
    points = pd.DataFrame({"longitude": [0.9, 5.0], "latitude": [0.1, 0.0], "year": [2000, 2000]})
    out = values_at_points(points, tmp_path)
    assert out["rx5day"].iloc[0] == 4.0 and np.isnan(out["rx5day"].iloc[1])
