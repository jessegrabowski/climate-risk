import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_risk.data.chirps import annual_extremes, pentad_totals
from climate_risk.data.covariate_table import gumbel_return_level, standardise_against_baseline


def test_gumbel_return_level_recovers_known_distribution():
    rng = np.random.default_rng(0)
    sample = rng.gumbel(loc=100.0, scale=20.0, size=(20000, 1))
    expected = 100.0 - 20.0 * np.log(-np.log(1 - 1 / 5))
    assert gumbel_return_level(sample)[0] == pytest.approx(expected, rel=0.01)


def test_pentads_are_six_per_month_and_sum_the_days():
    dates = pd.date_range("2021-01-01", "2021-12-31")
    daily = xr.DataArray(np.ones((len(dates), 1, 1)), coords={"time": dates}, dims=("time", "y", "x"))
    pentads = pentad_totals(daily)
    assert pentads.sizes["pentad_id"] == 72
    assert float(pentads.sum()) == 365.0
    assert float(pentads.max()) == 6.0  # a 31-day month ends on a 6-day pentad


def test_rx5day_finds_the_wettest_window():
    dates = pd.date_range("2021-01-01", "2021-12-31")
    values = np.zeros((len(dates), 1, 1))
    values[100:103] = 10.0
    daily = xr.DataArray(values, coords={"time": dates}, dims=("time", "y", "x"))
    assert float(annual_extremes(daily)["rx5day"].squeeze()) == 30.0


def test_standardise_uses_baseline_years_only():
    years = np.arange(1981, 2026)
    totals = np.where(years[:, None] < 1991, 1e6, np.arange(len(years))[:, None].astype(float))
    z = standardise_against_baseline(totals, years)
    inside = (years >= 1991) & (years <= 2020)
    assert z[inside].mean() == pytest.approx(0.0, abs=1e-9)
    assert z[inside].std(ddof=1) == pytest.approx(1.0)
