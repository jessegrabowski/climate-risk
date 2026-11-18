import numpy as np
import polars as pl
import pytest
import xarray as xr

from climate_risk.dsge.capital import capital_stock_usd

RATIO = pl.DataFrame({"year": [2018, 2019], "capital_output": [3.8, 4.0]})


def _gdp(years: list[int], values: list[float]) -> xr.DataArray:
    return xr.DataArray(values, dims="year", coords={"year": years})


def test_capital_is_ratio_times_gdp() -> None:
    capital = capital_stock_usd(_gdp([2018, 2019], [100.0, 200.0]), RATIO)
    np.testing.assert_allclose(capital.values, [380.0, 800.0])


def test_ratio_is_held_flat_after_the_last_pwt_year(caplog: pytest.LogCaptureFixture) -> None:
    capital = capital_stock_usd(_gdp([2019, 2030], [100.0, 100.0]), RATIO)
    np.testing.assert_allclose(capital.values, [400.0, 400.0])
    assert "Holding the capital-output ratio" in caplog.text


def test_ratio_is_held_flat_before_the_first_pwt_year() -> None:
    capital = capital_stock_usd(_gdp([2010], [10.0]), RATIO)
    np.testing.assert_allclose(capital.values, [38.0])


def test_price_factor_restates_dollars() -> None:
    capital = capital_stock_usd(_gdp([2018], [100.0]), RATIO, price_factor=1.25)
    np.testing.assert_allclose(capital.values, [475.0])


def test_nonpositive_price_factor_raises() -> None:
    with pytest.raises(ValueError, match="price_factor"):
        capital_stock_usd(_gdp([2018], [100.0]), RATIO, price_factor=0.0)
