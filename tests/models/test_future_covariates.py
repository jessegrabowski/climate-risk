from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from climate_risk.models import future_covariates
from climate_risk.models.climate_economy import OBSERVED
from climate_risk.models.future_covariates import FUTURE_COVARIATE_COLUMNS, project_future_covariates

CACHE_DIR = Path("unused")


def _historical_row(
    *,
    iso: str = "AAA",
    year: int = 2025,
    population: float = 10.0,
    ln_population_density: float | None = 2.0,
    ln_gdp_pc: float = 7.0,
    square_ln_gdp_pc: float = 49.0,
    precip_deviation: float = 5.0,
    co2: float | None = 420.0,
    dev_from_trend_ocean_temp: float = 0.1,
) -> dict:
    return {
        "ISO": iso,
        "date": date(year, 1, 1),
        "population": population,
        "ln_population_density": ln_population_density,
        "ln_gdp_pc": ln_gdp_pc,
        "square_ln_gdp_pc": square_ln_gdp_pc,
        "precip_deviation": precip_deviation,
        "co2": co2,
        "dev_from_trend_ocean_temp": dev_from_trend_ocean_temp,
    }


def _forecast_idata(
    *, year: int, log_pop: float, log_gdp: float, log_co2: float, precip: float, temp: float
) -> xr.DataTree:
    values = {"log_co2": log_co2, "log_pop": log_pop, "log_gdp": log_gdp, "precip": precip, "temp": temp}
    data = np.array([[values.get(name, 0.0) for name in OBSERVED]])
    ds = xr.Dataset(
        {"forecast_observed": (["chain", "draw", "time", "observed_state"], data[None, None, :, :])},
        coords={"time": [pd.Timestamp(f"{year}-01-01")], "observed_state": OBSERVED},
    )
    return xr.DataTree(dataset=ds)


@pytest.fixture(autouse=True)
def _stub_real_data_lookups(monkeypatch):
    """Every year in these tests is either historical or exercises the forecast arithmetic
    directly, so nothing here needs the real World Bank, GPCC or ocean-heat caches."""
    monkeypatch.setattr(future_covariates, "_density_per_capita", lambda iso, cache_dir: 0.5)
    monkeypatch.setattr(future_covariates, "_baseline_precip", lambda iso, cache_dir: 1000.0)
    monkeypatch.setattr(
        future_covariates, "_dev_from_trend_ocean_temp", lambda temp_std_forecast, cache_dir: temp_std_forecast
    )


def test_reads_a_historical_year_directly():
    panel = pl.DataFrame([_historical_row(year=2025, ln_population_density=2.0)])

    result = project_future_covariates(
        iso="AAA", forecast_idata=xr.DataTree(), target_years=[2025], historical_panel=panel, cache_dir=CACHE_DIR
    )

    assert result.columns == FUTURE_COVARIATE_COLUMNS
    assert result["year"].to_list() == [2025]
    assert result["ln_population_density"].item() == 2.0


def test_backfills_a_lagging_population_density_from_population():
    row = _historical_row(year=2025, population=10.0, ln_population_density=None)
    panel = pl.DataFrame([row], schema_overrides={"ln_population_density": pl.Float64})

    result = project_future_covariates(
        iso="AAA", forecast_idata=xr.DataTree(), target_years=[2025], historical_panel=panel, cache_dir=CACHE_DIR
    )

    # density_per_capita is stubbed to 0.5; population is stored in millions.
    assert result["ln_population_density"].item() == pytest.approx(np.log(10.0 * 1e6 * 0.5))


def test_raises_when_a_historical_row_has_an_unfillable_null():
    row = _historical_row(year=2025, co2=None)
    panel = pl.DataFrame([row], schema_overrides={"co2": pl.Float64})

    with pytest.raises(ValueError, match="unfilled null covariate"):
        project_future_covariates(
            iso="AAA", forecast_idata=xr.DataTree(), target_years=[2025], historical_panel=panel, cache_dir=CACHE_DIR
        )


def test_projects_a_future_year_from_the_forecasts_posterior_mean():
    panel = pl.DataFrame([_historical_row(year=2020)])  # unrelated year, just to give a typed empty frame
    forecast_idata = _forecast_idata(
        year=2030, log_pop=np.log(20.0), log_gdp=np.log(140.0), log_co2=np.log(450.0), precip=np.log(1200.0), temp=0.2
    )

    result = project_future_covariates(
        iso="AAA", forecast_idata=forecast_idata, target_years=[2030], historical_panel=panel, cache_dir=CACHE_DIR
    )

    row = result.row(0, named=True)
    assert row["year"] == 2030
    assert row["population"] == pytest.approx(20.0)  # exp(log_pop) * 1e6, back in millions
    assert row["ln_gdp_pc"] == pytest.approx(np.log(140.0 / 20.0))
    assert row["square_ln_gdp_pc"] == pytest.approx(row["ln_gdp_pc"] ** 2)
    assert row["co2"] == pytest.approx(450.0)
    assert row["precip_deviation"] == pytest.approx(1200.0 - 1000.0)  # baseline_precip stubbed to 1000.0
    assert row["dev_from_trend_ocean_temp"] == pytest.approx(0.2)  # stub returns its input unchanged
    assert row["ln_population_density"] == pytest.approx(np.log(20.0 * 1e6 * 0.5))


def test_combines_a_historical_year_and_a_forecast_year_in_one_call():
    panel = pl.DataFrame([_historical_row(year=2025)])
    forecast_idata = _forecast_idata(
        year=2050, log_pop=np.log(5.0), log_gdp=np.log(50.0), log_co2=np.log(500.0), precip=np.log(800.0), temp=0.0
    )

    result = project_future_covariates(
        iso="AAA", forecast_idata=forecast_idata, target_years=[2025, 2050], historical_panel=panel, cache_dir=CACHE_DIR
    )

    assert result["year"].to_list() == [2025, 2050]
