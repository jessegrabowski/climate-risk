from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from statsmodels.tsa.seasonal import STL

from climate_risk.data.world_bank import load_wb_data
from climate_risk.data_functions.combine_data import build_time_series, total_precipitation
from climate_risk.replication_data import CLIMATOLOGY_BASELINE

# The seasonal period replication_data.py fits the historical ocean-heat trend with. Extending the
# series with a forecast and refitting has to use the same period, or the two deviations disagree
# for reasons that have nothing to do with the forecast.
OCEAN_TREND_PERIOD = 3

FUTURE_COVARIATE_COLUMNS = [
    "ISO",
    "year",
    "population",
    "ln_population_density",
    "ln_gdp_pc",
    "square_ln_gdp_pc",
    "precip_deviation",
    "co2",
    "dev_from_trend_ocean_temp",
]


def _baseline_precip(iso: str, cache_dir: Path) -> float:
    """A country's mean annual precipitation over the climatology baseline, in GPCC's own units."""
    precip = total_precipitation(cache_dir).to_pandas()
    first, last = CLIMATOLOGY_BASELINE
    block = precip.loc[(precip["ISO"] == iso) & (precip["date"].dt.year.between(first, last))]

    return float(block["precip"].mean())


def _density_per_capita(iso: str, cache_dir: Path) -> float:
    """Population density per unit of population, from the country's latest covered year.

    World Bank publishes population density and population as separate indicators that do not
    always update in the same year, so a future or lagging population_density is rebuilt from this
    ratio and a population figure instead.
    """
    wb = load_wb_data(cache_dir).to_pandas()
    home = wb.loc[wb["country_code"] == iso].dropna(subset=["population_density", "population"]).sort_values("year")
    latest = home.iloc[-1]

    return float(latest["population_density"] / latest["population"])


def _dev_from_trend_ocean_temp(temp_std_forecast: pd.Series, cache_dir: Path) -> pd.Series:
    """Extend the historical ocean-heat series with a forecast and refit its trend deviation.

    ``temp_std_forecast`` is standardised the way :mod:`climate_risk.models.climate_economy` fits
    it (centred and scaled by the historical series' own mean and standard deviation), so it is
    converted back to raw units before joining the two.
    """
    time_series = build_time_series(cache_dir).to_pandas().set_index("date")
    ocean_heat = time_series["Temp"].dropna()

    forecast_raw = temp_std_forecast * ocean_heat.std() + ocean_heat.mean()
    extended = pd.concat([ocean_heat[~ocean_heat.index.isin(forecast_raw.index)], forecast_raw]).sort_index()

    trend: pd.Series = extended - STL(extended, period=OCEAN_TREND_PERIOD).fit().trend

    return trend


def project_future_covariates(
    *, iso: str, forecast_idata: xr.DataTree, target_years: list[int], historical_panel: pl.DataFrame, cache_dir: Path
) -> pl.DataFrame:
    """
    Build covariate rows for a country at the requested years.

    A requested year already covered by ``historical_panel`` is read from it directly. A later year
    is point-projected from ``forecast_idata``'s posterior mean: this carries forward the climate-
    economy forecast's central trajectory only, not its own uncertainty, so the damage curve built
    from these rows reflects the frequency and damage models' posterior spread around a fixed future
    rather than the full joint uncertainty of both stages.

    Parameters
    ----------
    iso : str
        ISO alpha-3 code of the country to project.
    forecast_idata : DataTree
        From :func:`~climate_risk.models.climate_economy_fit.forecast`, for the same country.
    target_years : list of int
        Years to build rows for.
    historical_panel : DataFrame
        A polars frame in the shape :func:`~climate_risk.replication_data.create_replication_data`
        returns, covering at least the years in ``target_years`` that are not in the future.
    cache_dir : Path
        Directory the source caches live under.

    Returns
    -------
    future : DataFrame
        Columns :data:`FUTURE_COVARIATE_COLUMNS`, one row per year in ``target_years``.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        import xarray as xr

        from climate_risk.models.future_covariates import project_future_covariates
        from climate_risk.replication_data import create_replication_data

        cache_dir = Path("data")
        forecast_idata = xr.open_datatree("data/idata/climate_economy_npl_forecast.nc")
        future = project_future_covariates(
            iso="NPL",
            forecast_idata=forecast_idata,
            target_years=[2025, 2030, 2040, 2050],
            historical_panel=create_replication_data(cache_dir),
            cache_dir=cache_dir,
        )
    """
    historical = historical_panel.filter((pl.col("ISO") == iso) & (pl.col("date").dt.year().is_in(target_years)))
    historical_years = set(historical["date"].dt.year().to_list())
    forecast_years = [year for year in target_years if year not in historical_years]

    density_per_capita = _density_per_capita(iso, cache_dir)

    # A lagging World Bank indicator (population_density itself) can go null for the most recent
    # year before others catch up; population does not lag the same way, so rebuild density from
    # it rather than silently dropping the row's contribution to the linear predictor.
    historical_rows = historical.select(
        "ISO",
        pl.col("date").dt.year().cast(pl.Int64).alias("year"),
        "population",
        pl.when(pl.col("ln_population_density").is_not_null())
        .then(pl.col("ln_population_density"))
        .otherwise((pl.col("population") * 1e6 * density_per_capita).log())
        .alias("ln_population_density"),
        "ln_gdp_pc",
        "square_ln_gdp_pc",
        "precip_deviation",
        "co2",
        "dev_from_trend_ocean_temp",
    )

    still_null = historical_rows.filter(pl.any_horizontal(pl.all().is_null()))
    if still_null.height:
        raise ValueError(f"{iso} has an unfilled null covariate for a requested historical year:\n{still_null}")

    if not forecast_years:
        return historical_rows

    forecast_observed = forecast_idata.forecast_observed.mean(dim=("chain", "draw"))
    times = pd.to_datetime(forecast_observed["time"].values)

    def series(name: str) -> pd.Series:
        return pd.Series(forecast_observed.sel(observed_state=name).values, index=times)

    log_gdp_per_cap: pd.Series = series("log_gdp") - series("log_pop")

    population: pd.Series = np.exp(series("log_pop")) * 1e6
    gdp_per_cap: pd.Series = np.exp(log_gdp_per_cap)
    precip_level: pd.Series = np.exp(series("precip"))
    co2: pd.Series = np.exp(series("log_co2"))
    trend_deviation = _dev_from_trend_ocean_temp(series("temp"), cache_dir)
    baseline_precip = _baseline_precip(iso, cache_dir)

    rows = []
    for year in forecast_years:
        as_of = pd.Timestamp(f"{year}-01-01")
        pop = float(population.loc[as_of])
        ln_gdp_pc = float(np.log(gdp_per_cap.loc[as_of]))
        rows.append(
            {
                "ISO": iso,
                "year": year,
                "population": pop / 1e6,
                "ln_population_density": float(np.log(pop * density_per_capita)),
                "ln_gdp_pc": ln_gdp_pc,
                "square_ln_gdp_pc": ln_gdp_pc**2,
                "precip_deviation": float(precip_level.loc[as_of]) - baseline_precip,
                "co2": float(co2.loc[as_of]),
                "dev_from_trend_ocean_temp": float(trend_deviation.loc[as_of]),
            }
        )

    forecast_rows = pl.DataFrame(rows, schema={name: dtype for name, dtype in historical_rows.schema.items()})

    return pl.concat([historical_rows, forecast_rows]).sort("year")
