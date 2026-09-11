from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import preliz as pz
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from preliz.distributions.distributions import Distribution

from climate_risk.data.pwt import load_pwt_data
from climate_risk.data.world_bank import load_wb_data
from climate_risk.data_functions.combine_data import build_country_year_panel, build_time_series
from climate_risk.models.climate_economy import OBSERVED, PREINDUSTRIAL_CO2_PPM, SHOCKS, ClimateEconomy
from climate_risk.sample import sample_or_load

# Elicitation runs inside model-building code, where a figure per call is noise, and this also
# changes what pz.maxent returns: with plotting on it hands back a (distribution, axes) pair.
pz.rcParams["plots.show_plot"] = False

# Maximum entropy subject to the stated interval: the least committal member of the family that
# respects it. Gamma keeps the scales off zero, where a variance decomposition has no gradient.
STATE_SCALES = {
    "co2": (0.0005, 0.005),
    "frontier": (0.002, 0.020),
    "pop": (0.0002, 0.002),
    "emp_rate": (0.001, 0.010),
    "gap": (0.005, 0.040),
    "ky": (0.005, 0.040),
    "precip": (0.020, 0.200),
    "temp": (0.005, 0.050),
}
OBS_SCALES = {
    "log_co2": (0.0001, 0.002),
    "log_pop": (0.00001, 0.001),
    "log_gdp": (0.002, 0.020),
    "temp": (0.02, 0.30),
    "precip": (0.020, 0.150),
    "log_emp_rate": (0.002, 0.030),
    "log_ky": (0.002, 0.030),
    "log_tfp_frontier": (0.002, 0.030),
}

# IPCC AR6's likely range for the transient climate response, and the labour-share and persistence
# ranges the Costa Rica notebook elicited. None of these depend on which country is being fit.
CLIMATE_RESPONSE_RANGE = (1.4, 2.2)
THETA_RANGE = (0.50, 0.70)
PHI_POP_RANGE = (0.93, 0.998)
PHI_EMP_RANGE = (0.90, 0.995)

# Where DICE-2016R puts marginal damage around 2 degrees, with enough spread to reach the larger
# values panel studies of hot countries report.
DAMAGE_ELASTICITY_MU = 0.02
DAMAGE_ELASTICITY_SIGMA = 0.04

# The centre of the elicited labour-share interval, used only to place anchors and as a fallback
# labour share when a country's own is not measured.
THETA_CENTRE = 0.60

CLIMATE_RESPONSE_CENTRE = 1.8


def solow_residual(*, code: str, panel: pd.DataFrame) -> pd.Series:
    """
    Log productivity for one country, from the identity the model uses.

    Parameters
    ----------
    code : str
        ISO country code.
    panel : DataFrame
        The Penn World Table panel, keyed by country code and year.

    Returns
    -------
    log_tfp : Series
        Log productivity centred on its own mean, indexed by date. The series is an index on an
        arbitrary base, so only its movement is comparable to anything else.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.pwt import load_pwt_data
        from climate_risk.models.climate_economy_fit import solow_residual

        pwt = load_pwt_data(Path("data")).to_pandas()
        log_tfp = solow_residual(code="USA", panel=pwt)
    """
    block = panel.loc[panel["country_code"] == code].set_index("year")
    share = block["labour_share"].mean()
    log_tfp: pd.Series = (
        np.log(block["pwt_real_gdp"]) - (1 - share) * np.log(block["capital"]) - share * np.log(block["employment"])
    )
    centred: pd.Series = log_tfp - log_tfp.mean()

    return centred


@dataclass(frozen=True, slots=True)
class ObservedSeries:
    """
    The eight series :class:`~climate_risk.models.climate_economy.ClimateEconomy` is fit against.

    Parameters
    ----------
    observed : DataFrame
        Columns :data:`~climate_risk.models.climate_economy.OBSERVED`, indexed by year.
    doublings : Series
        Doublings of atmospheric carbon above the pre-industrial level, indexed by year, covering
        at least ``observed``'s span.
    first : Timestamp
        First year any series in ``observed`` carries a value.
    last : Timestamp
        Last year any series in ``observed`` carries a value.
    pwt_last : Timestamp
        Last year the Penn World Table series (``log_emp_rate``, ``log_ky``, ``log_tfp_frontier``)
        carry a value. Earlier than ``last`` wherever the Table's coverage ends before the panel's.
    sample_years : float
        ``last`` minus ``first``, in years.
    """

    observed: pd.DataFrame
    doublings: pd.Series
    first: pd.Timestamp
    last: pd.Timestamp
    pwt_last: pd.Timestamp
    sample_years: float


def build_observed_series(*, iso: str, frontier: str, cache_dir: Path, start: str) -> ObservedSeries:
    """
    Assemble the eight series a country's climate-economy model is fit against.

    Parameters
    ----------
    iso : str
        ISO alpha-3 code of the country to fit.
    frontier : str
        ISO alpha-3 code of the productivity frontier the country is measured against, typically
        ``"USA"``.
    cache_dir : Path
        Directory the source caches live under.
    start : str
        First year to keep, as a four-digit string. Series with no coverage this early are simply
        null over the gap; the state-space model imputes it.

    Returns
    -------
    series : ObservedSeries

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.models.climate_economy_fit import build_observed_series

        series = build_observed_series(iso="NPL", frontier="USA", cache_dir=Path("data"), start="1990")
    """
    time_series = build_time_series(cache_dir).to_pandas().set_index("year")
    panel = build_country_year_panel(cache_dir).to_pandas()
    pwt = load_pwt_data(cache_dir).to_pandas()
    pwt["year"] = pd.to_datetime(pwt["year"], format="%Y")

    country = (
        panel.loc[panel["ISO"] == iso]
        .set_index("Start_Year")[["real_gdp_usd", "population", "precip"]]
        .rename_axis("year")
        .sort_index()
    )
    home = pwt.loc[pwt["country_code"] == iso].set_index("year")
    ocean_heat = time_series["Temp"]

    log_ky_raw = np.log(home["capital"] / home["pwt_real_gdp"])

    observed = pd.DataFrame(
        {
            "log_co2": np.log(time_series["co2"]),
            "log_pop": np.log(country["population"] / 1e6),
            "log_gdp": np.log(country["real_gdp_usd"] / 1e6),
            "temp": (ocean_heat - ocean_heat.mean()) / ocean_heat.std(),
            "precip": np.log(country["precip"]),
            "log_emp_rate": np.log(home["employment"] / home["pwt_population"]),
            "log_ky": log_ky_raw - log_ky_raw.mean(),
            "log_tfp_frontier": solow_residual(code=frontier, panel=pwt),
        }
    ).loc[start:][OBSERVED]

    index = observed["log_gdp"].dropna().index
    first, last = index.min(), index.max()
    ky_index = observed["log_ky"].dropna().index
    pwt_last = ky_index.max() if len(ky_index) else last
    sample_years = (last - first).days / 365.25

    doublings = np.log2(time_series["co2"] / PREINDUSTRIAL_CO2_PPM).loc[start:]

    return ObservedSeries(
        observed=observed, doublings=doublings, first=first, last=last, pwt_last=pwt_last, sample_years=sample_years
    )


@dataclass(frozen=True, slots=True)
class GrowthAccounting:
    """
    The rates and anchors a country's climate-economy priors are placed from.

    Parameters
    ----------
    labour_share : float
        Cobb-Douglas labour share, either the country's own Penn World Table average or
        ``labour_share_override``.
    labour_share_is_measured : bool
        Whether ``labour_share`` came from the country's own data.
    frontier_growth : float
        Annual growth of frontier productivity, continuously compounded.
    gap_drift : float
        Annual rate the country falls behind the frontier, ``frontier_growth`` minus the country's
        own productivity growth. ``0.0`` when the country's own growth is not measured.
    depreciation : float
        Average capital depreciation rate.
    population_growth_mean : float
        Average population growth over the full observed window.
    population_growth_recent : float
        Average population growth over the last five years of the window.
    gap_anchor : float
        The productivity gap to the frontier at ``first``, in log points behind.
    gap_anchor_is_measured : bool
        Whether ``gap_anchor`` came from the Penn World Table's own cross-country TFP level
        (``ctfp``), as opposed to the PPP GDP-per-worker proxy.
    gdp_intercept_anchor : float
        Where the output equation's constant sits at ``first``, solved from the identity.
    ky_slope_centre : float
        Centre of the capital-output ratio's response to population growth, from the Solow steady
        state.
    initial_level : ndarray
        Anchors for ``initial_level`` (frontier productivity, log population, log employment rate),
        in that order.
    initial_growth : ndarray
        Anchors for ``initial_growth`` (population growth, employment rate growth), in that order.
    """

    labour_share: float
    labour_share_is_measured: bool
    frontier_growth: float
    gap_drift: float
    depreciation: float
    population_growth_mean: float
    population_growth_recent: float
    gap_anchor: float
    gap_anchor_is_measured: bool
    gdp_intercept_anchor: float
    ky_slope_centre: float
    initial_level: np.ndarray
    initial_growth: np.ndarray


def growth_accounting(
    *,
    iso: str,
    frontier: str,
    series: ObservedSeries,
    pwt: pd.DataFrame,
    cache_dir: Path,
    labour_share_override: float | None = None,
) -> GrowthAccounting:
    """
    Compute the rates and anchors a country's climate-economy priors are placed from.

    Falls back gracefully where the Penn World Table's expanded national-accounts fields, capital
    stock, labour share and cross-country TFP, do not cover a country: this is common for smaller
    and lower-income economies, not a defect specific to any one of them. Without a labour share the
    caller must supply ``labour_share_override``. Without ``ctfp`` the initial gap is proxied from
    World Bank PPP GDP per worker instead of measured directly.

    Parameters
    ----------
    iso : str
        ISO alpha-3 code of the country being fit.
    frontier : str
        ISO alpha-3 code of the productivity frontier.
    series : ObservedSeries
        From :func:`build_observed_series`, for the same ``iso`` and ``frontier``.
    cache_dir : Path
        Directory the source caches live under. Only read from when ``ctfp`` is absent and the PPP
        gap proxy needs World Bank data.
    pwt : DataFrame
        The full Penn World Table panel, as :func:`~climate_risk.data.pwt.load_pwt_data` returns it.
    labour_share_override : float, optional
        Labour share to use when the country's own is not covered by the Penn World Table. Required
        in that case; ignored otherwise. Default None.

    Returns
    -------
    accounting : GrowthAccounting

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.pwt import load_pwt_data
        from climate_risk.models.climate_economy_fit import build_observed_series, growth_accounting

        pwt = load_pwt_data(Path("data")).to_pandas()
        series = build_observed_series(iso="NPL", frontier="USA", cache_dir=Path("data"), start="1990")
        accounting = growth_accounting(
            iso="NPL", frontier="USA", series=series, pwt=pwt, cache_dir=Path("data"), labour_share_override=0.487
        )
    """
    pwt = pwt.assign(year=pd.to_datetime(pwt["year"], format="%Y"))
    home = pwt.loc[pwt["country_code"] == iso].set_index("year")
    window = home.loc[series.first : series.pwt_last]

    if home["labour_share"].notna().any():
        labour_share = float(home.loc[series.first :, "labour_share"].mean())
        labour_share_is_measured = True
    else:
        if labour_share_override is None:
            raise ValueError(f"{iso} has no labour share in the Penn World Table; pass labour_share_override.")
        labour_share = labour_share_override
        labour_share_is_measured = False

    frontier_growth = float(
        (series.observed["log_tfp_frontier"].dropna().iloc[-1] - series.observed["log_tfp_frontier"].dropna().iloc[0])
        / series.sample_years
    )

    if home["capital"].notna().any():
        years_covered = (series.pwt_last - series.first).days / 365.25
        own_growth = {
            name: (np.log(window[name].iloc[-1]) - np.log(window[name].iloc[0])) / years_covered
            for name in ("pwt_real_gdp", "capital", "employment", "pwt_population")
        }
        tfp_growth = (
            own_growth["pwt_real_gdp"]
            - (1 - labour_share) * own_growth["capital"]
            - labour_share * own_growth["employment"]
        )
        gap_drift = float(frontier_growth - tfp_growth)
    else:
        gap_drift = 0.0

    depreciation = float(window["depreciation"].mean())
    population_growth = series.observed["log_pop"].dropna().diff().dropna()
    population_growth_mean = float(population_growth.mean())
    population_growth_recent = float(population_growth.iloc[-5:].mean())

    if home["tfp_relative_to_usa"].notna().any():
        ctfp_first = home.loc[[series.first], "tfp_relative_to_usa"].to_numpy()[0]
        gap_anchor = float(-np.log(ctfp_first))
        gap_anchor_is_measured = True
    else:
        gap_anchor = _ppp_gap_proxy(iso=iso, frontier=frontier, year=series.first.year, cache_dir=cache_dir)
        gap_anchor_is_measured = False

    anchor_first = series.observed.loc[series.first]
    log_pop_series = series.observed["log_pop"].dropna()
    log_emp_series = series.observed["log_emp_rate"].dropna()

    gdp_intercept_anchor = float(
        anchor_first["log_gdp"]
        - (series.observed["log_tfp_frontier"].dropna().iloc[0] - gap_anchor) / THETA_CENTRE
        - anchor_first["log_pop"]
        - anchor_first["log_emp_rate"]
    )

    output_per_worker_growth = frontier_growth / THETA_CENTRE
    ky_slope_centre = 1.0 / (output_per_worker_growth + population_growth_recent + depreciation)

    initial_level = np.array(
        [series.observed["log_tfp_frontier"].dropna().iloc[0], anchor_first["log_pop"], anchor_first["log_emp_rate"]]
    )
    initial_growth = np.array(
        [
            (log_pop_series.iloc[5] - log_pop_series.iloc[0]) / 5,
            (log_emp_series.iloc[5] - log_emp_series.iloc[0]) / 5,
        ]
    )

    return GrowthAccounting(
        labour_share=labour_share,
        labour_share_is_measured=labour_share_is_measured,
        frontier_growth=frontier_growth,
        gap_drift=gap_drift,
        depreciation=depreciation,
        population_growth_mean=population_growth_mean,
        population_growth_recent=population_growth_recent,
        gap_anchor=gap_anchor,
        gap_anchor_is_measured=gap_anchor_is_measured,
        gdp_intercept_anchor=gdp_intercept_anchor,
        ky_slope_centre=ky_slope_centre,
        initial_level=initial_level,
        initial_growth=initial_growth,
    )


def _ppp_gap_proxy(*, iso: str, frontier: str, year: int, cache_dir: Path) -> float:
    """GDP-per-worker at PPP relative to the frontier, as a productivity gap when ctfp is absent.

    Reads the nearest year both PPP GDP per capita and the employment rate are covered, since the
    two series do not always start in the same year the anchor otherwise would.
    """
    wb = load_wb_data(cache_dir).to_pandas()

    def gdp_per_worker(code: str) -> float:
        covered = wb.loc[
            (wb["country_code"] == code) & wb["gdp_per_cap_ppp_usd"].notna() & wb["employment_rate"].notna()
        ]
        if covered.empty:
            raise ValueError(f"No PPP GDP per capita and employment rate ever both cover {code}.")
        row = covered.iloc[(covered["year"] - year).abs().argmin()]
        return float(row["gdp_per_cap_ppp_usd"] / (row["employment_rate"] / 100))

    return -float(np.log(gdp_per_worker(iso) / gdp_per_worker(frontier)))


def calibrate_warming_equation(
    *, climate_economy: ClimateEconomy, series: ObservedSeries, climate_response: float = CLIMATE_RESPONSE_CENTRE
) -> tuple[float, float, float, float]:
    """
    Calibrate the warming path and the equation relating it to observed ocean heat.

    Fits ocean heat to the warming path the observed carbon stock implies, and backs out how far
    warming sat below its equilibrium at ``series.first``: chasing a steadily rising forcing leaves
    a shortfall of ``step * persistence / (1 - persistence)``.

    Parameters
    ----------
    climate_economy : ClimateEconomy
        Instance to read the thermal persistence from.
    series : ObservedSeries
        From :func:`build_observed_series`.
    climate_response : float, optional
        Transient climate response, degrees per doubling, used only to place this warming path.
        Default :data:`CLIMATE_RESPONSE_CENTRE`.

    Returns
    -------
    warming_first : float
        Warming above pre-industrial at ``series.first``, net of the equilibrium lag.
    loading : float
        Standard deviations of ocean heat per degree of warming.
    intercept : float
        Where that line sits, in standard deviations.
    residual_sd : float
        Scatter about the line, the measurement error the series can support.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.models.climate_economy import ClimateEconomy
        from climate_risk.models.climate_economy_fit import build_observed_series, calibrate_warming_equation

        series = build_observed_series(iso="NPL", frontier="USA", cache_dir=Path("data"), start="1990")
        climate_economy = ClimateEconomy(population_growth_mean=0.01)
        warming_first, loading, intercept, residual_sd = calibrate_warming_equation(
            climate_economy=climate_economy, series=series
        )
    """
    persistence = climate_economy.thermal_persistence
    realised_drift = float(
        (series.doublings.loc[series.last] - series.doublings.loc[series.first]) / series.sample_years
    )
    warming_first = float(
        climate_response * series.doublings.loc[series.first]
        - climate_response * realised_drift * persistence / (1 - persistence)
    )

    path = [warming_first]
    for value in series.doublings.loc[series.first :].to_numpy()[:-1]:
        path.append(persistence * path[-1] + (1 - persistence) * climate_response * value)
    path_array = np.array(path)

    observed_temp = series.observed["temp"].to_numpy()
    mask = np.isfinite(observed_temp)
    loading, intercept = np.polyfit(path_array[mask], observed_temp[mask], 1)
    residual = observed_temp[mask] - (intercept + loading * path_array[mask])

    return warming_first, float(loading), float(intercept), float(residual.std())


def build_priors() -> dict[str, Distribution]:
    """
    Elicit the maximum-entropy priors that do not depend on which country is being fit.

    Parameters
    ----------
    None

    Returns
    -------
    priors : dict of str to Distribution
        One entry per shock scale (keyed by :data:`~climate_risk.models.climate_economy.SHOCKS`
        name) and per observed-series measurement-error scale (keyed by
        :data:`~climate_risk.models.climate_economy.OBSERVED` name), plus ``"climate_response"``,
        ``"theta"``, ``"phi_pop"`` and ``"phi_emp"``.

    Examples
    --------
    .. code-block:: python

        from climate_risk.models.climate_economy_fit import build_priors

        priors = build_priors()
        priors["theta"].plot_pdf()
    """
    state_priors = {
        name: pz.maxent(pz.Gamma(), lower=low, upper=high, mass=0.90) for name, (low, high) in STATE_SCALES.items()
    }
    obs_priors = {
        name: pz.maxent(pz.Gamma(), lower=low, upper=high, mass=0.90) for name, (low, high) in OBS_SCALES.items()
    }

    elicited = {
        "climate_response": pz.maxent(
            pz.Gamma(), lower=CLIMATE_RESPONSE_RANGE[0], upper=CLIMATE_RESPONSE_RANGE[1], mass=0.90
        ),
        "theta": pz.maxent(pz.Beta(), lower=THETA_RANGE[0], upper=THETA_RANGE[1], mass=0.90),
        "phi_pop": pz.maxent(pz.Beta(), lower=PHI_POP_RANGE[0], upper=PHI_POP_RANGE[1], mass=0.90),
        "phi_emp": pz.maxent(pz.Beta(), lower=PHI_EMP_RANGE[0], upper=PHI_EMP_RANGE[1], mass=0.90),
    }

    return {**state_priors, **obs_priors, **elicited}


def build_model(
    *,
    iso: str,
    frontier: str,
    series: ObservedSeries,
    accounting: GrowthAccounting,
    priors: dict[str, Distribution],
) -> tuple[pm.Model, ClimateEconomy]:
    """
    Assemble a country's climate-economy model, ready to sample.

    Parameters
    ----------
    iso : str
        ISO alpha-3 code of the country being fit. Used only to compute ``temp_loading``'s and
        ``ky_slope``'s calibrated centres.
    frontier : str
        ISO alpha-3 code of the productivity frontier.
    series : ObservedSeries
        From :func:`build_observed_series`.
    accounting : GrowthAccounting
        From :func:`growth_accounting`, for the same country.
    priors : dict of str to Distribution
        From :func:`build_priors`.

    Returns
    -------
    model : Model
        A PyMC model, not yet sampled.
    climate_economy : ClimateEconomy
        The state-space object the model was built from, needed to smooth and forecast the fit.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.pwt import load_pwt_data
        from climate_risk.models.climate_economy_fit import (
            build_model,
            build_observed_series,
            build_priors,
            growth_accounting,
        )

        cache_dir = Path("data")
        pwt = load_pwt_data(cache_dir).to_pandas()
        series = build_observed_series(iso="NPL", frontier="USA", cache_dir=cache_dir, start="1990")
        accounting = growth_accounting(
            iso="NPL", frontier="USA", series=series, pwt=pwt, cache_dir=Path("data"), labour_share_override=0.487
        )
        model, climate_economy = build_model(
            iso="NPL", frontier="USA", series=series, accounting=accounting, priors=build_priors()
        )
    """
    climate_economy = ClimateEconomy(population_growth_mean=accounting.population_growth_mean)
    warming_first, temp_loading_centre, temp_intercept, _ = calibrate_warming_equation(
        climate_economy=climate_economy, series=series
    )
    loading_prior = pz.maxent(pz.Gamma(), lower=temp_loading_centre - 0.8, upper=temp_loading_centre + 0.8, mass=0.90)
    ky_slope_prior = pz.maxent(
        pz.Gamma(), lower=accounting.ky_slope_centre / 2, upper=accounting.ky_slope_centre * 1.6, mass=0.90
    )

    realised_co2_drift = float(
        (series.doublings.loc[series.last] - series.doublings.loc[series.first]) / series.sample_years
    )

    with pm.Model(coords=climate_economy.coords) as model:
        pm.Normal("initial_level", mu=accounting.initial_level, sigma=np.array([0.05, 0.1, 0.1]), dims=["level_block"])
        pm.Normal("initial_growth", mu=accounting.initial_growth, sigma=0.004, dims=["growth_block"])
        pm.Normal("initial_warming", mu=warming_first, sigma=0.05)
        pm.Normal("initial_co2", mu=float(series.doublings.loc[series.first]), sigma=0.05)
        pm.Normal("initial_gap", mu=accounting.gap_anchor, sigma=0.20)

        pm.Normal("drift_co2", mu=realised_co2_drift, sigma=0.002)
        pm.Normal("drift_frontier", mu=accounting.frontier_growth, sigma=0.004)
        pm.Normal("drift_gap", mu=accounting.gap_drift, sigma=0.006)
        pm.Normal("ky_mean", mu=0.0, sigma=0.05)
        ky_slope_prior.to_pymc(name="ky_slope")
        pm.Normal("gdp_intercept", mu=accounting.gdp_intercept_anchor, sigma=1.0)

        priors["climate_response"].to_pymc(name="climate_response")
        priors["theta"].to_pymc(name="theta")
        pm.Deterministic(
            "phi",
            pt.stack([priors["phi_pop"].to_pymc(name="phi_pop"), priors["phi_emp"].to_pymc(name="phi_emp")]),
            dims=["growth_block"],
        )
        pm.Beta("rho", alpha=2.0, beta=2.0, dims=["ar_block"])

        pm.Normal("damage_elasticity", mu=DAMAGE_ELASTICITY_MU, sigma=DAMAGE_ELASTICITY_SIGMA)
        loading_prior.to_pymc(name="temp_loading")
        pm.Normal("precip_loading", mu=0.0, sigma=0.3)
        pm.Normal(
            "obs_intercept",
            mu=np.array([temp_intercept, float(series.observed["precip"].mean())]),
            sigma=np.array([0.4, 0.3]),
            dims=["intercept"],
        )

        pm.Deterministic(
            "sigma_state",
            pt.stack([priors[name].to_pymc(name=f"sigma_state_{name}") for name in SHOCKS]),
            dims=["shock"],
        )
        pm.Deterministic(
            "sigma_obs",
            pt.stack([priors[name].to_pymc(name=f"sigma_obs_{name}") for name in OBSERVED]),
            dims=["observed_state"],
        )

        climate_economy.build_statespace_graph(series.observed)

    return model, climate_economy


def fit(
    *, model: pm.Model, idata_path: Path, force_resample: bool = False, sample_kwargs: dict | None = None
) -> xr.DataTree:
    """
    Sample a climate-economy model, or load a previous fit.

    Parameters
    ----------
    model : Model
        From :func:`build_model`.
    idata_path : Path
        Where the fit is saved to and loaded from.
    force_resample : bool, optional
        Sample again even when ``idata_path`` already holds a fit. Default False.
    sample_kwargs : dict, optional
        Extra keyword arguments for :func:`pymc.sample`. Default ``chains=6, tune=1000, draws=1000,
        target_accept=0.95``, the Costa Rica notebook's own settings.

    Returns
    -------
    idata : DataTree
        The posterior.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.models.climate_economy_fit import build_model, fit

        model, climate_economy = build_model(...)
        idata = fit(model=model, idata_path=Path("data/idata/climate_economy_npl.nc"))
    """
    defaults = {"chains": 6, "tune": 1000, "draws": 1000, "target_accept": 0.95, "random_seed": 0}
    resolved_kwargs = {**defaults, **(sample_kwargs or {})}

    return sample_or_load(
        str(idata_path), model=model, force_resample=force_resample, sample_kwargs=resolved_kwargs, save_results=True
    )


def forecast(
    *, climate_economy: ClimateEconomy, idata: xr.DataTree, start: pd.Timestamp, periods: int, random_seed: int = 0
) -> xr.DataTree:
    """
    Project a fitted climate-economy model forward.

    Parameters
    ----------
    climate_economy : ClimateEconomy
        From :func:`build_model`, for the same fit as ``idata``.
    idata : DataTree
        The posterior, from :func:`fit`.
    start : Timestamp
        Year to continue the forecast from, typically the fit's last observed year.
    periods : int
        Number of years to forecast.
    random_seed : int, optional
        Default 0.

    Returns
    -------
    ahead : DataTree
        Carries ``forecast_observed`` (dims ``observed_state``, ``time``) and ``forecast_latent``
        (dims ``state``, ``time``).

    Examples
    --------
    .. code-block:: python

        from climate_risk.models.climate_economy_fit import fit, forecast

        idata = fit(model=model, idata_path=idata_path)
        ahead = forecast(climate_economy=climate_economy, idata=idata, start=series.last, periods=76)
    """
    ahead: xr.DataTree = climate_economy.forecast(idata, start=start, periods=periods, random_seed=random_seed)

    return ahead
