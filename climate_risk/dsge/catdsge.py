import logging
import warnings

from typing import Any

import arviz as az
import gEconpy as ge
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from gEconpy.model.model import Model
from gEconpy.model.statespace import DSGEStateSpace
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import explicit_graph_inputs
from scipy.optimize import least_squares

from climate_risk.dsge.model_files import GCN_DIR

_log = logging.getLogger(__name__)

QUARTERS_PER_YEAR = 4
SAMPLE_START = "2012Q1"
SAMPLE_END = "2024Q4"
LOCKDOWN_START = "2020Q1"
LOCKDOWN_END = "2021Q4"

# The wage-setting identity's terms reach 1e8, so float rounding leaves its exact steady state outside gEconpy's default
# relative tolerance of 1e-10.
STEADY_STATE_RTOL = 1e-8


def configure_session() -> None:
    """
    Set up a notebook session for the CatDSGE model, changing process-wide settings.

    Quiets gEconpy's logging and model-building output and the statespace's missing-data warning, and loosens gEconpy's
    steady-state tolerance to ``STEADY_STATE_RTOL`` for steady states solved directly. The functions of this module
    that solve a steady state set that tolerance themselves, so none of them depends on this call.

    Examples
    --------
    Call it once at the top of a notebook:

    .. code-block:: python

        from climate_risk.dsge.catdsge import configure_session

        configure_session()
    """
    logging.getLogger("gEconpy").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message="Provided data contains missing values")

    ge.rcParams["build.verbose"] = False
    ge.rcParams["steady_state.rtol"] = STEADY_STATE_RTOL


def compute_data_targets(panel: pd.DataFrame) -> pd.Series:
    """
    Measure India's great ratios over the estimation window, the targets the calibration matches.

    Parameters
    ----------
    panel : DataFrame
        The India panel indexed by quarterly period, as
        :func:`~climate_risk.data_functions.india_panel.load_india_quarterly_panel` returns it with its index
        converted to periods.

    Returns
    -------
    targets : Series
        Expenditure shares of GDP, the labor share by the PWT and the ILO, and government debt over annual GDP. Debt is
        averaged over the readings inside the window, the same fiscal years :func:`build_observed` observes.

    Examples
    --------
    The targets for the panel the model is estimated on:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data_functions.india_panel import load_india_quarterly_panel
        from climate_risk.dsge.catdsge import compute_data_targets

        panel = load_india_quarterly_panel(Path("data"))
        panel.index = panel.index.to_period("Q")

        targets = compute_data_targets(panel)
    """
    window = panel.loc[SAMPLE_START:SAMPLE_END]
    real_totals = window[["gdp", "consumption", "investment", "government", "exports", "imports"]].sum()

    return pd.Series(
        {
            "consumption / GDP": window["nominal_consumption_share"].mean(),
            "investment / GDP": window["nominal_investment_share"].mean(),
            "government / GDP": real_totals["government"] / real_totals["gdp"],
            "exports / GDP": real_totals["exports"] / real_totals["gdp"],
            "imports / GDP": real_totals["imports"] / real_totals["gdp"],
            "labor share": window["labour_share_pwt"].mean(),
            "labor share (ILO)": window["labour_share_ilo"].mean(),
            "government debt / annual GDP": window["government_debt"].mean(),
        },
        name="India, 2012 to 2024",
    )


MODEL_RATIOS = [
    "consumption / GDP",
    "investment / GDP",
    "government / GDP",
    "exports / GDP",
    "imports / GDP",
    "labor share",
    "government debt / annual GDP",
]


def model_ratios(model: Model, **parameter_updates: float) -> pd.Series:
    """
    Compute the steady-state expenditure and income shares of GDP at the given parameters.

    Parameters
    ----------
    model : Model
        The CatDSGE model.
    **parameter_updates : float
        Parameter values replacing the model's own.

    Returns
    -------
    ratios : Series
        The ratios :func:`compute_data_targets` measures in the data, all missing where the steady state does not
        solve.

    Examples
    --------
    The great ratios at the India calibration:

    .. code-block:: python

        from pathlib import Path

        import gEconpy as ge

        from climate_risk.data_functions.india_panel import load_india_quarterly_panel
        from climate_risk.dsge.catdsge import calibrate_india, model_ratios
        from climate_risk.dsge.model_files import GCN_DIR

        model = ge.model_from_gcn(GCN_DIR / "catdsge_v4.gcn")
        panel = load_india_quarterly_panel(Path("data"))
        panel.index = panel.index.to_period("Q")

        ratios = model_ratios(model, **calibrate_india(model, panel))
    """
    with ge.rc_context({"steady_state.rtol": STEADY_STATE_RTOL}):
        steady_state = model.steady_state(verbose=False, progressbar=False, **parameter_updates)
    if not steady_state.success:
        return pd.Series(np.nan, index=MODEL_RATIOS)

    gdp = steady_state["GDP_ss"]
    world_price_imports = (
        steady_state["q_ss"] * steady_state["p_M_star_ss"] * steady_state["nu_M_ss"] * steady_state["M_ss"]
    )

    return pd.Series(
        {
            "consumption / GDP": steady_state["C_ss"] / gdp,
            "investment / GDP": steady_state["I_ss"] / gdp,
            "government / GDP": steady_state["G_ss"] / gdp,
            "exports / GDP": steady_state["p_X_ss"] * steady_state["X_ss"] / gdp,
            "imports / GDP": world_price_imports / gdp,
            "labor share": steady_state["w_ss"] * steady_state["L_ss"] / gdp,
            "government debt / annual GDP": steady_state["B_ss"] / (QUARTERS_PER_YEAR * gdp),
        }
    )


CALIBRATED_PARAMETERS = {
    "psi_G": (0.01, 0.5),
    "alpha_Y": (0.05, 0.99),
    "phi_Y_star": (0.01, 5.0),
    "alpha": (0.1, 0.7),
    "b_target": (1.0, 6.0),
}
CALIBRATED_RATIOS = [
    "government / GDP",
    "imports / GDP",
    "exports / GDP",
    "investment / GDP",
    "labor share",
    "government debt / annual GDP",
]
MISSING_STEADY_STATE_PENALTY = 10.0


def calibrate(model: Model, quarterly_base: dict[str, float], data_targets: pd.Series) -> dict[str, float]:
    """
    Find the values of ``CALIBRATED_PARAMETERS`` whose steady state matches ``CALIBRATED_RATIOS`` in least squares.

    Six ratios meet five parameters, so the fit leaves residuals, which are logged at INFO. A fit that does not
    converge, or whose steady state does not solve, raises.

    Parameters
    ----------
    model : Model
        The CatDSGE model.
    quarterly_base : dict of str to float
        Values of the parameters held fixed during the fit.
    data_targets : Series
        The targets, from :func:`compute_data_targets`.

    Returns
    -------
    calibrated : dict of str to float
        The fitted value of each of ``CALIBRATED_PARAMETERS``.

    Examples
    --------
    Fit the calibrated parameters with every other parameter at the model file's values:

    .. code-block:: python

        from pathlib import Path

        import gEconpy as ge

        from climate_risk.data_functions.india_panel import load_india_quarterly_panel
        from climate_risk.dsge.catdsge import calibrate, compute_data_targets
        from climate_risk.dsge.model_files import GCN_DIR

        model = ge.model_from_gcn(GCN_DIR / "catdsge_v4.gcn")
        panel = load_india_quarterly_panel(Path("data"))
        panel.index = panel.index.to_period("Q")

        calibrated = calibrate(model, quarterly_base={}, data_targets=compute_data_targets(panel))
    """
    targets = data_targets[CALIBRATED_RATIOS]

    def target_gap(values: np.ndarray) -> np.ndarray:
        updates = dict(zip(CALIBRATED_PARAMETERS, values, strict=True))
        ratios = model_ratios(model, **quarterly_base, **updates)
        gaps = (ratios[targets.index] - targets).to_numpy()
        return np.where(np.isfinite(gaps), gaps, MISSING_STEADY_STATE_PENALTY)

    lower, upper = zip(*CALIBRATED_PARAMETERS.values(), strict=True)
    debt_target = QUARTERS_PER_YEAR * data_targets["government debt / annual GDP"]
    starting_guesses = {**model.parameters(), "b_target": debt_target}
    starting_values = np.clip([starting_guesses[name] for name in CALIBRATED_PARAMETERS], lower, upper)
    fit = least_squares(target_gap, x0=starting_values, bounds=(lower, upper))
    calibrated = {name: float(value) for name, value in zip(CALIBRATED_PARAMETERS, fit.x, strict=True)}
    residuals = model_ratios(model, **quarterly_base, **calibrated)[targets.index] - targets

    if not fit.success:
        raise RuntimeError(f"The calibration did not converge: {fit.message}")
    if not np.isfinite(residuals).all():
        raise RuntimeError(f"The steady state does not solve at the calibrated values {calibrated}.")
    _log.info(f"Calibration residuals, model minus data: {residuals.round(4).to_dict()}")

    return calibrated


def calibrate_india(model: Model, panel: pd.DataFrame) -> dict[str, float]:
    """
    Calibrate the CatDSGE model to India at a quarterly frequency.

    The model files give every shock persistence ``rho_*`` other than ``rho_i`` at an annual frequency, and each is
    converted to quarterly here, so those values have to stay annual in the model files. The remaining fixed
    parameters are set from the literature and the panel, and :func:`calibrate` fits ``CALIBRATED_PARAMETERS`` on top.

    Parameters
    ----------
    model : Model
        The CatDSGE model, built from ``catdsge_v4.gcn``.
    panel : DataFrame
        The India panel indexed by quarterly period.

    Returns
    -------
    parameters : dict of str to float
        Every parameter the calibration sets. Pass them to ``model.parameters`` for the full parameter set.

    Examples
    --------
    The calibrated parameters, then every model parameter with the calibration applied:

    .. code-block:: python

        from pathlib import Path

        import gEconpy as ge

        from climate_risk.data_functions.india_panel import load_india_quarterly_panel
        from climate_risk.dsge.catdsge import calibrate_india
        from climate_risk.dsge.model_files import GCN_DIR

        model = ge.model_from_gcn(GCN_DIR / "catdsge_v4.gcn")
        panel = load_india_quarterly_panel(Path("data"))
        panel.index = panel.index.to_period("Q")

        india_parameters = calibrate_india(model, panel)
        model_parameters = model.parameters(**india_parameters)
    """
    shock_persistence = {
        name: value ** (1 / QUARTERS_PER_YEAR)
        for name, value in model.parameters().items()
        if name.startswith("rho_") and name != "rho_i"
    }
    trend_growth = np.log(panel.loc[SAMPLE_START:SAMPLE_END, "gdp"]).diff().mean()
    elasticity_of_substitution = 1 + 1 / 0.14

    quarterly_base = {
        **shock_persistence,
        "beta": 0.995,
        "delta": 0.025 + trend_growth,
        "pi_bar": 1.047 ** (1 / QUARTERS_PER_YEAR),
        "psi_H": elasticity_of_substitution,
        "psi_M": elasticity_of_substitution,
        "psi_X": elasticity_of_substitution,
        "tau_M": 0.06,
        "tau_X": 0.0,
        "tau_K": 0.0,
        "phi_H": 0.3,
        "chi": 2.0,
        "gamma_I": 5.0,
        "theta_H": 0.63,
        "theta_M": 0.63,
        "theta_X": 0.63,
        "theta_w": 0.75,
        "gamma_H": 0.2,
        "gamma_M": 0.2,
        "gamma_X": 0.2,
        "gamma_w": 0.5,
        "phi_pi": 1.5,
        "phi_y": 0.125,
        "rho_i": 0.85,
        "phi_B": QUARTERS_PER_YEAR * -0.10,
    }

    calibrated = calibrate(model=model, quarterly_base=quarterly_base, data_targets=compute_data_targets(panel))

    return {**quarterly_base, **calibrated}


DISASTER_SHOCK = "epsilon_xi_K"
SHOCK_SIZES = {
    DISASTER_SHOCK: -0.10,
    "epsilon_A": -0.01,
    "epsilon_i": 0.01 / QUARTERS_PER_YEAR,
    "epsilon_G": 0.05,
    "epsilon_p_M_star": 0.10,
    "epsilon_q": 0.05,
    "epsilon_Y_star": -0.01,
}
VARIABLE_LABELS = {
    "GDP": "GDP (%)",
    "C": "consumption (%)",
    "I": "investment (%)",
    "K": "capital stock (%)",
    "L": "hours (%)",
    "z": "utilization (%)",
    "w": "real wage (%)",
    "pi": "inflation (pp, annual)",
    "i": "policy rate (pp, annual)",
    "X": "exports (%)",
    "M": "imports (%)",
    "B": "government debt (%)",
}
IRF_QUANTILES = [0.03, 0.5, 0.97]


def to_display_units(irf: xr.DataArray, variable: str, policy_rate_ss: float) -> xr.DataArray:
    """
    Convert one variable's log-deviation response to the units it is reported in.

    Parameters
    ----------
    irf : DataArray
        Responses in log deviations from steady state, with a ``variable`` dimension.
    variable : str
        The variable to convert.
    policy_rate_ss : float
        Steady-state gross quarterly policy rate, which converts the rate's log deviation to percentage points.

    Returns
    -------
    response : DataArray
        Percent deviation for quantities, and annualized percentage points for inflation ``pi`` and the policy rate
        ``i``.

    Examples
    --------
    A 1% quarterly inflation response reads as 4 annualized percentage points:

    .. code-block:: python

        import xarray as xr

        from climate_risk.dsge.catdsge import to_display_units

        irf = xr.DataArray([0.01, 0.01], dims=["variable"], coords={"variable": ["pi", "GDP"]})
        inflation = to_display_units(irf, variable="pi", policy_rate_ss=1.015)
    """
    response = irf.sel(variable=variable)
    if variable == "pi":
        return 100 * QUARTERS_PER_YEAR * response
    if variable == "i":
        rate_change: xr.DataArray = xr.ufuncs.expm1(response)
        return 100 * QUARTERS_PER_YEAR * policy_rate_ss * rate_change

    return 100 * response


def responses_in_display_units(irf: xr.DataArray, variables: list[str], policy_rate_ss: float) -> xr.DataArray:
    """
    Convert the responses of several variables to the units each is reported in.

    Parameters
    ----------
    irf : DataArray
        Responses in log deviations from steady state, with a ``variable`` dimension.
    variables : list of str
        The variables to keep, in order.
    policy_rate_ss : float
        Steady-state gross quarterly policy rate.

    Returns
    -------
    responses : DataArray
        ``irf`` restricted to ``variables``, each in the units :func:`to_display_units` gives it.

    Examples
    --------
    GDP in percent and inflation in annualized percentage points, side by side:

    .. code-block:: python

        import xarray as xr

        from climate_risk.dsge.catdsge import responses_in_display_units

        irf = xr.DataArray([0.01, 0.01], dims=["variable"], coords={"variable": ["pi", "GDP"]})
        responses = responses_in_display_units(irf, variables=["GDP", "pi"], policy_rate_ss=1.015)
    """
    return xr.concat(
        [to_display_units(irf=irf, variable=variable, policy_rate_ss=policy_rate_ss) for variable in variables],
        dim="variable",
    )


def response_quantiles(
    irf: xr.DataArray, variables: list[str], policy_rate_ss: float, solved: xr.DataArray
) -> xr.DataArray:
    """
    Summarize the posterior responses by their median and 94% interval, over the draws whose model solved.

    Parameters
    ----------
    irf : DataArray
        Responses in log deviations, with ``chain``, ``draw`` and ``variable`` dimensions.
    variables : list of str
        The variables to keep, in order.
    policy_rate_ss : float
        Steady-state gross quarterly policy rate.
    solved : DataArray of bool
        Whether each draw's linear solution converged, over ``chain`` and ``draw``.

    Returns
    -------
    quantiles : DataArray
        The responses in display units at ``IRF_QUANTILES``, over a ``quantile`` dimension.

    Examples
    --------
    Three draws, the last unsolved and so left out:

    .. code-block:: python

        import numpy as np
        import xarray as xr

        from climate_risk.dsge.catdsge import response_quantiles

        irf = xr.DataArray(
            np.full((1, 3, 4, 1), 0.01), dims=["chain", "draw", "time", "variable"], coords={"variable": ["GDP"]}
        )
        solved = xr.DataArray([[True, True, False]], dims=["chain", "draw"])
        quantiles = response_quantiles(irf=irf, variables=["GDP"], policy_rate_ss=1.015, solved=solved)
    """
    display_irf = responses_in_display_units(irf=irf.where(solved), variables=variables, policy_rate_ss=policy_rate_ss)

    return display_irf.quantile(IRF_QUANTILES, dim=["chain", "draw"])


GROWTH_OBSERVABLES = {
    "gdp_growth": ("gdp", "GDP"),
    "consumption_growth": ("consumption", "C"),
    "investment_growth": ("investment", "I"),
    "export_growth": ("exports", "X"),
    "import_growth": ("imports", "M"),
}
# Observables whose measured log level carries i.i.d. noise, which enters their growth rate as u[] - u[-1].
LEVEL_NOISE_STATES = {"consumption_growth": "u_C"}
OPEN_ECONOMY_OBSERVATION_EQUATIONS = {
    "real_exchange_rate_growth": "log(q[]) - log(q[-1])",
    "foreign_rate": "log(1 + i_star[]) - log(1 + i_star[ss])",
    "foreign_inflation": "log(pi_star[])",
    "foreign_output_growth": "log(Y_star[]) - log(Y_star[-1])",
    "world_import_price_growth": "log(p_M_star[]) - log(p_M_star[-1])",
}
TAX_REVENUE = (
    "(tau_C * C[] + tau_K * r[] * z[] * K[-1] + tau_L[] * w[] * L[]"
    " + tau_M[] * q[] * p_M_star[] * nu_M[] * M[] + tau_X * p_X[] * X[])"
)
STEADY_STATE_TAX_REVENUE = TAX_REVENUE.replace("[-1]", "[ss]").replace("[]", "[ss]")
FISCAL_OBSERVATION_EQUATIONS = {
    "debt_ratio": f"log(B[]) - log({QUARTERS_PER_YEAR} * GDP[])",
    "tax_revenue_ratio": f"log({TAX_REVENUE}) - log(GDP[]) - (log({STEADY_STATE_TAX_REVENUE}) - log(GDP[ss]))",
}
ANNUAL_OBSERVABLES = [*FISCAL_OBSERVATION_EQUATIONS]
OBSERVED_STATES = [*GROWTH_OBSERVABLES, "pi", *OPEN_ECONOMY_OBSERVATION_EQUATIONS, *ANNUAL_OBSERVABLES]
MOMENT_STATES = [name for name in OBSERVED_STATES if name not in ANNUAL_OBSERVABLES]
TEMPORAL_AGGREGATION = {"tax_revenue_ratio": "mean"}
OBSERVABLE_SHORT_NAMES = {
    **{name: variable for name, (_, variable) in GROWTH_OBSERVABLES.items()},
    "pi": "pi",
    "real_exchange_rate_growth": "q",
    "foreign_rate": "i*",
    "foreign_inflation": "pi*",
    "foreign_output_growth": "Y*",
    "world_import_price_growth": "p_M*",
}


def growth_observation_equation(name: str, variable: str, level_noise: bool) -> str:
    """
    Write the observation equation of a growth observable.

    Parameters
    ----------
    name : str
        The observable, a key of ``GROWTH_OBSERVABLES``.
    variable : str
        The model variable whose log growth it measures.
    level_noise : bool
        Whether the model carries the level noise states of ``LEVEL_NOISE_STATES``.

    Returns
    -------
    equation : str
        The log growth of ``variable``, plus the change in the observable's level noise when ``level_noise`` is set
        and it has any.

    Examples
    --------
    Consumption growth with its level noise:

    .. code-block:: python

        from climate_risk.dsge.catdsge import growth_observation_equation

        equation = growth_observation_equation(name="consumption_growth", variable="C", level_noise=True)
    """
    equation = f"log({variable}[]) - log({variable}[-1])"
    if level_noise and name in LEVEL_NOISE_STATES:
        noise = LEVEL_NOISE_STATES[name]
        equation += f" + {noise}[] - {noise}[-1]"

    return equation


def configure_statespace(statespace: DSGEStateSpace, level_noise: bool) -> None:
    """
    Configure a CatDSGE statespace to observe ``OBSERVED_STATES``, each with measurement error.

    Parameters
    ----------
    statespace : DSGEStateSpace
        The statespace to configure, in place.
    level_noise : bool
        Whether the model carries the level noise states of ``LEVEL_NOISE_STATES``, which the growth observables then
        include.

    Examples
    --------
    :func:`load_estimation_statespace` and :func:`load_simulation_statespace` already call it. Configuring a statespace
    built by hand:

    .. code-block:: python

        import gEconpy as ge

        from climate_risk.dsge.catdsge import configure_statespace
        from climate_risk.dsge.model_files import GCN_DIR

        statespace = ge.statespace_from_gcn(GCN_DIR / "catdsge_v4.gcn", not_loglin_variables=["B_star", "T_lump"])
        configure_statespace(statespace, level_noise=False)
    """
    growth_equations = {
        name: growth_observation_equation(name=name, variable=variable, level_noise=level_noise)
        for name, (_, variable) in GROWTH_OBSERVABLES.items()
    }
    statespace.configure(
        observed_states=OBSERVED_STATES,
        measurement_error=OBSERVED_STATES,
        full_shock_covariance=True,
        ss_obs_intercept=["pi"],
        observation_equations={
            **growth_equations,
            **OPEN_ECONOMY_OBSERVATION_EQUATIONS,
            **FISCAL_OBSERVATION_EQUATIONS,
        },
        temporal_aggregation=TEMPORAL_AGGREGATION,
        aggregation_period=QUARTERS_PER_YEAR,
        solver="scan_cycle_reduction",
        mode="JAX",
        verbose=False,
    )


def load_estimation_statespace() -> DSGEStateSpace:
    """
    Build the configured estimation statespace, with no capital destruction shock and with consumption level noise.

    Returns
    -------
    statespace : DSGEStateSpace
        The statespace of ``catdsge_v4_estimation.gcn``, observing ``OBSERVED_STATES``.

    Examples
    --------
    The statespace :func:`build_pymc_model` takes:

    .. code-block:: python

        from climate_risk.dsge.catdsge import load_estimation_statespace

        statespace = load_estimation_statespace()
    """
    statespace = ge.statespace_from_gcn(
        GCN_DIR / "catdsge_v4_estimation.gcn", not_loglin_variables=["B_star", "T_lump", "u_C"]
    )
    configure_statespace(statespace, level_noise=True)

    return statespace


def load_simulation_statespace() -> DSGEStateSpace:
    """
    Build the configured simulation statespace, the estimation model with the capital destruction block added back.

    Returns
    -------
    statespace : DSGEStateSpace
        The statespace of ``catdsge_v4.gcn``, observing ``OBSERVED_STATES``.

    Examples
    --------
    The statespace disasters are simulated in:

    .. code-block:: python

        from climate_risk.dsge.catdsge import load_simulation_statespace

        simulation_statespace = load_simulation_statespace()
    """
    statespace = ge.statespace_from_gcn(
        GCN_DIR / "catdsge_v4.gcn", not_loglin_variables=["B_star", "T_lump", "U", "U_C"]
    )
    configure_statespace(statespace, level_noise=False)

    return statespace


def build_observed(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build the observables the model is fit to over ``SAMPLE_START`` to ``SAMPLE_END``.

    Parameters
    ----------
    panel : DataFrame
        The India panel indexed by quarterly period.

    Returns
    -------
    observed : DataFrame
        One column per observable, indexed by quarter. The lockdown quarters are missing, growth rates and the open
        economy observables are demeaned, and the debt ratio is kept in log levels.

    Examples
    --------
    The observables, indexed by timestamp as the statespace reads them:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data_functions.india_panel import load_india_quarterly_panel
        from climate_risk.dsge.catdsge import build_observed

        panel = load_india_quarterly_panel(Path("data"))
        panel.index = panel.index.to_period("Q")

        observed = build_observed(panel)
        fit_data = observed.set_axis(observed.index.to_timestamp())
    """
    world_inflation = np.log(panel["world_price_level"]).diff()

    # The World Bank series covers central government tax revenue only, so its level sits far below the model's and
    # only deviations are observed. Its FY2022 reading, 6.9% against 10 to 12% in every earlier year, is a break in
    # the series.
    log_tax_revenue_ratio = np.log(panel["tax_revenue_gdp"]).loc[:"2022Q4"]

    # A rise in the BIS index is a real appreciation of the rupee, the opposite direction to the model's q.
    observed = pd.DataFrame(
        {
            **{name: np.log(panel[column]).diff() for name, (column, _) in GROWTH_OBSERVABLES.items()},
            "pi": np.log(panel["cpi"]).diff(),
            "real_exchange_rate_growth": -np.log(panel["real_effective_exchange_rate"]).diff(),
            "foreign_rate": np.log1p(panel["world_rate_3m"] / 100) / QUARTERS_PER_YEAR,
            "foreign_inflation": world_inflation,
            "foreign_output_growth": np.log(panel["world_output"]).diff(),
            "world_import_price_growth": np.log(panel["commodity_all"]).diff() - world_inflation,
            "debt_ratio": np.log(panel["government_debt"]),
            "tax_revenue_ratio": log_tax_revenue_ratio,
        }
    ).loc[SAMPLE_START:SAMPLE_END]
    observed.loc[LOCKDOWN_START:LOCKDOWN_END] = np.nan
    # National accounts exports jump 20% in 2012Q3 and fall 21% in 2012Q4, while customs goods exports in dollars fall
    # 3% and rise 2%, so both quarters are treated as missing.
    observed.loc["2012Q3":"2012Q4", "export_growth"] = np.nan

    demeaned = [*GROWTH_OBSERVABLES, *OPEN_ECONOMY_OBSERVATION_EQUATIONS, "tax_revenue_ratio"]
    observed[demeaned] -= observed[demeaned].mean()

    return observed


STRUCTURAL_PRIORS = {
    "phi_H": (pm.Beta, 0.30, 0.15),
    "chi": (pm.Gamma, 2.0, 1.0),
    "psi_z": (pm.Gamma, 0.20, 0.15),
    "gamma_I": (pm.Gamma, 5.0, 2.0),
    "theta_H": (pm.Beta, 0.63, 0.10),
    "gamma_H": (pm.Beta, 0.20, 0.10),
    "phi_pi": (pm.Gamma, 1.5, 0.25),
    "phi_y": (pm.Gamma, 0.125, 0.05),
    "rho_i": (pm.Beta, 0.85, 0.05),
    "phi_debt": (pm.Gamma, 0.083, 0.039),
}
WAGE_SLOPE_PRIOR = {"mu": -2.51, "sigma": 0.47}
ESTIMATED_STRUCTURAL = [*STRUCTURAL_PRIORS, "theta_w"]

# Units are those the shock enters its equation in: log deviations for quantities and prices, quarterly rates for
# the two policy rates, levels for the two tax rates, and shares of the reset price for the two trade markups. The
# modes put the prior predictive median sd of each quarterly observable near the data's, except GDP growth, CPI
# inflation and the foreign rate, which world import prices and foreign inflation propagate into.
SHOCK_SCALE_MODES = {
    "epsilon_A": 0.0021,
    "epsilon_beta": 0.0009,
    "epsilon_xi_I": 0.003,
    "epsilon_xi_H": 0.0025,
    "epsilon_xi_w": 0.0025,
    "epsilon_q": 0.013,
    "epsilon_p_M_star": 0.042,
    "epsilon_Y_star": 0.0018,
    "epsilon_pi_star": 0.0012,
    "epsilon_i_star": 0.001,
    "epsilon_i": 0.000375,
    "epsilon_G": 0.0045,
    "epsilon_tau_L": 0.00045,
    "epsilon_tau_M": 0.00045,
    "epsilon_xi_M": 0.0025,
    "epsilon_xi_X": 0.0025,
    "epsilon_xi_XD": 0.006,
}
SHOCK_SCALE_ALPHA = 3.0
SHOCK_CORRELATION_ETA = 4.0
NOISE_SHOCK_OBSERVABLES = {f"epsilon_{state}": name for name, state in LEVEL_NOISE_STATES.items()}

FOREIGN_OBSERVABLES = ["foreign_rate", "foreign_inflation", "foreign_output_growth", "world_import_price_growth"]
# Observed in levels, so their error shares are scaled by the variance of their period-to-period changes.
LEVEL_OBSERVABLES = ["foreign_rate", "debt_ratio", "tax_revenue_ratio"]
TYPICAL_ERROR_SHARES = {"domestic": 0.05, "foreign": 0.02}
ERROR_SHARE_LOGIT_SD = 0.3
ERROR_SHARE_SPREAD_PRIOR = {"sigma": 0.3}


def noise_sd_scales(fit_data: pd.DataFrame) -> dict[str, float]:
    """
    Set the prior scale of each level noise standard deviation.

    Parameters
    ----------
    fit_data : DataFrame
        The observables, from :func:`build_observed`.

    Returns
    -------
    scales : dict of str to float
        For each noise shock, the standard deviation that would explain all of its observable's variance.

    Examples
    --------
    The scale of the consumption level noise:

    .. code-block:: python

        import numpy as np
        import pandas as pd

        from climate_risk.dsge.catdsge import noise_sd_scales

        fit_data = pd.DataFrame({"consumption_growth": np.random.default_rng(0).normal(0, 0.01, 40)})
        scales = noise_sd_scales(fit_data)
    """
    return {shock: float(fit_data[name].std() / np.sqrt(2)) for shock, name in NOISE_SHOCK_OBSERVABLES.items()}


def error_scale(fit_data: pd.DataFrame, name: str) -> float:
    """
    Measure the variability that an observable's measurement error is a share of.

    Parameters
    ----------
    fit_data : DataFrame
        The observables, from :func:`build_observed`.
    name : str
        The observable.

    Returns
    -------
    scale : float
        The standard deviation of the observable over the periods it is observed, or of its period-to-period changes
        when it is one of ``LEVEL_OBSERVABLES``.

    Examples
    --------
    A level observable is scaled by its changes, so a steady trend has no scale:

    .. code-block:: python

        import pandas as pd

        from climate_risk.dsge.catdsge import error_scale

        fit_data = pd.DataFrame({"foreign_rate": [1.0, 1.1, 1.2, 1.3]})
        scale = error_scale(fit_data, "foreign_rate")
    """
    series = fit_data[name].dropna() if name in ANNUAL_OBSERVABLES else fit_data[name]

    return float(series.diff().std() if name in LEVEL_OBSERVABLES else series.std())


def fixed_parameters(statespace: DSGEStateSpace, model_parameters: dict[str, float]) -> dict[str, float]:
    """
    Select the statespace parameters held fixed in estimation.

    Parameters
    ----------
    statespace : DSGEStateSpace
        A configured CatDSGE statespace.
    model_parameters : dict of str to float
        Every model parameter, with the calibration applied, as ``model.parameters(**calibrate_india(model, panel))``
        returns them. The calibration alone does not name every parameter.

    Returns
    -------
    fixed : dict of str to float
        The value of each statespace parameter that is neither estimated nor part of the error or shock covariance.

    Examples
    --------
    The parameters the estimation holds at the model file's values:

    .. code-block:: python

        import gEconpy as ge

        from climate_risk.dsge.catdsge import fixed_parameters, load_estimation_statespace
        from climate_risk.dsge.model_files import GCN_DIR

        model = ge.model_from_gcn(GCN_DIR / "catdsge_v4.gcn")
        fixed = fixed_parameters(load_estimation_statespace(), model.parameters())
    """
    return {
        name: model_parameters[name]
        for name in statespace.param_names
        if not name.startswith("error_") and name != "state_cov" and name not in ESTIMATED_STRUCTURAL
    }


def wage_calvo_probability(slope: Any, discount: Any) -> Any:
    r"""
    Invert the slope of the wage Phillips curve for the Calvo probability of keeping a wage unchanged.

    The slope :math:`\kappa = (1 - \theta)(1 - \beta \theta) / \theta` has two roots in :math:`\theta`, and the one
    inside the unit interval is

    .. math::

        \theta = \frac{s - \sqrt{s^2 - 4 \beta}}{2 \beta}, \qquad s = 1 + \beta + \kappa.

    Parameters
    ----------
    slope : float, ndarray or TensorVariable
        The slope :math:`\kappa`, positive.
    discount : float, ndarray or TensorVariable
        The discount factor :math:`\beta`.

    Returns
    -------
    theta : float, ndarray or TensorVariable
        The Calvo probability, of the type of ``slope``.

    Examples
    --------
    A slope of 0.08 at a discount factor of 0.995:

    .. code-block:: python

        from climate_risk.dsge.catdsge import wage_calvo_probability

        theta_w = wage_calvo_probability(slope=0.08, discount=0.995)
    """
    slope_sum = 1 + discount + slope

    return (slope_sum - (slope_sum**2 - 4 * discount) ** 0.5) / (2 * discount)


def build_pymc_model(
    statespace: DSGEStateSpace, fit_data: pd.DataFrame, model_parameters: dict[str, float]
) -> pm.Model:
    """
    Build the CatDSGE estimation model.

    The model has the structural priors of ``STRUCTURAL_PRIORS``, a prior on the log slope of the wage Phillips curve
    that sets ``theta_w``, an LKJ shock covariance scaled by ``SHOCK_SCALE_MODES``, and measurement error as a share of
    each observable's variability.

    Parameters
    ----------
    statespace : DSGEStateSpace
        The configured estimation statespace, from :func:`load_estimation_statespace`.
    fit_data : DataFrame
        The observables, from :func:`build_observed`.
    model_parameters : dict of str to float
        Every model parameter, with the calibration applied, of which those not estimated are held fixed. See
        :func:`fixed_parameters`.

    Returns
    -------
    pymc_model : Model
        The model, with the statespace's likelihood built on ``fit_data``.

    Examples
    --------
    The estimation model on the India panel at its calibration:

    .. code-block:: python

        from pathlib import Path

        import gEconpy as ge

        from climate_risk.data_functions.india_panel import load_india_quarterly_panel
        from climate_risk.dsge.catdsge import (
            build_observed,
            build_pymc_model,
            calibrate_india,
            load_estimation_statespace,
        )
        from climate_risk.dsge.model_files import GCN_DIR

        model = ge.model_from_gcn(GCN_DIR / "catdsge_v4.gcn")
        panel = load_india_quarterly_panel(Path("data"))
        panel.index = panel.index.to_period("Q")
        model_parameters = model.parameters(**calibrate_india(model, panel))

        statespace = load_estimation_statespace()
        observed = build_observed(panel)
        pymc_model = build_pymc_model(
            statespace=statespace,
            fit_data=observed.set_axis(observed.index.to_timestamp()),
            model_parameters=model_parameters,
        )
    """
    estimated_shocks = list(statespace.shock_names)
    structural_shocks = [shock for shock in estimated_shocks if shock not in NOISE_SHOCK_OBSERVABLES]
    shock_scale_modes = np.array([SHOCK_SCALE_MODES[name] for name in structural_shocks])
    noise_scales = noise_sd_scales(fit_data)
    # Structural shocks first, then noise shocks, mapped back to the state space's shock order.
    shock_order = np.array([[*structural_shocks, *NOISE_SHOCK_OBSERVABLES].index(shock) for shock in estimated_shocks])
    discount_factor = model_parameters["beta"]

    observable_names = list(statespace.coords["observed_state"])
    observable_sd = np.array([error_scale(fit_data, name) for name in observable_names])
    error_groups = list(TYPICAL_ERROR_SHARES)
    error_group_index = np.array(
        [error_groups.index("foreign" if name in FOREIGN_OBSERVABLES else "domestic") for name in observable_names]
    )
    typical_error_logits = np.array([np.log(share / (1 - share)) for share in TYPICAL_ERROR_SHARES.values()])

    with pm.Model(coords=statespace.coords | {"error_group": error_groups}) as pymc_model:
        for name, (distribution, mean, sd) in STRUCTURAL_PRIORS.items():
            distribution(name, mu=mean, sigma=sd)
        for name, value in fixed_parameters(statespace, model_parameters).items():
            pm.Data(name, value)

        log_wage_slope = pm.Normal("log_wage_slope", **WAGE_SLOPE_PRIOR)
        pm.Deterministic("theta_w", wage_calvo_probability(slope=pm.math.exp(log_wage_slope), discount=discount_factor))

        standardized_cholesky, _, standardized_sigmas = pm.LKJCholeskyCov(
            "shock_cov",
            n=len(structural_shocks),
            eta=SHOCK_CORRELATION_ETA,
            sd_dist=pm.InverseGamma.dist(alpha=SHOCK_SCALE_ALPHA, beta=SHOCK_SCALE_ALPHA + 1),
            compute_corr=True,
        )
        shock_cholesky = shock_scale_modes[:, None] * standardized_cholesky
        shock_sigmas = shock_scale_modes * standardized_sigmas
        for index, shock in enumerate(structural_shocks):
            pm.Deterministic(f"sigma_{shock}", shock_sigmas[index])
        noise_sigmas = pt.stack(
            [pm.HalfNormal(f"sigma_{shock}", sigma=noise_scales[shock]) for shock in NOISE_SHOCK_OBSERVABLES]
        )

        block_cov = pt.linalg.block_diag(pm.math.dot(shock_cholesky, shock_cholesky.T), pt.diag(noise_sigmas**2))
        pm.Deterministic("state_cov", block_cov[shock_order][:, shock_order], dims=statespace.param_dims["state_cov"])

        error_share_logit_mean = pm.Normal(
            "error_share_logit_mean", mu=typical_error_logits, sigma=ERROR_SHARE_LOGIT_SD, dims="error_group"
        )
        error_share_spread = pm.HalfNormal("error_share_spread", **ERROR_SHARE_SPREAD_PRIOR)
        error_share_offset = pm.Normal("error_share_offset", dims="observed_state")
        error_share_logit = error_share_logit_mean[error_group_index] + error_share_spread * error_share_offset
        error_share = pm.Deterministic("error_share", pm.math.sigmoid(error_share_logit), dims="observed_state")
        for index, name in enumerate(observable_names):
            pm.Deterministic(f"error_sigma_{name}", pt.sqrt(error_share[index]) * observable_sd[index])

        statespace.build_statespace_graph(data=fit_data)

    return pymc_model


def statespace_trace(
    draws: xr.Dataset, parameter_values: dict[str, float], observed_data: xr.Dataset, group: str
) -> xr.DataTree:
    """
    Assemble posterior or prior draws into the trace the statespace samplers read.

    Parameters
    ----------
    draws : Dataset
        The draws, over ``chain`` and ``draw``, carrying ``phi_H``.
    parameter_values : dict of str to float
        The fixed parameters, broadcast over the draws.
    observed_data : Dataset
        The data, stored as the trace's ``observed_data`` group.
    group : str
        The group the draws go under, ``"posterior"`` or ``"prior"``.

    Returns
    -------
    trace : DataTree
        ``draws`` and the fixed parameters under ``group``, with ``observed_data`` and an empty ``constant_data``.

    Examples
    --------
    The posterior of a saved estimation, ready for ``statespace.sample_unconditional_posterior``:

    .. code-block:: python

        from pathlib import Path

        import gEconpy as ge
        import xarray as xr

        from climate_risk.data_functions.india_panel import load_india_quarterly_panel
        from climate_risk.dsge.catdsge import (
            calibrate_india,
            fixed_parameters,
            load_estimation_statespace,
            statespace_trace,
        )
        from climate_risk.dsge.model_files import GCN_DIR

        model = ge.model_from_gcn(GCN_DIR / "catdsge_v4.gcn")
        panel = load_india_quarterly_panel(Path("data"))
        panel.index = panel.index.to_period("Q")
        model_parameters = model.parameters(**calibrate_india(model, panel))

        statespace = load_estimation_statespace()
        trace = statespace_trace(
            draws=xr.open_datatree("catdsge_idata.nc")["posterior"].to_dataset(),
            parameter_values=fixed_parameters(statespace, model_parameters),
            observed_data=xr.Dataset(),
            group="posterior",
        )
    """
    held_fixed = xr.Dataset({name: xr.full_like(draws["phi_H"], value) for name, value in parameter_values.items()})

    return xr.DataTree.from_dict(
        {group: xr.merge([draws, held_fixed]), "observed_data": observed_data, "constant_data": xr.Dataset()}
    )


def simulation_trace(
    simulation_statespace: DSGEStateSpace,
    estimated_shocks: list[str],
    draws: xr.Dataset,
    model_parameters: dict[str, float],
    observed_data: xr.Dataset,
    group: str,
) -> xr.DataTree:
    """
    Carry estimated draws over to the simulation statespace, which adds the disaster shock.

    Parameters
    ----------
    simulation_statespace : DSGEStateSpace
        The statespace from :func:`load_simulation_statespace`.
    estimated_shocks : list of str
        The shocks of the estimation statespace, in the order of the ``shock`` coordinate of ``draws["state_cov"]``.
    draws : Dataset
        Draws of the estimation model, over ``chain`` and ``draw``, carrying ``state_cov``.
    model_parameters : dict of str to float
        Every model parameter, with the calibration applied. See :func:`fixed_parameters`.
    observed_data : Dataset
        The data, stored as the trace's ``observed_data`` group.
    group : str
        The group the draws go under, ``"posterior"`` or ``"prior"``.

    Returns
    -------
    trace : DataTree
        The trace :func:`statespace_trace` builds, with a shock covariance holding the estimated covariance and an
        uncorrelated disaster of standard deviation ``SHOCK_SIZES[DISASTER_SHOCK]``.

    Examples
    --------
    The posterior of a saved estimation in the simulation statespace:

    .. code-block:: python

        from pathlib import Path

        import gEconpy as ge
        import xarray as xr

        from climate_risk.data_functions.india_panel import load_india_quarterly_panel
        from climate_risk.dsge.catdsge import (
            calibrate_india,
            load_estimation_statespace,
            load_simulation_statespace,
            simulation_trace,
        )
        from climate_risk.dsge.model_files import GCN_DIR

        model = ge.model_from_gcn(GCN_DIR / "catdsge_v4.gcn")
        panel = load_india_quarterly_panel(Path("data"))
        panel.index = panel.index.to_period("Q")
        model_parameters = model.parameters(**calibrate_india(model, panel))

        trace = simulation_trace(
            simulation_statespace=load_simulation_statespace(),
            estimated_shocks=list(load_estimation_statespace().shock_names),
            draws=xr.open_datatree("catdsge_idata.nc")["posterior"].to_dataset(),
            model_parameters=model_parameters,
            observed_data=xr.Dataset(),
            group="posterior",
        )
    """
    simulation_shocks = list(simulation_statespace.shock_names)
    shock_embedding = np.array(
        [[float(shock == estimated) for estimated in estimated_shocks] for shock in simulation_shocks]
    )
    disaster_variance = np.diag(
        [SHOCK_SIZES[DISASTER_SHOCK] ** 2 * float(shock == DISASTER_SHOCK) for shock in simulation_shocks]
    )
    estimated_cov = draws["state_cov"].transpose("chain", "draw", "shock", "shock_aux").to_numpy()
    simulation_cov = xr.DataArray(
        shock_embedding @ estimated_cov @ shock_embedding.T + disaster_variance,
        dims=["chain", "draw", "shock", "shock_aux"],
        coords={
            "chain": draws["chain"],
            "draw": draws["draw"],
            "shock": simulation_shocks,
            "shock_aux": simulation_shocks,
        },
    )

    return statespace_trace(
        draws=draws.drop_vars(["state_cov", "shock", "shock_aux"]).assign(state_cov=simulation_cov),
        parameter_values=fixed_parameters(simulation_statespace, model_parameters),
        observed_data=observed_data,
        group=group,
    )


def bind_to_model(graphs: list[Any], model: pm.Model) -> list[Any]:
    """
    Replace the statespace parameter placeholders in ``graphs`` with the model variables of the same name.

    Parameters
    ----------
    graphs : list of TensorVariable
        Graphs built from a statespace's parameters, such as its transition matrix.
    model : Model
        The model whose variables replace the placeholders.

    Returns
    -------
    bound : list of TensorVariable
        ``graphs`` with every placeholder the model names replaced.

    Examples
    --------
    The simulation statespace's transition matrix as a function of the estimation model's parameters:

    .. code-block:: python

        from pathlib import Path

        import gEconpy as ge

        from climate_risk.data_functions.india_panel import load_india_quarterly_panel
        from climate_risk.dsge.catdsge import (
            bind_to_model,
            build_observed,
            build_pymc_model,
            calibrate_india,
            load_estimation_statespace,
            load_simulation_statespace,
        )
        from climate_risk.dsge.model_files import GCN_DIR

        model = ge.model_from_gcn(GCN_DIR / "catdsge_v4.gcn")
        panel = load_india_quarterly_panel(Path("data"))
        panel.index = panel.index.to_period("Q")
        model_parameters = model.parameters(**calibrate_india(model, panel))

        observed = build_observed(panel)
        pymc_model = build_pymc_model(
            statespace=load_estimation_statespace(),
            fit_data=observed.set_axis(observed.index.to_timestamp()),
            model_parameters=model_parameters,
        )
        (transition,) = bind_to_model([load_simulation_statespace().ssm["transition"]], model=pymc_model)
    """
    placeholders = [variable for variable in explicit_graph_inputs(graphs) if variable.name in model.named_vars]

    return list(
        graph_replace(graphs, replace={variable: model[variable.name] for variable in placeholders}, strict=False)
    )


def build_welfare_model(pymc_model: pm.Model, simulation_statespace: DSGEStateSpace) -> pm.Model:
    """
    Add the disaster's first-order welfare cost to a copy of the estimation model.

    The disaster is a one-time ``SHOCK_SIZES[DISASTER_SHOCK]`` capital destruction shock. Its discounted consumption,
    hours and wage-dispersion responses are valued at the steady-state marginal utilities, and expressed both as a
    share of the value of steady-state consumption forever and in quarters of steady-state consumption.

    Parameters
    ----------
    pymc_model : Model
        The estimation model, from :func:`build_pymc_model`. It is copied, not changed.
    simulation_statespace : DSGEStateSpace
        The statespace from :func:`load_simulation_statespace`.

    Returns
    -------
    welfare_model : Model
        The copy, with deterministics ``welfare_cost`` and ``compensating_transfer``, each also split into
        ``_consumption`` and ``_hours`` parts.

    Examples
    --------
    The welfare cost at each posterior draw of a saved estimation:

    .. code-block:: python

        from pathlib import Path

        import gEconpy as ge
        import pymc as pm
        import xarray as xr

        from climate_risk.data_functions.india_panel import load_india_quarterly_panel
        from climate_risk.dsge.catdsge import (
            build_observed,
            build_pymc_model,
            build_welfare_model,
            calibrate_india,
            load_estimation_statespace,
            load_simulation_statespace,
        )
        from climate_risk.dsge.model_files import GCN_DIR

        model = ge.model_from_gcn(GCN_DIR / "catdsge_v4.gcn")
        panel = load_india_quarterly_panel(Path("data"))
        panel.index = panel.index.to_period("Q")
        model_parameters = model.parameters(**calibrate_india(model, panel))

        observed = build_observed(panel)
        pymc_model = build_pymc_model(
            statespace=load_estimation_statespace(),
            fit_data=observed.set_axis(observed.index.to_timestamp()),
            model_parameters=model_parameters,
        )
        welfare_model = build_welfare_model(pymc_model, simulation_statespace=load_simulation_statespace())
        posterior = xr.open_datatree("catdsge_idata.nc")["posterior"].to_dataset()
        with welfare_model:
            welfare = pm.compute_deterministics(posterior, var_names=["welfare_cost", "compensating_transfer"])
    """
    simulation_states = list(simulation_statespace.state_names)
    simulation_shocks = list(simulation_statespace.shock_names)
    simulation_steady_state = {
        variable.name: value for variable, value in simulation_statespace.steady_state_mapping.items()
    }

    with pymc_model.copy() as welfare_model:
        transition, selection, consumption_ss, hours_ss, wage_dispersion_ss = bind_to_model(
            graphs=[
                simulation_statespace.ssm["transition"],
                simulation_statespace.ssm["selection"],
                simulation_steady_state["C_ss"],
                simulation_steady_state["L_ss"],
                simulation_steady_state["nu_w_ss"],
            ],
            model=welfare_model,
        )
        discount = welfare_model["beta"]
        habit = welfare_model["phi_H"]
        risk_aversion = welfare_model["sigma"]
        inverse_frisch = welfare_model["chi"]

        disaster_impact = selection[:, simulation_shocks.index(DISASTER_SHOCK)] * SHOCK_SIZES[DISASTER_SHOCK]
        discounted_response = pt.linalg.solve(pt.eye(len(simulation_states)) - discount * transition, disaster_impact)
        consumption_response, hours_response, wage_dispersion_response = (
            discounted_response[simulation_states.index(name)] for name in ["C", "L", "nu_w"]
        )

        habit_adjusted_consumption = consumption_ss * (1 - habit)
        labor_disutility = welfare_model["Theta"] * wage_dispersion_ss * hours_ss ** (1 + inverse_frisch)
        marginal_utility_of_consumption = habit_adjusted_consumption**-risk_aversion * (1 - discount * habit)
        consumption_value = marginal_utility_of_consumption * consumption_ss * consumption_response
        hours_value = -labor_disutility * (hours_response + wage_dispersion_response / (1 + inverse_frisch))
        value_of_permanent_consumption = habit_adjusted_consumption ** (1 - risk_aversion) / (1 - discount)
        value_of_one_quarter_of_consumption = marginal_utility_of_consumption * consumption_ss

        for suffix, value in [
            ("_consumption", consumption_value),
            ("_hours", hours_value),
            ("", consumption_value + hours_value),
        ]:
            pm.Deterministic(f"welfare_cost{suffix}", value / value_of_permanent_consumption)
            pm.Deterministic(f"compensating_transfer{suffix}", value / value_of_one_quarter_of_consumption)

    return welfare_model


def median_and_hdi(path: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Summarize a path by its posterior median and 94% HDI, over the draws on which it is fully observed.

    Parameters
    ----------
    path : DataArray
        Draws of a path, over ``chain``, ``draw`` and the path's own dimensions.

    Returns
    -------
    median : DataArray
        The median over draws.
    hdi : DataArray
        The 94% HDI over draws, with a ``ci_bound`` dimension.

    Examples
    --------
    A band around 40 quarters of a path, from two chains of 100 draws:

    .. code-block:: python

        import numpy as np
        import xarray as xr

        from climate_risk.dsge.catdsge import median_and_hdi

        path = xr.DataArray(np.random.default_rng(0).normal(size=(2, 100, 40)), dims=["chain", "draw", "time"])
        median, hdi = median_and_hdi(path)
    """
    draws = path.stack(sample=("chain", "draw")).drop_vars(["sample", "chain", "draw"]).dropna("sample", how="any")

    return draws.median("sample"), az.hdi(draws, prob=0.94, dim=["sample"])
