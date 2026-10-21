import logging
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from matplotlib.collections import QuadMesh
from matplotlib.colors import Normalize
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import explicit_graph_inputs
from scipy.optimize import least_squares

import gEconpy as ge

from gEconpy.model.model import Model
from gEconpy.model.statespace import DSGEStateSpace

from climate_risk.dsge.model_files import GCN_DIR
from climate_risk.plotting import PALETTE, configure_plot_style

QUARTERS_PER_YEAR = 4
SAMPLE_START = "2012Q1"
SAMPLE_END = "2024Q4"
LOCKDOWN_START = "2020Q1"
LOCKDOWN_END = "2021Q4"


def configure_session() -> None:
    configure_plot_style(add_grid=True)
    logging.getLogger("gEconpy").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message="Provided data contains missing values")

    ge.rcParams["build.verbose"] = False
    # The wage-setting identity's terms reach 1e8, so float rounding leaves its exact steady state outside the default
    # relative tolerance of 1e-10.
    ge.rcParams["steady_state.rtol"] = 1e-8


def compute_data_targets(panel: pd.DataFrame) -> pd.Series:
    window = panel.loc[SAMPLE_START:SAMPLE_END]
    # The panel places each fiscal year's debt on the first quarter of the next calendar year.
    debt_by_quarter = panel["government_debt"].dropna()
    government_debt = pd.Series(debt_by_quarter.to_numpy(), index=debt_by_quarter.index.year - 1)
    real_totals = window[["gdp", "consumption", "investment", "government", "exports", "imports"]].sum()
    return pd.Series({"consumption / GDP": window["nominal_consumption_share"].mean(),
                      "investment / GDP": window["nominal_investment_share"].mean(),
                      "government / GDP": real_totals["government"] / real_totals["gdp"],
                      "exports / GDP": real_totals["exports"] / real_totals["gdp"],
                      "imports / GDP": real_totals["imports"] / real_totals["gdp"],
                      "labor share": window["labour_share_pwt"].mean(),
                      "labor share (ILO)": window["labour_share_ilo"].mean(),
                      "government debt / annual GDP": government_debt.loc[2012:2024].mean()},
                     name="India, 2012 to 2024")


def model_ratios(model: Model, **parameter_updates: float) -> pd.Series:
    """Steady-state expenditure and income shares of GDP at the given parameters."""
    steady_state = model.steady_state(verbose=False,
                                      progressbar=False,
                                      **parameter_updates)
    gdp = steady_state["GDP_ss"]
    world_price_imports = (steady_state["q_ss"]
                           * steady_state["p_M_star_ss"]
                           * steady_state["nu_M_ss"]
                           * steady_state["M_ss"])
    return pd.Series({"consumption / GDP": steady_state["C_ss"] / gdp,
                      "investment / GDP": steady_state["I_ss"] / gdp,
                      "government / GDP": steady_state["G_ss"] / gdp,
                      "exports / GDP": steady_state["p_X_ss"] * steady_state["X_ss"] / gdp,
                      "imports / GDP": world_price_imports / gdp,
                      "labor share": steady_state["w_ss"] * steady_state["L_ss"] / gdp,
                      "government debt / annual GDP": steady_state["B_ss"] / (QUARTERS_PER_YEAR * gdp)})


CALIBRATED_PARAMETERS = {"psi_G": (0.01, 0.5),
                         "alpha_Y": (0.05, 0.99),
                         "phi_Y_star": (0.01, 5.0),
                         "alpha": (0.1, 0.7),
                         "b_target": (1.0, 6.0)}
CALIBRATED_RATIOS = ["government / GDP",
                     "imports / GDP",
                     "exports / GDP",
                     "investment / GDP",
                     "labor share",
                     "government debt / annual GDP"]
MISSING_STEADY_STATE_PENALTY = 10.0


def calibrate(model: Model, quarterly_base: dict[str, float], data_targets: pd.Series) -> dict[str, float]:
    """Least-squares values of `CALIBRATED_PARAMETERS` matching the model's ratios to `CALIBRATED_RATIOS`."""
    targets = data_targets[CALIBRATED_RATIOS]

    def target_gap(values):
        updates = dict(zip(CALIBRATED_PARAMETERS, values, strict=True))
        try:
            ratios = model_ratios(model, **quarterly_base, **updates)
        except Exception:
            return np.full(len(targets), MISSING_STEADY_STATE_PENALTY)
        gaps = (ratios[targets.index] - targets).to_numpy()
        return np.where(np.isfinite(gaps), gaps, MISSING_STEADY_STATE_PENALTY)

    lower, upper = zip(*CALIBRATED_PARAMETERS.values(), strict=True)
    debt_target = QUARTERS_PER_YEAR * data_targets["government debt / annual GDP"]
    starting_guesses = {**model.parameters(), "b_target": debt_target}
    starting_values = np.clip([starting_guesses[name] for name in CALIBRATED_PARAMETERS], lower, upper)
    fit = least_squares(target_gap,
                        x0=starting_values,
                        bounds=(lower, upper))
    return dict(zip(CALIBRATED_PARAMETERS, fit.x, strict=True))


def calibrate_india(model: Model, panel: pd.DataFrame) -> dict[str, float]:
    """The India calibration: parameters fixed before calibration, with the calibrated ones on top."""
    shock_persistence = {name: value ** (1 / QUARTERS_PER_YEAR)
                         for name, value in model.parameters().items()
                         if name.startswith("rho_") and name != "rho_i"}
    trend_growth = np.log(panel.loc[SAMPLE_START:SAMPLE_END, "gdp"]).diff().mean()
    elasticity_of_substitution = 1 + 1 / 0.14

    quarterly_base = {**shock_persistence,
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
                      "phi_B": QUARTERS_PER_YEAR * -0.10}

    calibrated = calibrate(model=model,
                           quarterly_base=quarterly_base,
                           data_targets=compute_data_targets(panel))
    return {**quarterly_base, **calibrated}


SOURCES = ["Sharma and Behera (2022)", "Sachdeva and Ahmad (2024)"]
SOURCE_COLORS = dict(zip(SOURCES, [PALETTE["primary"], "tab:green"], strict=True))


def plot_estimate(ax: plt.Axes, row, y_position: float) -> None:
    """One published estimate: its interval, dotted when prior-dominated, and its value, hollow when fixed."""
    color = SOURCE_COLORS[row.source]
    if np.isfinite(row.lower):
        ax.hlines(y=y_position,
                  xmin=row.lower,
                  xmax=row.upper,
                  color=color,
                  ls=":" if row.status == "prior-dominated" else "-",
                  lw=2.0)
    ax.scatter(x=row.value,
               y=y_position,
               s=50,
               color="white" if row.status == "fixed" else color,
               edgecolor=color,
               linewidth=1.5,
               zorder=3)


DISASTER_SHOCK = "epsilon_xi_K"
SHOCK_SIZES = {DISASTER_SHOCK: -0.10,
               "epsilon_A": -0.01,
               "epsilon_i": 0.01 / QUARTERS_PER_YEAR,
               "epsilon_G": 0.05,
               "epsilon_p_M_star": 0.10,
               "epsilon_q": 0.05,
               "epsilon_Y_star": -0.01}
VARIABLE_LABELS = {"GDP": "GDP (%)",
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
                   "B": "government debt (%)"}
IRF_QUANTILES = [0.03, 0.5, 0.97]


def to_display_units(irf: xr.DataArray, variable: str, policy_rate_ss: float) -> xr.DataArray:
    """Percent deviation for quantities, annualized percentage points for inflation and the policy rate."""
    response = irf.sel(variable=variable)
    if variable == "pi":
        return 100 * QUARTERS_PER_YEAR * response
    if variable == "i":
        return 100 * QUARTERS_PER_YEAR * policy_rate_ss * np.expm1(response)
    return 100 * response


def plot_impact_heatmap(ax: plt.Axes,
                        impact: xr.DataArray,
                        literature_points: pd.DataFrame,
                        xlabel: str,
                        ylabel: str,
                        yscale: str,
                        norm: Normalize) -> QuadMesh:
    """Draw the impact over a two-parameter grid, its zero contour, and the literature's estimates of both parameters.

    Parameters
    ----------
    ax : Axes
        The axes to draw on.
    impact : DataArray
        GDP at impact as a share, with dims ``(y, x)`` named for the two parameters.
    literature_points : DataFrame
        One row per source in ``SOURCE_COLORS``, with a column for each of the two parameters.
    xlabel, ylabel : str
        Axis labels.
    yscale : {"linear", "log"}
        Scale of the y axis.
    norm : Normalize
        Color normalization of the impact in percent, shared by every panel that shares a colorbar.

    Returns
    -------
    mesh : QuadMesh
        The heatmap, for the caller's colorbar.
    """
    y_name, x_name = impact.dims
    x_grid = impact[x_name].to_numpy()
    y_grid = impact[y_name].to_numpy()
    mesh = ax.pcolormesh(x_grid,
                         y_grid,
                         100 * impact.values,
                         cmap="RdBu",
                         norm=norm,
                         shading="auto")
    contour = ax.contour(x_grid,
                         y_grid,
                         impact.values,
                         levels=[0.0],
                         colors="k",
                         linewidths=1.5)
    ax.clabel(contour, fmt={0.0: "GDP = 0"})

    for source, color in SOURCE_COLORS.items():
        ax.scatter(x=literature_points.loc[source, x_name],
                   y=literature_points.loc[source, y_name],
                   s=90,
                   color=color,
                   edgecolor="k",
                   zorder=10,
                   label=source)

    ax.set(yscale=yscale,
           xlim=(x_grid[0], x_grid[-1]),
           ylim=(y_grid[0], y_grid[-1]),
           xlabel=xlabel,
           ylabel=ylabel)
    ax.grid(visible=False)
    return mesh


GROWTH_OBSERVABLES = {"gdp_growth": ("gdp", "GDP"),
                      "consumption_growth": ("consumption", "C"),
                      "investment_growth": ("investment", "I"),
                      "export_growth": ("exports", "X"),
                      "import_growth": ("imports", "M")}
# Observables whose measured log level carries i.i.d. noise, which enters their growth rate as u[] - u[-1].
LEVEL_NOISE_STATES = {"consumption_growth": "u_C"}
OPEN_ECONOMY_OBSERVATION_EQUATIONS = {"real_exchange_rate_growth": "log(q[]) - log(q[-1])",
                                      "foreign_rate": "log(1 + i_star[]) - log(1 + i_star[ss])",
                                      "foreign_inflation": "log(pi_star[])",
                                      "foreign_output_growth": "log(Y_star[]) - log(Y_star[-1])",
                                      "world_import_price_growth": "log(p_M_star[]) - log(p_M_star[-1])"}
TAX_REVENUE = ("(tau_C * C[] + tau_K * r[] * z[] * K[-1] + tau_L[] * w[] * L[]"
               " + tau_M[] * q[] * p_M_star[] * nu_M[] * M[] + tau_X * p_X[] * X[])")
STEADY_STATE_TAX_REVENUE = TAX_REVENUE.replace("[-1]", "[ss]").replace("[]", "[ss]")
FISCAL_OBSERVATION_EQUATIONS = {"debt_ratio": f"log(B[]) - log({QUARTERS_PER_YEAR} * GDP[])",
                                "tax_revenue_ratio": f"log({TAX_REVENUE}) - log(GDP[])"
                                                     f" - (log({STEADY_STATE_TAX_REVENUE}) - log(GDP[ss]))"}
ANNUAL_OBSERVABLES = [*FISCAL_OBSERVATION_EQUATIONS]
OBSERVED_STATES = [*GROWTH_OBSERVABLES, "pi", *OPEN_ECONOMY_OBSERVATION_EQUATIONS, *ANNUAL_OBSERVABLES]
MOMENT_STATES = [name for name in OBSERVED_STATES if name not in ANNUAL_OBSERVABLES]
TEMPORAL_AGGREGATION = {"tax_revenue_ratio": "mean"}


def growth_observation_equation(name: str, variable: str, level_noise: bool) -> str:
    equation = f"log({variable}[]) - log({variable}[-1])"
    if level_noise and name in LEVEL_NOISE_STATES:
        noise = LEVEL_NOISE_STATES[name]
        equation += f" + {noise}[] - {noise}[-1]"
    return equation


def configure_statespace(statespace: DSGEStateSpace, level_noise: bool) -> None:
    """Observe the fit variables, each with measurement error, and the level noise states when the model has them."""
    growth_equations = {name: growth_observation_equation(name=name,
                                                          variable=variable,
                                                          level_noise=level_noise)
                        for name, (_, variable) in GROWTH_OBSERVABLES.items()}
    statespace.configure(observed_states=OBSERVED_STATES,
                         measurement_error=OBSERVED_STATES,
                         full_shock_covariance=True,
                         ss_obs_intercept=["pi"],
                         observation_equations={**growth_equations,
                                                **OPEN_ECONOMY_OBSERVATION_EQUATIONS,
                                                **FISCAL_OBSERVATION_EQUATIONS},
                         temporal_aggregation=TEMPORAL_AGGREGATION,
                         aggregation_period=QUARTERS_PER_YEAR,
                         solver="scan_cycle_reduction",
                         mode="JAX",
                         verbose=False)


def load_estimation_statespace() -> DSGEStateSpace:
    """The estimation model: no capital destruction shock, and consumption level noise in a measurement block."""
    statespace = ge.statespace_from_gcn(GCN_DIR / "catdsge_v4_estimation.gcn",
                                        not_loglin_variables=["B_star", "T_lump", "u_C"])
    configure_statespace(statespace, level_noise=True)
    return statespace


def load_simulation_statespace() -> DSGEStateSpace:
    """The estimation model with the capital destruction block added back."""
    statespace = ge.statespace_from_gcn(GCN_DIR / "catdsge_v4.gcn",
                                        not_loglin_variables=["B_star", "T_lump", "U", "U_C"])
    configure_statespace(statespace, level_noise=False)
    return statespace


def build_observed(panel: pd.DataFrame) -> pd.DataFrame:
    """The fit observables over the sample, indexed by quarter, with the lockdown quarters missing."""
    world_inflation = np.log(panel["world_price_level"]).diff()

    # The World Bank series covers central government tax revenue only, so its level sits far below the model's and
    # only deviations are observed. Its FY2022 reading, 6.9% against 10 to 12% in every earlier year, is a break in
    # the series.
    log_tax_revenue_ratio = np.log(panel["tax_revenue_gdp"]).loc[:"2022Q4"]

    # A rise in the BIS index is a real appreciation of the rupee, the opposite direction to the model's q.
    observed = pd.DataFrame({**{name: np.log(panel[column]).diff()
                                for name, (column, _) in GROWTH_OBSERVABLES.items()},
                             "pi": np.log(panel["cpi"]).diff(),
                             "real_exchange_rate_growth": -np.log(panel["real_effective_exchange_rate"]).diff(),
                             "foreign_rate": np.log1p(panel["world_rate_3m"] / 100) / QUARTERS_PER_YEAR,
                             "foreign_inflation": world_inflation,
                             "foreign_output_growth": np.log(panel["world_output"]).diff(),
                             "world_import_price_growth": np.log(panel["commodity_all"]).diff() - world_inflation,
                             "debt_ratio": np.log(panel["government_debt"]),
                             "tax_revenue_ratio": log_tax_revenue_ratio}).loc[SAMPLE_START:SAMPLE_END]
    observed.loc[LOCKDOWN_START:LOCKDOWN_END] = np.nan
    # National accounts exports jump 20% in 2012Q3 and fall 21% in 2012Q4, while customs goods exports in dollars fall
    # 3% and rise 2%, so both quarters are treated as missing.
    observed.loc["2012Q3":"2012Q4", "export_growth"] = np.nan

    demeaned = [*GROWTH_OBSERVABLES, *OPEN_ECONOMY_OBSERVATION_EQUATIONS, "tax_revenue_ratio"]
    observed[demeaned] -= observed[demeaned].mean()
    return observed


STRUCTURAL_PRIORS = {"phi_H": (pm.Beta, 0.30, 0.15),
                     "chi": (pm.Gamma, 2.0, 1.0),
                     "psi_z": (pm.Gamma, 0.20, 0.15),
                     "gamma_I": (pm.Gamma, 5.0, 2.0),
                     "theta_H": (pm.Beta, 0.63, 0.10),
                     "gamma_H": (pm.Beta, 0.20, 0.10),
                     "phi_pi": (pm.Gamma, 1.5, 0.25),
                     "phi_y": (pm.Gamma, 0.125, 0.05),
                     "rho_i": (pm.Beta, 0.85, 0.05),
                     "phi_debt": (pm.Gamma, 0.083, 0.039)}
WAGE_SLOPE_PRIOR = {"mu": -2.51, "sigma": 0.47}
ESTIMATED_STRUCTURAL = [*STRUCTURAL_PRIORS, "theta_w"]

# Units are those the shock enters its equation in: log deviations for quantities and prices, quarterly rates for
# the two policy rates, levels for the two tax rates, and shares of the reset price for the two trade markups. The
# modes put the prior predictive median sd of each quarterly observable near the data's, except GDP growth, CPI
# inflation and the foreign rate, which world import prices and foreign inflation propagate into.
SHOCK_SCALE_MODES = {"epsilon_A": 0.0021,
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
                     "epsilon_xi_XD": 0.006}
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
    """Prior scale of each level noise sd: the noise that would explain all of its observable's variance."""
    return {shock: fit_data[name].std() / np.sqrt(2) for shock, name in NOISE_SHOCK_OBSERVABLES.items()}


def error_scale(fit_data: pd.DataFrame, name: str) -> float:
    """Sd of an observable, or of its changes for one observed in levels, over the periods it is observed."""
    series = fit_data[name].dropna() if name in ANNUAL_OBSERVABLES else fit_data[name]
    return series.diff().std() if name in LEVEL_OBSERVABLES else series.std()


def fixed_parameters(statespace: DSGEStateSpace, model_parameters: dict[str, float]) -> dict[str, float]:
    """Values of the statespace parameters that are neither estimated nor part of the error or shock covariance."""
    estimated = {*ESTIMATED_STRUCTURAL}
    return {name: model_parameters[name]
            for name in statespace.param_names
            if not name.startswith("error_") and name != "state_cov" and name not in estimated}


def build_pymc_model(statespace: DSGEStateSpace,
                     fit_data: pd.DataFrame,
                     model_parameters: dict[str, float]) -> pm.Model:
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
    error_group_index = np.array([error_groups.index("foreign" if name in FOREIGN_OBSERVABLES else "domestic")
                                  for name in observable_names])
    typical_error_logits = np.array([np.log(share / (1 - share)) for share in TYPICAL_ERROR_SHARES.values()])

    with pm.Model(coords=statespace.coords | {"error_group": error_groups}) as pymc_model:
        for name, (distribution, mean, sd) in STRUCTURAL_PRIORS.items():
            distribution(name, mu=mean, sigma=sd)
        for name, value in fixed_parameters(statespace, model_parameters).items():
            pm.Data(name, value)

        log_wage_slope = pm.Normal("log_wage_slope", **WAGE_SLOPE_PRIOR)
        slope_sum = 1 + discount_factor + pm.math.exp(log_wage_slope)
        pm.Deterministic("theta_w",
                         (slope_sum - pm.math.sqrt(slope_sum**2 - 4 * discount_factor)) / (2 * discount_factor))

        standardized_cholesky, _, standardized_sigmas = pm.LKJCholeskyCov(
            "shock_cov",
            n=len(structural_shocks),
            eta=SHOCK_CORRELATION_ETA,
            sd_dist=pm.InverseGamma.dist(alpha=SHOCK_SCALE_ALPHA, beta=SHOCK_SCALE_ALPHA + 1),
            compute_corr=True)
        shock_cholesky = shock_scale_modes[:, None] * standardized_cholesky
        shock_sigmas = shock_scale_modes * standardized_sigmas
        for index, shock in enumerate(structural_shocks):
            pm.Deterministic(f"sigma_{shock}", shock_sigmas[index])
        noise_sigmas = pt.stack([pm.HalfNormal(f"sigma_{shock}", sigma=noise_scales[shock])
                                 for shock in NOISE_SHOCK_OBSERVABLES])

        block_cov = pt.linalg.block_diag(pm.math.dot(shock_cholesky, shock_cholesky.T), pt.diag(noise_sigmas**2))
        pm.Deterministic("state_cov",
                         block_cov[shock_order][:, shock_order],
                         dims=statespace.param_dims["state_cov"])

        error_share_logit_mean = pm.Normal("error_share_logit_mean",
                                           mu=typical_error_logits,
                                           sigma=ERROR_SHARE_LOGIT_SD,
                                           dims="error_group")
        error_share_spread = pm.HalfNormal("error_share_spread", **ERROR_SHARE_SPREAD_PRIOR)
        error_share_offset = pm.Normal("error_share_offset", dims="observed_state")
        error_share_logit = error_share_logit_mean[error_group_index] + error_share_spread * error_share_offset
        error_share = pm.Deterministic("error_share",
                                       pm.math.sigmoid(error_share_logit),
                                       dims="observed_state")
        for index, name in enumerate(observable_names):
            pm.Deterministic(f"error_sigma_{name}", pt.sqrt(error_share[index]) * observable_sd[index])

        statespace.build_statespace_graph(data=fit_data)

    return pymc_model


def statespace_trace(draws: xr.Dataset,
                     parameter_values: dict[str, float],
                     observed_data: xr.Dataset,
                     group: str) -> xr.DataTree:
    """`draws` under `group` with the fixed parameters broadcast over them, as the statespace samplers read it."""
    held_fixed = xr.Dataset({name: xr.full_like(draws["phi_H"], value)
                             for name, value in parameter_values.items()})
    return xr.DataTree.from_dict({group: xr.merge([draws, held_fixed]),
                                  "observed_data": observed_data,
                                  "constant_data": xr.Dataset()})


def simulation_trace(simulation_statespace: DSGEStateSpace,
                     estimated_shocks: list[str],
                     draws: xr.Dataset,
                     model_parameters: dict[str, float],
                     observed_data: xr.Dataset,
                     group: str) -> xr.DataTree:
    """`draws` in the simulation statespace: the estimated shock covariance plus an uncorrelated disaster."""
    simulation_shocks = list(simulation_statespace.shock_names)
    shock_embedding = np.array([[float(shock == estimated) for estimated in estimated_shocks]
                                for shock in simulation_shocks])
    disaster_variance = np.diag([SHOCK_SIZES[DISASTER_SHOCK] ** 2 * float(shock == DISASTER_SHOCK)
                                 for shock in simulation_shocks])
    estimated_cov = draws["state_cov"].transpose("chain", "draw", "shock", "shock_aux").to_numpy()
    simulation_cov = xr.DataArray(shock_embedding @ estimated_cov @ shock_embedding.T + disaster_variance,
                                  dims=["chain", "draw", "shock", "shock_aux"],
                                  coords={"chain": draws["chain"],
                                          "draw": draws["draw"],
                                          "shock": simulation_shocks,
                                          "shock_aux": simulation_shocks})

    return statespace_trace(draws=draws.drop_vars(["state_cov", "shock", "shock_aux"]).assign(state_cov=simulation_cov),
                            parameter_values=fixed_parameters(simulation_statespace, model_parameters),
                            observed_data=observed_data,
                            group=group)


def response_quantiles(irf: xr.DataArray,
                       variables: list[str],
                       policy_rate_ss: float,
                       solved: xr.DataArray) -> xr.DataArray:
    """Median and 94% interval of `irf` in display units over the solved draws."""
    display_irf = xr.concat([to_display_units(irf=irf.where(solved),
                                              variable=variable,
                                              policy_rate_ss=policy_rate_ss)
                             for variable in variables],
                            dim="variable")
    return display_irf.quantile(IRF_QUANTILES, dim=["chain", "draw"])


def plot_response_band(ax: plt.Axes, quantiles: xr.DataArray, reference: xr.DataArray | None) -> None:
    """The median and 94% interval of one response, with `reference` dashed on top when it is given."""
    ax.fill_between(quantiles["time"],
                    quantiles.sel(quantile=IRF_QUANTILES[0]),
                    quantiles.sel(quantile=IRF_QUANTILES[-1]),
                    color=PALETTE["primary"],
                    alpha=0.3,
                    linewidth=0)
    ax.plot(quantiles["time"],
            quantiles.sel(quantile=0.5),
            color=PALETTE["primary"],
            lw=1.5,
            label="median and 94% interval")
    if reference is not None:
        ax.plot(reference["time"],
                reference,
                color="k",
                ls="--",
                lw=1.2,
                label="India calibration")
    ax.axhline(0.0, color="0.6", lw=0.75)


def add_response_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles=handles,
               labels=labels,
               loc="upper center",
               bbox_to_anchor=(0.5, 0.0),
               ncol=len(labels),
               frameon=False)


def plot_response_grid(irfs: xr.DataArray,
                       reference_irf: xr.DataArray | None,
                       shock_labels: dict[str, str],
                       variables: list[str],
                       policy_rate_ss: float,
                       solved: xr.DataArray) -> plt.Figure:
    """A grid of shocks by variables, each panel the distribution of one response over the solved draws.

    Parameters
    ----------
    irfs : DataArray
        Responses in log deviations, with dims ``chain``, ``draw``, ``shock``, ``time`` and ``variable``.
    reference_irf : DataArray or None
        One response path per shock and variable, with dims ``shock``, ``time`` and ``variable``, drawn dashed on top
        of the band. None draws no reference.
    shock_labels : dict of str to str
        The shocks to plot, one row each, mapped to their row labels.
    variables : list of str
        The variables to plot, one column each.
    policy_rate_ss : float
        Steady-state policy rate, for converting its response to annualized percentage points.
    solved : DataArray
        True at the draws whose linear solution converged.

    Returns
    -------
    fig : Figure
        The grid.
    """
    fig, axes = plt.subplots(nrows=len(shock_labels),
                             ncols=len(variables),
                             figsize=(2.3 * len(variables), 2 * len(shock_labels)),
                             sharex=True,
                             squeeze=False,
                             layout="constrained")
    for row, (shock, shock_label) in enumerate(shock_labels.items()):
        quantiles = response_quantiles(irf=irfs.sel(shock=shock),
                                       variables=variables,
                                       policy_rate_ss=policy_rate_ss,
                                       solved=solved)
        for column, variable in enumerate(variables):
            ax = axes[row, column]
            reference = None
            if reference_irf is not None:
                reference = to_display_units(irf=reference_irf.sel(shock=shock),
                                             variable=variable,
                                             policy_rate_ss=policy_rate_ss)
            plot_response_band(ax=ax,
                               quantiles=quantiles.sel(variable=variable),
                               reference=reference)
            if row == 0:
                ax.set_title(VARIABLE_LABELS[variable], loc="left", fontsize=10)
            if column == 0:
                ax.set_ylabel(shock_label, rotation=0, ha="right", va="center")

    add_response_legend(fig=fig, ax=axes[0, 0])
    fig.supxlabel("quarters after the shock")
    return fig


def plot_shock_responses(irf: xr.DataArray,
                         reference_irf: xr.DataArray | None,
                         variables: list[str],
                         policy_rate_ss: float,
                         solved: xr.DataArray) -> plt.Figure:
    """One panel per variable, four to a row, each the distribution of the response to one shock over the solved draws.

    Parameters
    ----------
    irf : DataArray
        Responses in log deviations, with dims ``chain``, ``draw``, ``time`` and ``variable``.
    reference_irf : DataArray or None
        One response path per variable, with dims ``time`` and ``variable``, drawn dashed on top of the band. None
        draws no reference.
    variables : list of str
        The variables to plot.
    policy_rate_ss : float
        Steady-state policy rate, for converting its response to annualized percentage points.
    solved : DataArray
        True at the draws whose linear solution converged.

    Returns
    -------
    fig : Figure
        The panels.
    """
    n_rows = int(np.ceil(len(variables) / 4))
    fig, axes = plt.subplots(nrows=n_rows,
                             ncols=4,
                             figsize=(14, 3 * n_rows),
                             sharex=True,
                             squeeze=False,
                             layout="constrained")
    quantiles = response_quantiles(irf=irf,
                                   variables=variables,
                                   policy_rate_ss=policy_rate_ss,
                                   solved=solved)
    for ax, variable in zip(axes.flat, variables):
        reference = None
        if reference_irf is not None:
            reference = to_display_units(irf=reference_irf,
                                         variable=variable,
                                         policy_rate_ss=policy_rate_ss)
        plot_response_band(ax=ax,
                           quantiles=quantiles.sel(variable=variable),
                           reference=reference)
        ax.set_title(VARIABLE_LABELS[variable], loc="left")
    for ax in axes.flat[len(variables):]:
        ax.set_axis_off()

    add_response_legend(fig=fig, ax=axes[0, 0])
    fig.supxlabel("quarters after the shock")
    return fig


OBSERVABLE_SHORT_NAMES = {**{name: variable for name, (_, variable) in GROWTH_OBSERVABLES.items()},
                          "pi": "pi",
                          "real_exchange_rate_growth": "q",
                          "foreign_rate": "i*",
                          "foreign_inflation": "pi*",
                          "foreign_output_growth": "Y*",
                          "world_import_price_growth": "p_M*"}
OUTER_INTERVAL_WIDTH = 2.5
INNER_INTERVAL_WIDTH = 8.0


def plot_density_against_data(ax: plt.Axes, values, data_value: float) -> None:
    """The density of `values` with `data_value` dashed, or a vertical line where `values` do not vary."""
    values = np.asarray(values)
    median = np.median(values)
    if np.ptp(values) <= 1e-9 * max(1.0, abs(median)):
        ax.axvline(median,
                   color=PALETTE["primary"],
                   lw=2.0)
        half_width = max(abs(data_value - median), 0.01 * abs(median))
        center = (data_value + median) / 2
        ax.set_xlim(center - 1.5 * half_width, center + 1.5 * half_width)
    else:
        density = az.kde(xr.DataArray(values, dims=["sample"]), dim="sample")
        ax.fill_between(density.sel(plot_axis="x"),
                        density.sel(plot_axis="y"),
                        color=PALETTE["primary"],
                        alpha=0.3,
                        linewidth=0)
        ax.plot(density.sel(plot_axis="x"),
                density.sel(plot_axis="y"),
                color=PALETTE["primary"],
                lw=1.5)
    ax.axvline(data_value,
               color="k",
               ls="--",
               lw=1.5)
    ax.set(yticks=[])
    ax.grid(visible=False, axis="y")


def plot_moment_grid(predictive_moments: xr.Dataset, data_moments: xr.Dataset, label: str) -> plt.Figure:
    """Densities of each observable's sample sd on the diagonal, and intervals of each correlation below it.

    Parameters
    ----------
    predictive_moments, data_moments : Dataset
        Output of `sample_moments` for the simulated observables and for the data.
    label : str
        Name of the predictive distribution in the legend, such as "prior predictive".

    Returns
    -------
    fig : Figure
        The grid of panels.
    """
    predictive_std_percent = 100 * predictive_moments["std"]
    data_std_percent = 100 * data_moments["std"].to_series()
    correlation_quantiles = predictive_moments["acf"].sel(lag=0).quantile([0.03, 0.25, 0.5, 0.75, 0.97],
                                                                          dim="sample")
    data_correlation = data_moments["acf"].sel(lag=0).to_pandas()

    n_observables = len(MOMENT_STATES)
    fig, axes = plt.subplots(nrows=n_observables,
                             ncols=n_observables,
                             figsize=(17, 15),
                             layout="constrained")
    for row, row_name in enumerate(MOMENT_STATES):
        for column, column_name in enumerate(MOMENT_STATES):
            ax = axes[row, column]
            if column > row:
                ax.set_axis_off()
                continue
            if column == row:
                std = predictive_std_percent.sel(variable=row_name)
                plot_density_against_data(ax=ax,
                                          values=std,
                                          data_value=data_std_percent[row_name])
                ax.set_xlim(0, max(float(std.quantile(0.99)), 1.2 * data_std_percent[row_name]))
                ax.set_title(f"sd({OBSERVABLE_SHORT_NAMES[row_name]}), %", loc="left", fontsize=10)
            else:
                plot_correlation_interval(ax=ax,
                                          interval=correlation_quantiles.sel(variable=row_name,
                                                                             variable_aux=column_name),
                                          data_value=data_correlation.loc[row_name, column_name])
            if column == 0 and row > 0:
                ax.set_ylabel(OBSERVABLE_SHORT_NAMES[row_name], rotation=0, ha="right", va="center")
            if row == n_observables - 1:
                ax.set_xlabel(OBSERVABLE_SHORT_NAMES[column_name])

    legend_handles = [plt.Line2D([],
                                 [],
                                 color=PALETTE["primary"],
                                 marker="o",
                                 markersize=9,
                                 markeredgecolor="white",
                                 lw=INNER_INTERVAL_WIDTH,
                                 label=f"{label} median and 50%"),
                      plt.Line2D([], [], color=PALETTE["primary"], lw=OUTER_INTERVAL_WIDTH, label=f"{label} 94%"),
                      plt.Line2D([],
                                 [],
                                 color="k",
                                 marker="o",
                                 markerfacecolor="none",
                                 lw=0.0,
                                 label="data correlation"),
                      plt.Line2D([], [], color="k", ls="--", lw=1.5, label="data standard deviation")]
    fig.legend(handles=legend_handles,
               loc="upper center",
               bbox_to_anchor=(0.5, 0.0),
               ncol=len(legend_handles),
               frameon=False)
    return fig


def plot_correlation_interval(ax: plt.Axes, interval: xr.DataArray, data_value: float) -> None:
    ax.hlines(y=0,
              xmin=float(interval.sel(quantile=0.03)),
              xmax=float(interval.sel(quantile=0.97)),
              color=PALETTE["primary"],
              lw=OUTER_INTERVAL_WIDTH)
    ax.hlines(y=0,
              xmin=float(interval.sel(quantile=0.25)),
              xmax=float(interval.sel(quantile=0.75)),
              color=PALETTE["primary"],
              lw=INNER_INTERVAL_WIDTH)
    ax.scatter(x=float(interval.sel(quantile=0.5)),
               y=0,
               color=PALETTE["primary"],
               edgecolors="white",
               s=90,
               linewidths=1.0,
               zorder=3)
    ax.scatter(x=data_value,
               y=0,
               facecolors="none",
               edgecolors="k",
               s=45,
               linewidths=1.2,
               zorder=4)
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set(xlim=(-1, 1),
           ylim=(-1, 1),
           yticks=[])
    ax.grid(visible=False, axis="y")


def bind_to_model(graphs: list, model: pm.Model) -> list:
    """Replace the statespace parameter placeholders in `graphs` with the model variables of the same name."""
    placeholders = [variable for variable in explicit_graph_inputs(graphs) if variable.name in model.named_vars]
    return graph_replace(graphs,
                         replace={variable: model[variable.name] for variable in placeholders},
                         strict=False)


def build_welfare_model(pymc_model: pm.Model, simulation_statespace: DSGEStateSpace) -> pm.Model:
    """`pymc_model` with the disaster's first-order welfare cost and compensating transfer as deterministics."""
    simulation_states = list(simulation_statespace.state_names)
    simulation_shocks = list(simulation_statespace.shock_names)
    simulation_steady_state = {variable.name: value
                               for variable, value in simulation_statespace.steady_state_mapping.items()}

    with pymc_model.copy() as welfare_model:
        transition, selection, consumption_ss, hours_ss, wage_dispersion_ss = bind_to_model(
            graphs=[simulation_statespace.ssm["transition"],
                    simulation_statespace.ssm["selection"],
                    simulation_steady_state["C_ss"],
                    simulation_steady_state["L_ss"],
                    simulation_steady_state["nu_w_ss"]],
            model=welfare_model)
        discount = welfare_model["beta"]
        habit = welfare_model["phi_H"]
        risk_aversion = welfare_model["sigma"]
        inverse_frisch = welfare_model["chi"]

        disaster_impact = selection[:, simulation_shocks.index(DISASTER_SHOCK)] * SHOCK_SIZES[DISASTER_SHOCK]
        discounted_response = pt.linalg.solve(pt.eye(len(simulation_states)) - discount * transition,
                                              disaster_impact)
        consumption_response, hours_response, wage_dispersion_response = (
            discounted_response[simulation_states.index(name)] for name in ["C", "L", "nu_w"])

        habit_adjusted_consumption = consumption_ss * (1 - habit)
        labor_disutility = welfare_model["Theta"] * wage_dispersion_ss * hours_ss ** (1 + inverse_frisch)
        marginal_utility_of_consumption = habit_adjusted_consumption ** -risk_aversion * (1 - discount * habit)
        consumption_value = marginal_utility_of_consumption * consumption_ss * consumption_response
        hours_value = -labor_disutility * (hours_response + wage_dispersion_response / (1 + inverse_frisch))
        value_of_permanent_consumption = habit_adjusted_consumption ** (1 - risk_aversion) / (1 - discount)
        value_of_one_quarter_of_consumption = marginal_utility_of_consumption * consumption_ss

        for suffix, value in [("_consumption", consumption_value),
                              ("_hours", hours_value),
                              ("", consumption_value + hours_value)]:
            pm.Deterministic(f"welfare_cost{suffix}", value / value_of_permanent_consumption)
            pm.Deterministic(f"compensating_transfer{suffix}", value / value_of_one_quarter_of_consumption)

    return welfare_model


def median_and_hdi(path: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Posterior median and 94% HDI of `path` over the draws on which it is fully observed."""
    draws = path.stack(sample=("chain", "draw")).drop_vars(["sample", "chain", "draw"]).dropna("sample", how="any")
    return draws.median("sample"), az.hdi(draws, prob=0.94, dim=["sample"])


WELFARE_INTERVAL_PROBABILITY = 0.98


def plot_clipped_densities(samples: xr.Dataset,
                           colors: list[str],
                           xlabel: str,
                           title: str,
                           mean_labels: str = "curve") -> plt.Axes:
    """Overlay the density of each variable in `samples` over its HDI, with each posterior mean marked and labeled.

    Parameters
    ----------
    samples : Dataset
        One variable per density, with a ``sample`` dimension.
    colors : list of str
        One color per variable.
    xlabel, title : str
        Axis label and figure title.
    mean_labels : {"curve", "legend"}, optional
        Label each mean above the curves, or in the legend entries when the means sit too close together to label
        apart. Default "curve".

    Returns
    -------
    ax : Axes
        The axes holding the densities.
    """
    if mean_labels == "legend":
        samples = samples.rename({name: f"{name}, mean {float(samples[name].mean()):.3g}" for name in samples})

    intervals = az.hdi(samples, prob=WELFARE_INTERVAL_PROBABILITY, dim=["sample"])
    densities = az.kde(samples, dim="sample")
    grid = densities.sel(plot_axis="x")
    inside_interval = (grid >= intervals.sel(ci_bound="lower")) & (grid <= intervals.sel(ci_bound="upper"))
    clipped = xr.Dataset({name: density.where(inside_interval[name]).assign_attrs(kind="kde")
                          for name, density in densities.items()})

    plot = az.plot_dist(samples,
                        sample_dims=["sample"],
                        kind="kde",
                        ci_kind="hdi",
                        ci_prob=WELFARE_INTERVAL_PROBABILITY,
                        point_estimate="mean",
                        cols=[],
                        aes={"color": ["__variable__"]},
                        aes_by_visuals={"point_estimate": ["color"]},
                        color=colors,
                        stats={"dist": clipped},
                        visuals={"face": True,
                                 "credible_interval": False,
                                 "point_estimate_text": False,
                                 "title": False},
                        figure_kwargs={"figsize": (9, 4)})
    plot.add_legend("__variable__",
                    title=f"{WELFARE_INTERVAL_PROBABILITY:.0%} HDI",
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.0),
                    ncol=len(samples.data_vars),
                    frameon=False)

    ax = plt.gcf().axes[0]
    if mean_labels == "curve":
        density_x = clipped.to_array("component").sel(plot_axis="x").to_numpy()
        density_y = clipped.to_array("component").sel(plot_axis="y").to_numpy()
        label_half_width = 0.05 * (np.nanmax(density_x) - np.nanmin(density_x))
        label_pad = 0.03 * np.nanmax(density_y)
        for name, color in zip(samples, colors, strict=True):
            mean = float(samples[name].mean())
            near_mean = np.abs(density_x - mean) <= label_half_width
            label_height = np.nanmax(np.where(near_mean, density_y, np.nan)) + label_pad
            ax.vlines(x=mean,
                      ymin=0.0,
                      ymax=label_height,
                      color=color,
                      lw=0.8,
                      ls=":")
            ax.text(mean,
                    label_height,
                    f"mean {mean:.3g}",
                    ha="center",
                    va="bottom",
                    color=color,
                    fontsize=9)
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    return ax
