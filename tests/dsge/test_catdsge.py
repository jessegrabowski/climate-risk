from types import SimpleNamespace

import gEconpy as ge
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr

from climate_risk.dsge.catdsge import (
    CALIBRATED_PARAMETERS,
    DISASTER_SHOCK,
    GROWTH_OBSERVABLES,
    OPEN_ECONOMY_OBSERVATION_EQUATIONS,
    QUARTERS_PER_YEAR,
    SHOCK_SIZES,
    build_observed,
    build_pymc_model,
    build_welfare_model,
    calibrate,
    compute_data_targets,
    error_scale,
    load_estimation_statespace,
    response_quantiles,
    simulation_trace,
    to_display_units,
    wage_calvo_probability,
)
from climate_risk.dsge.model_files import GCN_DIR

QUARTERS = pd.period_range("2010Q1", "2025Q4", freq="Q")


def india_panel(rng):
    """Every column the CatDSGE functions read, quarterly, with annual columns on their closing quarters only."""
    growing = {
        column: np.exp(np.cumsum(rng.normal(0.01, 0.01, size=len(QUARTERS))))
        for column in [
            "gdp",
            "consumption",
            "investment",
            "government",
            "exports",
            "imports",
            "cpi",
            "real_effective_exchange_rate",
            "world_price_level",
            "world_output",
            "commodity_all",
        ]
    }
    closing = pd.Series(QUARTERS.quarter == 1, index=QUARTERS)
    annual = {
        column: pd.Series(value, index=QUARTERS).where(closing)
        for column, value in {
            "government_debt": 0.7,
            "tax_revenue_gdp": 0.1,
            "nominal_consumption_share": 0.6,
            "nominal_investment_share": 0.3,
            "labour_share_pwt": 0.5,
            "labour_share_ilo": 0.6,
        }.items()
    }

    return pd.DataFrame({**growing, "world_rate_3m": rng.uniform(0, 5, size=len(QUARTERS)), **annual}, index=QUARTERS)


def test_the_debt_target_averages_the_debt_the_estimation_observes():
    """Debt is observed in log levels and not demeaned, so a target over other fiscal years than the observations
    would calibrate a steady state the data do not sit around."""
    panel = india_panel(np.random.default_rng(0))
    panel["government_debt"] = np.nan
    panel.loc[pd.Period("2011Q1"), "government_debt"] = 9.0
    panel.loc[pd.Period("2013Q1"), "government_debt"] = 0.5
    panel.loc[pd.Period("2024Q1"), "government_debt"] = 0.7
    panel.loc[pd.Period("2025Q1"), "government_debt"] = 9.0

    targets = compute_data_targets(panel)
    observed_debt = np.exp(build_observed(panel)["debt_ratio"].dropna())

    assert targets["government debt / annual GDP"] == pytest.approx(0.6)
    assert targets["government debt / annual GDP"] == pytest.approx(observed_debt.mean())


def test_lockdown_quarters_and_the_2012_export_jump_are_missing():
    observed = build_observed(india_panel(np.random.default_rng(1)))

    assert observed.loc["2020Q1":"2021Q4"].isna().all().all()
    assert observed.loc["2012Q3":"2012Q4", "export_growth"].isna().all()
    assert observed.loc["2012Q3":"2012Q4", "gdp_growth"].notna().all()


def test_growth_observables_are_demeaned_and_the_debt_level_is_not():
    """Debt enters as a log level the model matches in steady state, so demeaning it would discard the target."""
    observed = build_observed(india_panel(np.random.default_rng(2)))

    assert observed[[*GROWTH_OBSERVABLES, *OPEN_ECONOMY_OBSERVATION_EQUATIONS]].mean().abs().max() < 1e-12
    assert observed["debt_ratio"].dropna().iloc[0] == pytest.approx(np.log(0.7))


def test_a_rupee_appreciation_lowers_the_observed_real_exchange_rate():
    """A rise in the BIS index is a real appreciation, the opposite direction to the model's q."""
    panel = india_panel(np.random.default_rng(3))
    panel["real_effective_exchange_rate"] = 100.0
    panel.loc["2015Q1":, "real_effective_exchange_rate"] = 110.0

    observed = build_observed(panel)

    assert observed["real_exchange_rate_growth"].idxmin() == pd.Period("2015Q1")


def test_an_observable_in_levels_is_scaled_by_its_changes():
    fit_data = pd.DataFrame({"foreign_rate": [1.0, 1.1, 1.2, 1.3], "gdp_growth": [1.0, 1.1, 1.2, 1.3]})

    assert error_scale(fit_data, "foreign_rate") == pytest.approx(0.0, abs=1e-12)
    assert error_scale(fit_data, "gdp_growth") == pytest.approx(fit_data["gdp_growth"].std())


def test_inflation_and_the_policy_rate_are_annualized_percentage_points():
    irf = xr.DataArray([0.01, 0.01, 0.01], dims=["variable"], coords={"variable": ["pi", "i", "GDP"]})

    assert float(to_display_units(irf, "pi", policy_rate_ss=1.02)) == pytest.approx(100 * QUARTERS_PER_YEAR * 0.01)
    assert float(to_display_units(irf, "i", policy_rate_ss=1.02)) == pytest.approx(
        100 * QUARTERS_PER_YEAR * 1.02 * np.expm1(0.01)
    )
    assert float(to_display_units(irf, "GDP", policy_rate_ss=1.02)) == pytest.approx(1.0)


def test_unsolved_draws_are_left_out_of_the_response_quantiles():
    """A draw whose solution failed carries garbage responses, which would otherwise widen every band."""
    irf = xr.DataArray(
        np.array([[[0.01]], [[0.01]], [[50.0]]]).reshape(1, 3, 1, 1),
        dims=["chain", "draw", "time", "variable"],
        coords={"variable": ["GDP"]},
    )
    solved = xr.DataArray([[True, True, False]], dims=["chain", "draw"])

    quantiles = response_quantiles(irf=irf, variables=["GDP"], policy_rate_ss=1.0, solved=solved)

    assert float(quantiles.max()) == pytest.approx(1.0)


def test_the_disaster_enters_the_simulation_covariance_uncorrelated():
    """The estimated covariance is embedded on its own shocks, and the disaster gets its own variance and nothing else."""
    estimated = ["epsilon_A", "epsilon_G"]
    estimated_cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    draws = xr.Dataset(
        {
            "state_cov": (("chain", "draw", "shock", "shock_aux"), estimated_cov[None, None]),
            "phi_H": (("chain", "draw"), [[0.3]]),
        },
        coords={"chain": [0], "draw": [0], "shock": estimated, "shock_aux": estimated},
    )
    statespace = SimpleNamespace(shock_names=["epsilon_A", DISASTER_SHOCK, "epsilon_G"], param_names=[])

    trace = simulation_trace(
        simulation_statespace=statespace,
        estimated_shocks=estimated,
        draws=draws,
        model_parameters={},
        observed_data=xr.Dataset(),
        group="posterior",
    )

    simulation_cov = trace["posterior"]["state_cov"].isel(chain=0, draw=0)
    disaster_row = simulation_cov.sel(shock=DISASTER_SHOCK)
    assert float(disaster_row.sel(shock_aux=DISASTER_SHOCK)) == pytest.approx(SHOCK_SIZES[DISASTER_SHOCK] ** 2)
    assert disaster_row.sel(shock_aux=["epsilon_A", "epsilon_G"]).to_numpy().tolist() == [0.0, 0.0]
    assert float(simulation_cov.sel(shock="epsilon_A", shock_aux="epsilon_G")) == 0.5


def test_the_wage_calvo_probability_inverts_the_wage_phillips_curve_slope():
    """The slope is quadratic in theta, and only the smaller root lies inside the unit interval."""
    discount, theta = 0.995, 0.75
    slope = (1 - theta) * (1 - discount * theta) / theta

    assert wage_calvo_probability(slope=slope, discount=discount) == pytest.approx(theta)


def test_a_calibration_whose_steady_state_never_solves_raises():
    """A failed steady state returns finite garbage, which a least-squares fit would otherwise accept as a match."""
    model = SimpleNamespace(
        parameters=lambda: dict.fromkeys(CALIBRATED_PARAMETERS, 0.3),
        steady_state=lambda **_: SimpleNamespace(success=False),
    )
    targets = pd.Series(
        {
            "government / GDP": 0.1,
            "imports / GDP": 0.2,
            "exports / GDP": 0.2,
            "investment / GDP": 0.3,
            "labor share": 0.5,
            "government debt / annual GDP": 0.7,
        }
    )

    with pytest.raises(RuntimeError, match="steady state does not solve"):
        calibrate(model, quarterly_base={}, data_targets=targets)


def test_a_consumption_drop_under_log_utility_costs_its_discounted_size_in_quarters_of_consumption():
    """With log utility, no habit and no hours response, the disaster's cost in quarters of steady-state consumption is
    the discounted sum of the consumption path: the impact divided by 1 - beta * rho for persistence rho."""
    discount, persistence = 0.99, 0.5
    with pm.Model() as pymc_model:
        for name, value in {"beta": discount, "phi_H": 0.0, "sigma": 1.0, "chi": 1.0, "Theta": 1.0}.items():
            pm.Data(name, value)
    statespace = SimpleNamespace(
        state_names=["C", "L", "nu_w"],
        shock_names=[DISASTER_SHOCK],
        steady_state_mapping={
            pt.scalar(name): pt.constant(value) for name, value in {"C_ss": 2.0, "L_ss": 0.3, "nu_w_ss": 1.0}.items()
        },
        ssm={
            "transition": pt.constant(np.diag([persistence, 0.0, 0.0])),
            "selection": pt.constant(np.array([[1.0], [0.0], [0.0]])),
        },
    )

    welfare_model = build_welfare_model(pymc_model, simulation_statespace=statespace)

    expected = SHOCK_SIZES[DISASTER_SHOCK] / (1 - discount * persistence)
    assert welfare_model["compensating_transfer"].eval() == pytest.approx(expected)
    assert welfare_model["welfare_cost"].eval() == pytest.approx(expected * (1 - discount))


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:Provided data contains missing values")
def test_the_estimation_model_builds_on_the_india_observables():
    """Builds the model without sampling, so a change in the gEconpy or pymc-extras API fails here first."""
    model = ge.model_from_gcn(GCN_DIR / "catdsge_v4.gcn", verbose=False)
    observed = build_observed(india_panel(np.random.default_rng(4)))
    statespace = load_estimation_statespace()

    pymc_model = build_pymc_model(
        statespace=statespace,
        fit_data=observed.set_axis(pd.PeriodIndex(observed.index).to_timestamp()),
        model_parameters=model.parameters(),
    )

    assert {"theta_w", "state_cov"} <= set(pymc_model.named_vars)
