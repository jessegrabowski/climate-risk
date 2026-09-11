import numpy as np
import pandas as pd
import pymc as pm
import pytest

from climate_risk.models.climate_economy import OBSERVED, SHOCKS, STATES, ClimateEconomy


@pytest.fixture
def climate_economy():
    return ClimateEconomy(population_growth_mean=0.01)


def test_the_state_space_dimensions_match_the_declared_blocks(climate_economy):
    assert climate_economy.k_states == len(STATES)
    assert climate_economy.k_endog == len(OBSERVED)
    assert climate_economy.k_posdef == len(SHOCKS)


def test_set_states_declares_every_latent_state_and_observed_series(climate_economy):
    states = climate_economy.set_states()

    assert [state.name for state in states] == STATES + OBSERVED
    # `STATES` and `OBSERVED` share three names (log_pop, log_emp_rate, log_tfp_frontier), each
    # appearing once as a latent state and once as its own directly-observed series, so the flags
    # only line up positionally, not through a name-keyed lookup.
    assert not any(state.observed for state in states[: len(STATES)])
    assert all(state.observed for state in states[len(STATES) :])


def test_set_shocks_declares_one_innovation_per_shock(climate_economy):
    shocks = climate_economy.set_shocks()

    assert [shock.name for shock in shocks] == [f"{name}_innovation" for name in SHOCKS]


def test_set_parameters_covers_every_free_variable_the_graph_registers(climate_economy):
    parameter_names = {parameter.name for parameter in climate_economy.set_parameters()}

    expected = {
        "initial_level",
        "initial_growth",
        "initial_warming",
        "initial_co2",
        "initial_gap",
        "drift_co2",
        "drift_frontier",
        "drift_gap",
        "ky_mean",
        "ky_slope",
        "gdp_intercept",
        "climate_response",
        "theta",
        "phi",
        "rho",
        "damage_elasticity",
        "temp_loading",
        "precip_loading",
        "obs_intercept",
        "sigma_state",
        "sigma_obs",
    }
    assert parameter_names == expected


def test_thermal_persistence_follows_from_the_adjustment_time():
    fast = ClimateEconomy(population_growth_mean=0.01, thermal_adjustment_years=1.0)
    slow = ClimateEconomy(population_growth_mean=0.01, thermal_adjustment_years=50.0)

    assert fast.thermal_persistence == pytest.approx(np.exp(-1.0))
    assert slow.thermal_persistence == pytest.approx(np.exp(-1.0 / 50.0))
    assert fast.thermal_persistence < slow.thermal_persistence


@pytest.mark.slow
def test_the_symbolic_graph_builds_and_draws_a_prior_predictive(climate_economy):
    years = pd.date_range("2000-01-01", periods=6, freq="YS")
    observed = pd.DataFrame(
        0.01 * np.arange(len(years) * len(OBSERVED)).reshape(len(years), -1), index=years, columns=OBSERVED
    )

    with pm.Model(coords=climate_economy.coords):
        pm.Normal("initial_level", mu=0.0, sigma=1.0, dims=["level_block"])
        pm.Normal("initial_growth", mu=0.0, sigma=1.0, dims=["growth_block"])
        pm.Normal("initial_warming", mu=0.0, sigma=1.0)
        pm.Normal("initial_co2", mu=0.0, sigma=1.0)
        pm.Normal("initial_gap", mu=0.0, sigma=1.0)
        pm.Normal("drift_co2", mu=0.0, sigma=1.0)
        pm.Normal("drift_frontier", mu=0.0, sigma=1.0)
        pm.Normal("drift_gap", mu=0.0, sigma=1.0)
        pm.Normal("ky_mean", mu=0.0, sigma=1.0)
        pm.HalfNormal("ky_slope", sigma=1.0)
        pm.Normal("gdp_intercept", mu=0.0, sigma=1.0)
        pm.HalfNormal("climate_response", sigma=1.0)
        pm.Beta("theta", alpha=2.0, beta=2.0)
        pm.Beta("phi", alpha=2.0, beta=2.0, dims=["growth_block"])
        pm.Beta("rho", alpha=2.0, beta=2.0, dims=["ar_block"])
        pm.Normal("damage_elasticity", mu=0.0, sigma=0.1)
        pm.HalfNormal("temp_loading", sigma=1.0)
        pm.Normal("precip_loading", mu=0.0, sigma=0.1)
        pm.Normal("obs_intercept", mu=0.0, sigma=1.0, dims=["intercept"])
        pm.HalfNormal("sigma_state", sigma=1.0, dims=["shock"])
        pm.HalfNormal("sigma_obs", sigma=1.0, dims=["observed_state"])

        climate_economy.build_statespace_graph(observed)
        prior = pm.sample_prior_predictive(draws=2, random_seed=0)

    assert prior.prior["theta"].shape == (1, 2)
