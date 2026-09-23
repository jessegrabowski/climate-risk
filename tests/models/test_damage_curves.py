import numpy as np
import polars as pl
import pytest
import xarray as xr

from climate_risk.models import damage, frequency
from climate_risk.models.damage_curves import (
    _future_mu,
    damage_return_years,
    simulate_country_damage,
    simulate_future_damage,
)

# Two countries, no nulls, so both models' _prepare_panel keeps every row in this order:
# AAA/2000 (obs_id 0), AAA/2001 (obs_id 1), BBB/2000 (obs_id 2), BBB/2002 (obs_id 3).
PANEL = pl.DataFrame(
    {
        "ISO": ["AAA", "AAA", "BBB", "BBB"],
        "date": [2000, 2001, 2000, 2002],
        "ln_population_density": [1.0, 1.1, 2.0, 2.2],
        "ln_gdp_pc": [3.0, 3.1, 4.0, 4.2],
        "square_ln_gdp_pc": [9.0, 9.61, 16.0, 17.64],
        "precip_deviation": [0.1, -0.2, 0.3, -0.4],
        "co2": [300.0, 301.0, 302.0, 303.0],
        "population": [1.0, 1.2, 2.0, 2.4],
        "hydrological_disasters": [0, 0, 0, 0],
        "Total_Damage_Adjusted_hydro": [0.0, 0.0, 0.0, 0.0],
    }
)


def _idata(var_name: str, draws: list[list[float]]) -> xr.DataTree:
    values = np.array([draws])  # one chain
    ds = xr.Dataset({var_name: (["chain", "draw", "obs_id"], values)})
    return xr.DataTree.from_dict({"posterior_predictive": ds})


FREQUENCY_IDATA = _idata("hydrological_disasters", [[2, 3, 5, 7], [4, 6, 8, 10]])
DAMAGE_IDATA = _idata("Total_Damage_Adjusted_hydro", [[10, 20, 30, 40], [1, 2, 3, 4]])


def test_simulate_country_damage_multiplies_paired_draws_for_one_country():
    draws = simulate_country_damage(PANEL, FREQUENCY_IDATA, DAMAGE_IDATA, "AAA")

    assert sorted(draws.sel(year=2000).values.tolist()) == [4, 20]
    assert sorted(draws.sel(year=2001).values.tolist()) == [12, 60]


def test_simulate_country_damage_only_keeps_the_requested_country():
    draws = simulate_country_damage(PANEL, FREQUENCY_IDATA, DAMAGE_IDATA, "BBB")

    assert sorted(draws["year"].values.tolist()) == [2000, 2002]
    assert sorted(draws.sel(year=2000).values.tolist()) == [24, 150]
    assert sorted(draws.sel(year=2002).values.tolist()) == [40, 280]


def test_simulate_country_damage_raises_for_an_unknown_country():
    with pytest.raises(ValueError, match="ZZZ"):
        simulate_country_damage(PANEL, FREQUENCY_IDATA, DAMAGE_IDATA, "ZZZ")


def test_simulate_country_damage_raises_when_chain_and_draw_counts_differ():
    mismatched = _idata("Total_Damage_Adjusted_hydro", [[10, 20, 30, 40]])

    with pytest.raises(ValueError, match="chain and draw counts"):
        simulate_country_damage(PANEL, FREQUENCY_IDATA, mismatched, "AAA")


def test_damage_return_years_reads_the_quantile_off_the_draws():
    draws = simulate_country_damage(PANEL, FREQUENCY_IDATA, DAMAGE_IDATA, "AAA")

    curve = damage_return_years(draws, return_years=[2])

    assert curve.filter(pl.col("year") == 2000)["damage"].item() == 12.0
    assert curve.filter(pl.col("year") == 2001)["damage"].item() == 36.0


def test_damage_return_years_has_one_row_per_year_and_return_year():
    draws = simulate_country_damage(PANEL, FREQUENCY_IDATA, DAMAGE_IDATA, "AAA")

    curve = damage_return_years(draws, return_years=[2, 10])

    assert curve.height == 4
    assert set(curve["return_year"].unique()) == {2, 10}
    assert set(curve["year"].unique()) == {2000, 2001}


def test_future_mu_combines_country_effect_and_standardized_features():
    # StandardScaler fit on f1=[-1,1] gives mean 0, std 1; on f2=[-2,2] gives mean 0, std 2. So the
    # future row (f1=1.0, f2=-2.0) standardizes to (1.0, -1.0).
    historical_panel = pl.DataFrame(
        {"ISO": ["AAA", "AAA"], "date": [2000, 2001], "f1": [-1.0, 1.0], "f2": [-2.0, 2.0], "target": [0.0, 0.0]}
    )
    future_covariates = pl.DataFrame({"ISO": ["AAA"], "year": [2030], "f1": [1.0], "f2": [-2.0]})
    posterior = xr.Dataset(
        {
            "country_effect": (["chain", "draw", "ISO"], [[[0.5]]]),
            "beta": (["chain", "draw", "feature"], [[[0.1, 0.2]]]),
        },
        coords={"ISO": ["AAA"], "feature": ["f1", "f2"]},
    )
    idata = xr.DataTree.from_dict({"posterior": posterior})

    mu = _future_mu(
        future_covariates=future_covariates,
        historical_panel=historical_panel,
        features=["f1", "f2"],
        target="target",
        idata=idata,
        iso="AAA",
    )

    expected_latent_rate = 0.5 + (1.0 * 0.1 + -1.0 * 0.2)
    assert mu.shape == (1, 1, 1)
    assert mu[0, 0, 0] == pytest.approx(np.logaddexp(0.0, expected_latent_rate))


def _future_country_idata(*, beta_len, country_effect, psi, **extra):
    """A posterior with one chain, one draw, and one country ("AAA"), over frequency.FEATURES."""
    data_vars = {
        "country_effect": (["chain", "draw", "ISO"], [[[country_effect]]]),
        "beta": (["chain", "draw", "feature"], [[[0.0] * beta_len]]),
        "psi": (["chain", "draw"], [[psi]]),
    }
    for name, value in extra.items():
        data_vars[name] = (["chain", "draw"], [[value]])

    posterior = xr.Dataset(data_vars, coords={"ISO": ["AAA"], "feature": frequency.FEATURES[:beta_len]})
    return xr.DataTree.from_dict({"posterior": posterior})


# frequency.FEATURES is a superset of damage.FEATURES, so one panel and one future row, each
# carrying every frequency feature, covers what both models read. `population` is kept positive
# (unlike the other features' placeholder -1.0/1.0) since damage.FEATURES derives `ln_population`
# from it.
_HISTORICAL_PANEL = pl.DataFrame(
    {
        "ISO": ["AAA", "AAA"],
        "date": [2000, 2001],
        frequency.TARGET: [0.0, 0.0],
        damage.TARGET: [0.0, 0.0],
        **{feature: [-1.0, 1.0] for feature in frequency.FEATURES if feature != "population"},
        "population": [1.0, 2.0],
    }
)
_FUTURE_COVARIATES = pl.DataFrame(
    {
        "ISO": ["AAA"],
        "year": [2030],
        **{feature: [1.0] for feature in frequency.FEATURES if feature != "population"},
        "population": [1.5],
    }
)


def test_simulate_future_damage_is_zero_when_frequency_is_fully_zero_inflated():
    frequency_idata = _future_country_idata(beta_len=len(frequency.FEATURES), country_effect=0.0, psi=0.0, alpha=1.0)
    damage_idata = _future_country_idata(beta_len=len(damage.FEATURES), country_effect=0.0, psi=1.0, sigma=1.0)

    draws = simulate_future_damage(
        future_covariates=_FUTURE_COVARIATES,
        historical_panel=_HISTORICAL_PANEL,
        frequency_idata=frequency_idata,
        damage_idata=damage_idata,
        iso="AAA",
    )

    assert (draws.values == 0.0).all()


def test_simulate_future_damage_has_sample_and_year_dims():
    frequency_idata = _future_country_idata(beta_len=len(frequency.FEATURES), country_effect=1.0, psi=0.9, alpha=2.0)
    damage_idata = _future_country_idata(beta_len=len(damage.FEATURES), country_effect=1.0, psi=0.5, sigma=1.0)

    draws = simulate_future_damage(
        future_covariates=_FUTURE_COVARIATES,
        historical_panel=_HISTORICAL_PANEL,
        frequency_idata=frequency_idata,
        damage_idata=damage_idata,
        iso="AAA",
    )

    assert set(draws.dims) == {"sample", "year"}
    assert draws.sizes["year"] == 1
