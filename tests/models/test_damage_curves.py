import numpy as np
import polars as pl
import pytest
import xarray as xr

from climate_risk.models.damage_curves import damage_return_years, simulate_country_damage

# Two countries, no nulls, so both models' _prepare_panel keeps every row in this order:
# AAA/2000 (obs_id 0), AAA/2001 (obs_id 1), BBB/2000 (obs_id 2), BBB/2002 (obs_id 3).
PANEL = pl.DataFrame(
    {
        "ISO": ["AAA", "AAA", "BBB", "BBB"],
        "year": [2000, 2001, 2000, 2002],
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
