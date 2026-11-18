import numpy as np
import pytest
import xarray as xr

from climate_risk.dsge.scenarios import (
    damage_fraction,
    quarterly_shock_path,
    select_return_period_scenarios,
    shock_trajectories,
)

YEARS = [2030, 2031]


def _draws(values: list[list[float]]) -> xr.DataArray:
    return xr.DataArray(np.array(values), dims=("sample", "year"), coords={"year": YEARS})


def test_damage_fraction_converts_thousands_of_usd() -> None:
    draws = _draws([[1e6, 2e6], [0.0, 5e6]])  # thousands of USD, i.e. $1B, $2B, $0, $5B
    capital = {2030: 100e9, 2031: 100e9}
    fraction = damage_fraction(draws, capital)
    np.testing.assert_allclose(fraction.values, [[0.01, 0.02], [0.0, 0.05]])


def test_damage_fraction_clips_total_losses() -> None:
    fraction = damage_fraction(_draws([[1e12, 0.0]]), {2030: 1e9, 2031: 1e9}, max_destruction=0.9)
    assert fraction.values.max() == pytest.approx(0.9)


def test_damage_fraction_rejects_missing_capital_year() -> None:
    with pytest.raises(ValueError, match="2031"):
        damage_fraction(_draws([[1.0, 1.0]]), {2030: 1e9})


def test_damage_fraction_rejects_nonpositive_capital() -> None:
    with pytest.raises(ValueError, match="positive"):
        damage_fraction(_draws([[1.0, 1.0]]), {2030: 1e9, 2031: 0.0})


def test_default_path_is_one_lump_in_q3() -> None:
    path = quarterly_shock_path(_draws([[0.10, 0.0]]))
    assert path.shape == (1, 8)
    expected = np.zeros(8)
    expected[2] = np.log(0.9)
    np.testing.assert_allclose(path[0], expected)


def test_concentrated_path_puts_the_event_in_one_quarter() -> None:
    path = quarterly_shock_path(_draws([[0.10, 0.20]]), quarter=2)
    expected = np.zeros(8)
    expected[2], expected[6] = np.log(0.9), np.log(0.8)
    np.testing.assert_allclose(path[0], expected)


def test_quarter_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="quarter"):
        quarterly_shock_path(_draws([[0.1, 0.1]]), quarter=4)


def test_shock_trajectories_only_drives_the_named_shock() -> None:
    path = np.array([[-0.1, 0.0, -0.2]])
    names = ["epsilon_a", "epsilon_xi_K", "epsilon_b"]
    out = shock_trajectories(path, names)
    assert out.shape == (1, 3, 3)
    np.testing.assert_array_equal(out[0, :, 1], path[0])
    assert not out[..., [0, 2]].any()


def test_shock_trajectories_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="not one of"):
        shock_trajectories(np.zeros((1, 4)), ["epsilon_a"])


def test_return_period_scenarios_rank_by_size_and_are_single_events() -> None:
    rng = np.random.default_rng(0)
    fraction = xr.DataArray(rng.uniform(0, 0.1, size=(500, 2)), dims=("sample", "year"), coords={"year": YEARS})
    scenarios = select_return_period_scenarios(fraction, [2, 100], window=5)
    for ry, scenario in scenarios.items():
        assert scenario.sizes["scenario"] == 5, ry
        assert ((scenario > 0).sum("year") == 1).all(), ry
    assert scenarios[100].max() > scenarios[2].max()
    expected = np.quantile(fraction.values.ravel(), 0.99)
    assert scenarios[100].max().item() == pytest.approx(expected, abs=0.005)


def test_return_period_rejects_one_year() -> None:
    with pytest.raises(ValueError, match="exceed 1"):
        select_return_period_scenarios(_draws([[0.1, 0.1]]), [1])
