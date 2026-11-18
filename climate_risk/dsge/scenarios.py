"""
Bridge from simulated annual climate damages to CatDSGE capital-destruction shock paths.

The climate-risk models produce draws of annual disaster damage in thousands of nominal US dollars
(EM-DAT's ``'000 US$``). CatDSGE takes a path of ``epsilon_xi_K`` where ``xi_K = exp(epsilon_xi_K)``
multiplies the previous period's capital stock, so a quarter that destroys a fraction ``d`` of capital
is the shock ``log(1 - d)``. Three steps connect the two:

1. :func:`damage_fraction` divides damages by a capital stock in matching dollars.
2. :func:`quarterly_shock_path` places each year's destruction in a single quarter.
3. :func:`select_return_period_scenarios` picks the draws that represent given return periods.

:func:`shock_trajectories` then pads the result to the full shock vector in the form
``impulse_response_function(..., shock_trajectory=...)`` expects.

Notes
-----
EM-DAT damages are direct losses to assets, so they map onto capital destruction. They are not a
productivity or output loss. Pairing these scenarios with a climate-economy model that also lowers
output with temperature counts the same harm twice, so use one or the other for the same channel.
"""

from collections.abc import Sequence

import numpy as np
import xarray as xr

SHOCK_NAME = "epsilon_xi_K"
THOUSAND_USD = 1e3
QUARTERS_PER_YEAR = 4
# CatDSGE's xi_K must stay positive, and a total wipe-out has no log; cap well short of it.
MAX_DESTRUCTION = 0.95
# Q3 (zero-indexed 2): the monsoon season, when most of India's flood damage falls.
EVENT_QUARTER = 2
# Scenario set: the 1000-year tail is not a loss estimate and is deliberately left out.
SCENARIO_RETURN_YEARS = (50, 100, 250)


def damage_fraction(
    damage_draws: xr.DataArray,
    capital_usd: xr.DataArray | dict[int, float],
    *,
    damage_units: float = THOUSAND_USD,
    max_destruction: float = MAX_DESTRUCTION,
) -> xr.DataArray:
    """
    Convert damage draws into the fraction of the capital stock destroyed.

    Parameters
    ----------
    damage_draws : DataArray
        Dims ``sample`` and ``year``, as :func:`~climate_risk.models.damage_curves.simulate_country_damage`
        or ``simulate_future_damage`` return, in ``damage_units`` US dollars.
    capital_usd : DataArray or dict
        Capital stock in nominal US dollars, indexed by ``year`` (or mapping year to value). It must
        cover every year in ``damage_draws``. A PPP capital series, such as the Penn World Table's,
        is not in matching dollars; scale it by a market-dollar GDP times the capital-output ratio.
    damage_units : float
        US dollars per damage unit; EM-DAT reports thousands, hence the default of 1e3.
    max_destruction : float
        Upper clip on the fraction, so ``log(1 - d)`` stays finite.

    Returns
    -------
    DataArray
        Same dims as ``damage_draws``, values in ``[0, max_destruction]``.
    """
    if not 0 < max_destruction < 1:
        raise ValueError(f"max_destruction must lie in (0, 1), got {max_destruction}")
    if isinstance(capital_usd, dict):
        years = sorted(capital_usd)
        capital_usd = xr.DataArray([capital_usd[y] for y in years], dims="year", coords={"year": years})
    missing = set(damage_draws["year"].values.tolist()) - set(capital_usd["year"].values.tolist())
    if missing:
        raise ValueError(f"capital_usd has no value for years {sorted(missing)}")
    capital = capital_usd.sel(year=damage_draws["year"])
    if (capital <= 0).any():
        raise ValueError("capital_usd must be positive")
    return (damage_draws * damage_units / capital).clip(0.0, max_destruction)


def quarterly_shock_path(
    fraction: xr.DataArray,
    *,
    quarter: int = EVENT_QUARTER,
) -> np.ndarray:
    """
    Turn annual destroyed fractions into a quarterly ``epsilon_xi_K`` path.

    Parameters
    ----------
    fraction : DataArray
        Dims ``sample`` and ``year`` from :func:`damage_fraction`.
    quarter : int
        Quarter (0 to 3) that takes the whole year's destruction. A flood or storm is a single lump, so
        the event is never spread across the year. The default, 2, is Q3, the Indian monsoon season.

    Returns
    -------
    ndarray
        Shape ``(n_sample, n_year * 4)`` of ``log(1 - d)`` shocks, years in order.
    """
    if quarter not in range(QUARTERS_PER_YEAR):
        raise ValueError(f"quarter must be 0 to 3, got {quarter}")
    annual = np.log1p(-fraction.transpose("sample", "year").values)
    n_sample, n_year = annual.shape
    path = np.zeros((n_sample, n_year, QUARTERS_PER_YEAR))
    path[:, :, quarter] = annual
    return path.reshape(n_sample, n_year * QUARTERS_PER_YEAR)


def shock_trajectories(
    shock_path: np.ndarray,
    shock_names: Sequence[str],
    *,
    shock_name: str = SHOCK_NAME,
) -> np.ndarray:
    """
    Embed a path of one shock into the model's full shock vector.

    Parameters
    ----------
    shock_path : ndarray
        Shape ``(n_scenario, n_quarter)``, from :func:`quarterly_shock_path`.
    shock_names : sequence of str
        The model's shock names in the order its state space uses.
    shock_name : str
        Which of them the path drives.

    Returns
    -------
    ndarray
        Shape ``(n_scenario, n_quarter, n_shock)``; each ``[i]`` slice is a ``shock_trajectory``.
    """
    names = list(shock_names)
    if shock_name not in names:
        raise ValueError(f"{shock_name!r} is not one of the model's shocks: {names}")
    out = np.zeros((*shock_path.shape, len(names)))
    out[..., names.index(shock_name)] = shock_path
    return out


def select_return_period_scenarios(
    fraction: xr.DataArray,
    return_years: Sequence[float],
    *,
    window: int | None = None,
) -> dict[float, xr.DataArray]:
    """
    Pick, for each return period, the draws whose annual destruction sits at that quantile.

    A ``T``-year event is the ``1 - 1/T`` quantile of the annual distribution (the convention of
    :func:`~climate_risk.models.damage_curves.damage_return_years`). The draws are ranked by their
    destroyed fraction pooled over all years, and the ``window`` draws straddling that rank are kept,
    so that several posterior draws represent each return period rather than a single one.

    Parameters
    ----------
    fraction : DataArray
        Dims ``sample`` and ``year``.
    return_years : sequence of float
        Return periods, each above 1.
    window : int, optional
        Number of (sample, year) cells kept per return period; defaults to 1% of all cells, at least 1.

    Returns
    -------
    dict
        Maps each return period to a DataArray of dims ``scenario`` and ``year`` holding one
        year's destroyed fraction in the chosen year and zero in the others, so each scenario is a
        single event of that size.
    """
    if any(ry <= 1 for ry in return_years):
        raise ValueError("return periods must exceed 1 year")
    stacked = fraction.transpose("sample", "year")
    values = stacked.values
    flat = values.ravel()
    n_cells = flat.size
    window = max(1, n_cells // 100) if window is None else window
    order = np.argsort(flat)
    n_year = values.shape[1]
    out = {}
    for ry in return_years:
        centre = round((1 - 1 / ry) * (n_cells - 1))
        lo = int(np.clip(centre - window // 2, 0, n_cells - window))
        cells = order[lo : lo + window]
        scenarios = np.zeros((window, n_year))
        scenarios[np.arange(window), cells % n_year] = flat[cells]
        out[ry] = xr.DataArray(
            scenarios,
            dims=("scenario", "year"),
            coords={"scenario": np.arange(window), "year": stacked["year"].values},
        )
    return out
