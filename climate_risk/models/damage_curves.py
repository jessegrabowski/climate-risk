from collections.abc import Sequence

import numpy as np
import polars as pl
import xarray as xr

from climate_risk.models.damage import TARGET as DAMAGE_TARGET
from climate_risk.models.damage import _prepare_panel as _prepare_damage_panel
from climate_risk.models.frequency import TARGET as FREQUENCY_TARGET
from climate_risk.models.frequency import _prepare_panel as _prepare_frequency_panel

# The two return years the paper's own worked example names (Sec 4.6: 100-RY and 500-RY), extended
# to a standard actuarial return-period grid.
RETURN_YEARS = [2, 5, 10, 25, 50, 100, 250, 500, 1000]


def simulate_country_damage(
    panel: pl.DataFrame,
    frequency_idata: xr.DataTree,
    damage_idata: xr.DataTree,
    iso: str,
) -> xr.DataArray:
    """
    Draw a country's joint annual-damage distribution (paper Sec 4.6: country-level damage curves).

    A country-year's total damage is the product of how many disasters occur and what each one
    costs, so a draw of it is a frequency draw times a damage draw. Pairing the two Stage 2 models'
    posterior predictive draws position-by-position, rather than drawing one against many, treats the
    two posteriors as independent, which is exactly true here: the models share no parameters, and
    it is also the paper's own assumption for this derivation ("D and P are conditionally
    independent").

    Parameters
    ----------
    panel : DataFrame
        A polars frame in the shape :func:`climate_risk.replication_data.create_replication_data`
        returns, the same one both models were fit on.
    frequency_idata : DataTree
        Posterior predictive from :func:`~climate_risk.models.frequency.build_hierarchical_frequency_model`,
        sampled with :func:`~climate_risk.sample.sample_or_load`.
    damage_idata : DataTree
        Posterior predictive from :func:`~climate_risk.models.damage.build_damage_model`, sampled
        with the same number of chains and draws as ``frequency_idata``.
    iso : str
        ISO alpha-3 code of the country to simulate.

    Returns
    -------
    damage_draws : DataArray
        Dims ``sample`` (chain and draw flattened together) and ``year``, carrying one simulated
        total-damage draw per sample, for every year both models kept a row for.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.models.damage import build_damage_model
        from climate_risk.models.damage_curves import simulate_country_damage
        from climate_risk.models.frequency import build_hierarchical_frequency_model
        from climate_risk.replication_data import create_replication_data
        from climate_risk.sample import sample_or_load

        panel = create_replication_data(Path("data"))
        frequency_idata = sample_or_load(
            "data/idata/frequency_stage2.nc", model=build_hierarchical_frequency_model(panel)
        )
        damage_idata = sample_or_load("data/idata/damage_stage2.nc", model=build_damage_model(panel))

        draws = simulate_country_damage(panel, frequency_idata, damage_idata, "NPL")
    """
    freq_rows = _prepare_frequency_panel(panel)
    damage_rows = _prepare_damage_panel(panel)

    freq_years = freq_rows.loc[freq_rows["ISO"] == iso, "year"]
    damage_years = damage_rows.loc[damage_rows["ISO"] == iso, "year"]

    common_years = sorted(set(freq_years) & set(damage_years))
    if not common_years:
        raise ValueError(f"{iso} has no year where both the frequency and damage panels kept a row.")

    freq_obs_id = [int(freq_years[freq_years == year].index[0]) for year in common_years]
    damage_obs_id = [int(damage_years[damage_years == year].index[0]) for year in common_years]

    freq_draws = frequency_idata.posterior_predictive[FREQUENCY_TARGET].isel(obs_id=freq_obs_id)
    damage_draws = damage_idata.posterior_predictive[DAMAGE_TARGET].isel(obs_id=damage_obs_id)

    if (
        freq_draws.sizes["chain"] != damage_draws.sizes["chain"]
        or freq_draws.sizes["draw"] != damage_draws.sizes["draw"]
    ):
        raise ValueError(
            "frequency_idata and damage_idata must share the same chain and draw counts to pair their "
            f"draws; got {freq_draws.sizes['chain']}x{freq_draws.sizes['draw']} and "
            f"{damage_draws.sizes['chain']}x{damage_draws.sizes['draw']}."
        )

    freq_draws = freq_draws.assign_coords(obs_id=common_years).rename(obs_id="year")
    damage_draws = damage_draws.assign_coords(obs_id=common_years).rename(obs_id="year")

    combined = (freq_draws * damage_draws).stack(sample=("chain", "draw"))
    result: xr.DataArray = combined.drop_vars(["chain", "draw", "sample"])

    return result


def damage_return_years(damage_draws: xr.DataArray, return_years: Sequence[float] = RETURN_YEARS) -> pl.DataFrame:
    """
    Read return-year damage values off a country's simulated annual-damage draws (paper Sec 4.6).

    A return year of :math:`RY` is the damage level equalled or exceeded, on average, once every
    :math:`RY` years: the :math:`1 - 1/RY` quantile of the draws for that year.

    Parameters
    ----------
    damage_draws : DataArray
        Dims ``sample`` and ``year``, as returned by :func:`simulate_country_damage`.
    return_years : sequence of float, optional
        Return periods to report, in years. Default :data:`RETURN_YEARS`.

    Returns
    -------
    curve : DataFrame
        One row per ``year`` and ``return_year`` pair, carrying the ``damage`` value at that
        quantile.

    Examples
    --------
    .. code-block:: python

        from climate_risk.models.damage_curves import RETURN_YEARS, damage_return_years, simulate_country_damage

        draws = simulate_country_damage(panel, frequency_idata, damage_idata, "NPL")
        curve = damage_return_years(draws, RETURN_YEARS)
    """
    quantiles = 1 - 1 / np.asarray(return_years, dtype=float)

    result = damage_draws.quantile(quantiles, dim="sample")
    result = result.assign_coords(quantile=list(return_years)).rename(quantile="return_year")

    return pl.from_pandas(result.rename("damage").to_dataframe().reset_index())
