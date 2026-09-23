from collections.abc import Sequence

import numpy as np
import polars as pl
import pymc as pm
import xarray as xr

from climate_risk.models import damage, frequency
from climate_risk.models.damage import TARGET as DAMAGE_TARGET
from climate_risk.models.damage import _prepare_panel as _prepare_damage_panel
from climate_risk.models.frequency import TARGET as FREQUENCY_TARGET
from climate_risk.models.frequency import _prepare_panel as _prepare_frequency_panel
from climate_risk.stats.scaling import standardize

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

    freq_years = freq_rows.loc[freq_rows["ISO"] == iso, "date"]
    damage_years = damage_rows.loc[damage_rows["ISO"] == iso, "date"]

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


def _standardized_features(
    *, future_covariates: pl.DataFrame, historical_panel: pl.DataFrame, features: list[str], target: str
) -> np.ndarray:
    """Standardize future rows with the scaler the historical panel would fit for these features."""
    complete_history = historical_panel.drop_nulls(features).with_columns(pl.col(target).fill_null(0)).to_pandas()
    scaler, _ = standardize(complete_history, features)

    _, standardized = standardize(future_covariates.to_pandas(), features, transformer_fitted=scaler)
    result: np.ndarray = standardized[[f"{feature}__standardized" for feature in features]].to_numpy()

    return result


def _future_mu(
    *,
    future_covariates: pl.DataFrame,
    historical_panel: pl.DataFrame,
    features: list[str],
    target: str,
    idata: xr.DataTree,
    iso: str,
) -> np.ndarray:
    """softplus(country_effect + X @ beta) at future covariate rows, dims (chain, draw, year)."""
    x = _standardized_features(
        future_covariates=future_covariates, historical_panel=historical_panel, features=features, target=target
    )
    posterior = idata.posterior
    x_da = xr.DataArray(x, dims=["year", "feature"], coords={"feature": features})
    latent_rate = posterior["country_effect"].sel(ISO=iso) + (x_da * posterior["beta"]).sum(dim="feature", skipna=False)
    result: np.ndarray = np.logaddexp(0.0, latent_rate.transpose("chain", "draw", "year").values)

    return result


def _wrap_future_draws(draws: np.ndarray, future_covariates: pl.DataFrame) -> xr.DataArray:
    """Package (chain, draw, year) draws into simulate_country_damage's (sample, year) shape."""
    tagged = xr.DataArray(draws, dims=["chain", "draw", "year"], coords={"year": future_covariates["year"].to_list()})
    stacked = tagged.stack(sample=("chain", "draw"))
    result: xr.DataArray = stacked.drop_vars(["chain", "draw", "sample"])

    return result


def simulate_future_damage(
    *,
    future_covariates: pl.DataFrame,
    historical_panel: pl.DataFrame,
    frequency_idata: xr.DataTree,
    damage_idata: xr.DataTree,
    iso: str,
    random_seed: int = 0,
) -> xr.DataArray:
    """
    Draw a country's total-damage distribution for years the models were never fit on.

    Unlike :func:`simulate_country_damage`, which pairs the two Stage 2 models' own posterior
    predictive draws position-by-position, this predicts at covariates the models have not seen:
    it standardises them with the scaler the training panel would fit, then draws directly from
    each model's posterior at those new points, the same product-of-independent-draws construction
    :func:`simulate_country_damage` uses otherwise.

    Parameters
    ----------
    future_covariates : DataFrame
        Columns ``ISO``, ``year``, and every column in
        :data:`~climate_risk.models.frequency.FEATURES` (a superset of damage's), one row per year
        to predict. From :func:`~climate_risk.models.future_covariates.project_future_covariates`.
    historical_panel : DataFrame
        The panel both models were fit on, in
        :func:`~climate_risk.replication_data.create_replication_data`'s shape. Read only to refit
        the standardization the training run used; not itself predicted from.
    frequency_idata : DataTree
        Posterior from :func:`~climate_risk.models.frequency.build_hierarchical_frequency_model`.
    damage_idata : DataTree
        Posterior from :func:`~climate_risk.models.damage.build_damage_model`, sharing the same
        chain and draw counts as ``frequency_idata``.
    iso : str
        ISO alpha-3 code of the country to predict. Must be one both models' ``ISO`` coordinate
        already carries.
    random_seed : int, optional
        Default 0.

    Returns
    -------
    damage_draws : DataArray
        Dims ``sample`` and ``year``, in the shape :func:`damage_return_years` reads.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        import xarray as xr

        from climate_risk.models.damage_curves import simulate_future_damage
        from climate_risk.models.future_covariates import project_future_covariates
        from climate_risk.replication_data import create_replication_data

        cache_dir = Path("data")
        historical_panel = create_replication_data(cache_dir)
        forecast_idata = xr.open_datatree("data/idata/climate_economy_npl_forecast.nc")
        future_covariates = project_future_covariates(
            iso="NPL",
            forecast_idata=forecast_idata,
            target_years=[2030, 2050],
            historical_panel=historical_panel,
            cache_dir=cache_dir,
        )
        frequency_idata = xr.open_datatree("data/idata/frequency_stage2.nc")
        damage_idata = xr.open_datatree("data/idata/damage_stage2.nc")

        draws = simulate_future_damage(
            future_covariates=future_covariates,
            historical_panel=historical_panel,
            frequency_idata=frequency_idata,
            damage_idata=damage_idata,
            iso="NPL",
        )
    """
    # damage.FEATURES reads `ln_population`, derived here rather than upstream so historical_panel and
    # future_covariates keep carrying raw `population` for frequency.FEATURES, which still standardizes it
    # directly.
    historical_panel = damage._with_ln_population(historical_panel)
    future_covariates = damage._with_ln_population(future_covariates)

    freq_mu = _future_mu(
        future_covariates=future_covariates,
        historical_panel=historical_panel,
        features=frequency.FEATURES,
        target=FREQUENCY_TARGET,
        idata=frequency_idata,
        iso=iso,
    )
    freq_posterior = frequency_idata.posterior
    freq_dist = pm.ZeroInflatedNegativeBinomial.dist(
        psi=freq_posterior["psi"].values[..., None],
        mu=np.clip(freq_mu, frequency.MIN_MU, frequency.MAX_MU),
        alpha=freq_posterior["alpha"].values[..., None],
    )
    freq_draws = _wrap_future_draws(pm.draw(freq_dist, random_seed=random_seed), future_covariates)

    damage_mu = _future_mu(
        future_covariates=future_covariates,
        historical_panel=historical_panel,
        features=damage.FEATURES,
        target=DAMAGE_TARGET,
        idata=damage_idata,
        iso=iso,
    )
    damage_posterior = damage_idata.posterior
    damage_dist = pm.HurdleLogNormal.dist(
        psi=damage_posterior["psi"].values[..., None],
        mu=np.clip(damage_mu, damage.MIN_MU, np.inf),
        sigma=damage_posterior["sigma"].values[..., None],
    )
    damage_draws = _wrap_future_draws(pm.draw(damage_dist, random_seed=random_seed), future_covariates)

    result: xr.DataArray = freq_draws * damage_draws

    return result
