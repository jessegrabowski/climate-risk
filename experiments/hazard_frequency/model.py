"""
The year-effect frequency model with local hazard covariates, and one slope that varies by region.

Same likelihood, priors and year effect as ``climate_risk.models.identification.build_year_effect_model`` (the 1b model
of the identification tests). Two additions: covariates may be any panel column, and one covariate (``roni_peak``) can
get a slope that differs by region, drawn around a shared mean (non-centred, half-normal spread).
"""

import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
import pytensor.tensor as pt

from climate_risk.models.blocks import add_data, add_hierarchical_effect
from climate_risk.models.frequency import MAX_MU, MIN_MU, TARGET
from climate_risk.models.identification import _prepare, _year_index


def within_country_z(panel: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """
    Replace each column by its z-score within the country (mean 0, sd 1 over that country's own years).

    A country with no spread in a column (or a single year) gets 0, so it contributes nothing to the slope.
    A missing value stays missing, so the row is dropped later rather than treated as an average year.
    """
    out = []
    for column in columns:
        centred = pl.col(column) - pl.col(column).mean().over("ISO")
        spread = pl.col(column).std().over("ISO")
        # A missing value stays missing: only a country that has data but no spread is set to 0.
        out.append(
            pl.when(pl.col(column).is_null())
            .then(None)
            .when(spread > 0)
            .then(centred / spread)
            .otherwise(0.0)
            .alias(column)
        )

    return panel.with_columns(out)


def build_hazard_model(
    panel: pl.DataFrame,
    features: list[str],
    *,
    regional_feature: str | None = None,
    region_column: str = "region",
    target: str = TARGET,
    year_effect_sd_prior: float = 0.3,
    regional_sd_prior: float = 0.5,
) -> pm.Model:
    """
    Year-effect frequency model with covariates ``features`` and an optional region-varying slope.

    Parameters
    ----------
    panel : DataFrame
        Replication panel with the columns in ``features``, ``regional_feature``, ``region_column`` and ``target``.
    features : list of str
        Covariates with one slope for all countries (each standardised over the rows kept).
    regional_feature : str, optional
        A covariate whose slope is ``beta_shared + regional_sd * offset[region]``. Default None (no such term).
    region_column : str, optional
        Region label of each row. Default ``region``.
    target : str, optional
        Count column. Default ``hydrological_disasters``.
    year_effect_sd_prior : float, optional
        Scale of the half-normal prior on the random-walk step size. Default 0.3.
    regional_sd_prior : float, optional
        Scale of the half-normal prior on the spread of regional slopes. Default 0.5.

    Returns
    -------
    Model
        Observed variable named ``target``. Slopes ``beta`` (dim ``feature``); with a regional term also
        ``beta_regional`` (dim ``region``) and ``regional_sd``.
    """
    everything = [*features, regional_feature] if regional_feature else list(features)
    standardized = _prepare(panel, everything, [target])
    iso_idx, isos = pd.factorize(standardized["ISO"], sort=True)
    years, year_idx = _year_index(standardized)
    coords: dict[str, object] = {
        "feature": features,
        "obs_id": np.arange(len(standardized)),
        "ISO": isos,
        "year": years,
    }
    if regional_feature:
        region_idx, regions = pd.factorize(standardized[region_column], sort=True)
        coords["region"] = regions

    with pm.Model(coords=coords) as model:
        X, y = add_data(
            [f"{feature}__standardized" for feature in features],
            standardized,
            target=target,
            dims=["obs_id", "feature"],
        )
        iso_idx_data = pm.Data("iso_idx", iso_idx, dims="obs_id")
        year_idx_data = pm.Data("year_idx", year_idx, dims="obs_id")

        country_effect, _, _, _ = add_hierarchical_effect(name="country", group_dim="ISO", use_zerosum_offset=True)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims="feature")

        year_effect_sd = pm.HalfNormal("year_effect_sd", sigma=year_effect_sd_prior)
        year_effect_step = pm.Normal("year_effect_step", mu=0.0, sigma=1.0, shape=len(years) - 1)
        year_effect = pm.Deterministic(
            "year_effect", pt.concatenate([pt.zeros(1), pt.cumsum(year_effect_sd * year_effect_step)]), dims="year"
        )

        linear = country_effect[iso_idx_data] + year_effect[year_idx_data] + X @ beta
        if regional_feature:
            region_idx_data = pm.Data("region_idx", region_idx, dims="obs_id")
            z = pm.Data("regional_x", standardized[f"{regional_feature}__standardized"].to_numpy(), dims="obs_id")
            shared = pm.Normal("beta_regional_shared", mu=0.0, sigma=1.0)
            regional_sd = pm.HalfNormal("regional_sd", sigma=regional_sd_prior)
            offset = pm.Normal("regional_offset", mu=0.0, sigma=1.0, dims="region")
            beta_regional = pm.Deterministic("beta_regional", shared + regional_sd * offset, dims="region")
            linear = linear + z * beta_regional[region_idx_data]

        mu = pt.clip(pt.softplus(linear), MIN_MU, MAX_MU)
        psi = pm.Beta("psi", alpha=1.0, beta=1.0)
        alpha = pm.Gamma("alpha", alpha=2.0, beta=1.0)

        pm.ZeroInflatedNegativeBinomial(target, psi=psi, mu=mu, alpha=alpha, observed=y, dims="obs_id")

    return model
