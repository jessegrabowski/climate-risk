import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
import pytensor.tensor as pt

from climate_risk.models.blocks import add_data, add_hierarchical_effect
from climate_risk.stats.scaling import standardize

# Stage 1/2 parity minus square_ln_gdp_pc: the paper's damage section reuses "exactly the same
# linear structure we used for frequency mu" without saying which covariates that means, and the
# frequency notebook's quadratic GDP term is a frequency-specific addition, not part of the paper's
# named covariate list either way (see climate_risk/models/frequency.py's own note on this). This
# module is asked for parity minus that term, so it is the five the paper names outright.
#
# Population is logged rather than standardized raw. Raw population (millions) puts India and China
# 7-9 standard deviations out on the ~150-country training panel, and softplus(beta * z) at that z is
# where the country-level damage curve's implausible tail (documented separately, not in this repo)
# traces back to. Logging it first is the same treatment `ln_population_density` already gets, and
# matches the fix Jesse's own from-scratch rewrite of the frequency model already adopted for the
# identical reason (unmerged PR #217, notebooks/global_emdat/01_zinb_glm.ipynb: "Standardized raw
# population puts China at eleven standard deviations, and a prior on a coefficient multiplying that
# has no scale to be set on").
FEATURES = [
    "ln_population_density",
    "ln_gdp_pc",
    "precip_deviation",
    "co2",
    "ln_population",
]

TARGET = "Total_Damage_Adjusted_hydro"

# HurdleLogNormal's mu is a LogNormal location and needs no positivity constraint on its own; it is
# softplus-linked anyway because the paper is explicit that damage reuses frequency's linear
# structure unchanged. Unlike frequency's rate, mu never gets exponentiated inside the likelihood,
# so there is no overflow mode to clip against here — only the floor matters, and softplus already
# provides it.
MIN_MU = 1e-8


def _with_ln_population(panel: pl.DataFrame) -> pl.DataFrame:
    """Add `ln_population`, the log of the millions-scale `population` column FEATURES now uses."""
    return panel.with_columns(pl.col("population").log().alias("ln_population"))


def _prepare_panel(panel: pl.DataFrame) -> pd.DataFrame:
    """
    Drop a null covariate, zero-fill a null damage figure, and standardise the covariates.

    A country-year's ``Total_Damage_Adjusted_hydro`` is null under exactly one condition: no
    hydrometeorological-class event was recorded that year at all. An event that was recorded but
    never had a dollar figure attached to it does not leave this column null — ``total_damage``
    (:mod:`climate_risk.data_functions.emdat_processing`) sums each country-year's events with
    ``polars``, and a sum over an existing group whose members are all null comes back ``0.0``, not
    null. Checked against the current panel: zero country-years, in Nepal or globally, carry a
    non-null, non-zero ``hydrological_disasters`` count alongside a null damage figure. So a null
    here is not the ambiguous case the frequency target's null was; it is a direct consequence of
    there being no qualifying event, not a gap that might be hiding an unrecorded one, and it is
    zero-filled on that basis, not by analogy with :mod:`climate_risk.models.frequency`.

    That said, the ambiguity the frequency model's null carried has not disappeared, it has moved:
    of the country-years that did have a qualifying event, 1,968 of 3,738 (53%) show exactly ``0.0``
    damage, and the mechanism above means most of those are events EM-DAT never priced rather than
    events confirmed to have cost nothing. That conflation is baked into ``total_damage`` upstream
    of this module, applies equally to a genuine zero and an unquantified one, and is out of scope
    to fix here; a Hurdle model's ``psi`` is being fit against that mixture, not against confirmed
    zero-damage years alone.

    Parameters
    ----------
    panel : DataFrame
        A polars frame in the shape :func:`climate_risk.replication_data.create_replication_data`
        returns, carrying ``ISO`` plus every column in :data:`FEATURES` and :data:`TARGET`.

    Returns
    -------
    DataFrame
        One row per retained country-year, carrying ``ISO``, the raw :data:`TARGET`, and one
        ``*__standardized`` column per entry of :data:`FEATURES`.
    """
    complete = (
        _with_ln_population(panel).drop_nulls(subset=FEATURES).with_columns(pl.col(TARGET).fill_null(0)).to_pandas()
    )
    _, standardized = standardize(complete, FEATURES)

    return standardized


def build_damage_model(panel: pl.DataFrame) -> pm.Model:
    """
    Stage 2 of the paper's disaster-damage model (§4.2): a hierarchical country intercept.

    At parity with :func:`climate_risk.models.frequency.build_hierarchical_frequency_model`: the
    same five covariates (:data:`FEATURES`, everything Stage 2 uses except ``square_ln_gdp_pc``),
    the same standardisation, a hierarchical country intercept built the same way with
    :func:`~climate_risk.models.blocks.add_hierarchical_effect` and ``use_zerosum_offset=True`` for
    the identical reason given there, and the same softplus link. The likelihood is a
    :class:`~pymc.HurdleLogNormal` on raw ``Total_Damage_Adjusted_hydro`` rather than a
    Zero-Inflated Negative Binomial on a count: ``psi`` is the probability a country-year has any
    recorded hydrometeorological damage at all, ``sigma`` (a Half-Normal) is the spread of
    log-damage where it is positive, and ``mu`` is the same country-effect-plus-covariates linear
    predictor Stage 2 already uses for frequency, softplus-linked because the paper states the
    damage model reuses that structure unchanged rather than because a LogNormal location needs it.

    Parameters
    ----------
    panel : DataFrame
        A polars frame in the shape :func:`climate_risk.replication_data.create_replication_data`
        returns, carrying ``ISO`` plus every column in :data:`FEATURES` and :data:`TARGET`.

    Returns
    -------
    Model
        Free variables ``country_effect_loc``, ``country_effect_scale``, ``country_effect_offset``
        (dims ``ISO``), ``beta`` (one coefficient per entry of :data:`FEATURES`), ``psi`` and
        ``sigma``, and an observed ``Total_Damage_Adjusted_hydro``.
    """
    standardized = _prepare_panel(panel)
    standardized_features = [f"{feature}__standardized" for feature in FEATURES]
    iso_idx, isos = pd.factorize(standardized["ISO"], sort=True)

    coords = {"feature": FEATURES, "obs_id": np.arange(len(standardized)), "ISO": isos}

    with pm.Model(coords=coords) as model:
        X, y = add_data(standardized_features, standardized, target=TARGET, dims=["obs_id", "feature"])
        iso_idx_data = pm.Data("iso_idx", iso_idx, dims="obs_id")

        country_effect, _, _, _ = add_hierarchical_effect(name="country", group_dim="ISO", use_zerosum_offset=True)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims="feature")

        latent_rate = country_effect[iso_idx_data] + X @ beta
        mu = pt.clip(pt.softplus(latent_rate), MIN_MU, np.inf)

        psi = pm.Beta("psi", alpha=1.0, beta=1.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        pm.HurdleLogNormal(TARGET, psi=psi, mu=mu, sigma=sigma, observed=y, dims="obs_id")

    return model
