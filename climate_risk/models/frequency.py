import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
import pytensor.tensor as pt

from climate_risk.models.blocks import add_data, add_hierarchical_effect
from climate_risk.stats.scaling import standardize

# The paper's own covariate list (Modeling disaster frequency): ln_population_density, ln_gdp_pc,
# precip_deviation, co2 and population. The notebook that implements it (02_frequency_modeling,
# "Covariates Only") also carries square_ln_gdp_pc, giving GDP per capita a quadratic term; the
# notebook is the executable spec, so its list is the one this module follows.
#
# Deliberately absent: log_distance_to_river and log_distance_to_coastline. Despite the name, they
# are not columns create_replication_data ever produces, and they are not covariates of this model
# at all. Paper §4.6 attaches them to grid cells for the separate point-level HSGP geospatial model
# (climate_risk/models/aggregated_poisson.py's eventual caller), which pairs each disaster with a
# raster cell rather than a country-year. Nothing here drops them from a candidate set; they were
# never candidates for a country-year panel in the first place.
FEATURES = [
    "ln_population_density",
    "ln_gdp_pc",
    "square_ln_gdp_pc",
    "precip_deviation",
    "co2",
    "population",
]

TARGET = "hydrological_disasters"

# Keeps mu away from the two places a Negative Binomial's log-probability is undefined or
# overflows: exactly zero, and a rate large enough that an early, unconstrained NUTS step turns it
# into inf. The notebook's covariates-only model applies the same clip for the same reason.
MIN_MU = 1e-8
MAX_MU = 300.0


def _prepare_panel(panel: pl.DataFrame) -> pd.DataFrame:
    """Drop a null covariate, zero-fill a null target, and standardise the covariates."""
    complete = panel.drop_nulls(subset=FEATURES).with_columns(pl.col(TARGET).fill_null(0)).to_pandas()
    _, standardized = standardize(complete, FEATURES)

    return standardized


def build_frequency_model(panel: pl.DataFrame) -> pm.Model:
    """
    Stage 1 of the paper's disaster-frequency model (§4.2): covariates only.

    A country-year's hydrometeorological disaster count is a Zero-Inflated Negative Binomial whose
    mean follows a softplus-linked linear function of standardised covariates. There is no country
    hierarchy and no spatial term yet: every observation shares one intercept and one set of
    coefficients, which is what "Stage 1" means here. The hierarchy, the Mundlak random slopes and
    the BYM spatial term are later stages, built on top of this one, not inside it.

    A country-year missing a covariate is dropped: there is nothing to standardise it against. A
    country-year whose ``hydrological_disasters`` count is null is treated differently, and filled
    with zero rather than dropped. ``create_replication_data`` only ever carries a country for the
    years it appears in EM-DAT's own coverage window (:func:`~climate_risk.data_functions.emdat_processing.country_year_grid`),
    so a null there already means "monitored, and no hydrological event of any type was recorded,"
    not "unknown" in the way an absent row would. Filling it with zero is still an assumption, not a
    fact the data states outright — the notebook this ports flags the same fill, on the same
    column, as "a terrible assumption" in its own comment, made there only to keep sampling
    tractable. It is made deliberately here instead: dropping these rows removes every zero
    observation from the likelihood, and a Zero-Inflated Negative Binomial with no zeros to explain
    has nothing to identify ``psi`` against, so it saturates near 1 and the zero-inflation the paper
    motivates never engages, silently. Zero-filling is what lets that parameter, and the model, do
    the job the paper describes; a discrepancy against Tables 9-11 traceable to it is not a bug.

    Parameters
    ----------
    panel : DataFrame
        A polars frame in the shape :func:`climate_risk.replication_data.create_replication_data`
        returns, carrying every column in :data:`FEATURES` and :data:`TARGET`.

    Returns
    -------
    Model
        Free variables ``intercept``, ``beta`` (one coefficient per entry of :data:`FEATURES`),
        ``psi`` and ``alpha``, and an observed ``hydrological_disasters``.
    """
    standardized = _prepare_panel(panel)
    standardized_features = [f"{feature}__standardized" for feature in FEATURES]

    coords = {"feature": FEATURES, "obs_id": np.arange(len(standardized))}

    with pm.Model(coords=coords) as model:
        X, y = add_data(standardized_features, standardized, target=TARGET, dims=["obs_id", "feature"])

        intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims="feature")

        latent_rate = intercept + X @ beta
        mu = pt.clip(pt.softplus(latent_rate), MIN_MU, MAX_MU)

        psi = pm.Beta("psi", alpha=1.0, beta=1.0)
        alpha = pm.Gamma("alpha", alpha=2.0, beta=1.0)

        pm.ZeroInflatedNegativeBinomial(TARGET, psi=psi, mu=mu, alpha=alpha, observed=y, dims="obs_id")

    return model


def build_hierarchical_frequency_model(panel: pl.DataFrame) -> pm.Model:
    """
    Stage 2 of the paper's disaster-frequency model (§4.2): a hierarchical country intercept.

    Everything from :func:`build_frequency_model` carries over unchanged — the same six covariates,
    the same standardisation, the same zero-filled target, the same softplus-linked Zero-Inflated
    Negative Binomial. The one addition is that the scalar ``intercept`` is replaced by a
    per-country effect, built with :func:`~climate_risk.models.blocks.add_hierarchical_effect`:
    every country gets its own baseline rate, drawn from a shared Normal(``country_loc``,
    ``country_scale``) rather than fitting 150-odd unrelated ones or forcing every country onto one
    global baseline. There is still no spatial term and no Mundlak random slopes; those are Stage 3.

    ``use_zerosum_offset=True`` matters here specifically because the effect already carries a
    separate location parameter (``country_loc``). ``effect = country_loc + country_scale *
    offset``, and if ``offset`` is an ordinary ``Normal`` with no constraint on its mean, the model
    can add any constant to ``country_loc`` and subtract the matching amount from every country's
    offset without changing a single ``effect`` value or the likelihood at all — ``country_loc`` and
    the offsets' shared level are not separately identified. NUTS still runs, but it now has to
    explore a ridge along that exact direction, which shows up as unusually poor mixing (low
    ``ess_bulk``/``ess_tail``, ``r_hat`` well above 1) concentrated on ``country_loc`` — and, in a
    pairs plot, as ``country_loc`` sitting in a tight negative-correlation band against the mean of
    ``country_effect_offset`` rather than a normal-looking cloud — while ``country_effect`` itself
    (the loc-plus-offset composite that actually enters the likelihood) still converges cleanly. A
    ``ZeroSumNormal`` offset is constrained to sum to zero across every country, which pins that
    shared level at exactly zero and removes the ridge: ``country_loc`` becomes the one thing left
    that can set it, and it does so uniquely.

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
        ``alpha``, and an observed ``hydrological_disasters``.
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
        mu = pt.clip(pt.softplus(latent_rate), MIN_MU, MAX_MU)

        psi = pm.Beta("psi", alpha=1.0, beta=1.0)
        alpha = pm.Gamma("alpha", alpha=2.0, beta=1.0)

        pm.ZeroInflatedNegativeBinomial(TARGET, psi=psi, mu=mu, alpha=alpha, observed=y, dims="obs_id")

    return model
