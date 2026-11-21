"""
EXPERIMENTAL panel helpers for the CO2-identification tests (tasks/2026-11-21-climate-identification-tests.md).

Nothing here changes how the published panel or models behave. Each helper takes the finished panel
from :func:`climate_risk.replication_data.create_replication_data` and the EM-DAT event table, and
returns a copy of the panel with one count column recomputed or added.

The convention is the panel's own: a country-year with no recorded event of the counted types reads as
null, not zero. The frequency models zero-fill a null target themselves, so null here means "no record",
which is the reporting assumption the panel already makes (see ``_counted_or_missing``).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
import pytensor.tensor as pt

from climate_risk.config.schema import EventFilters
from climate_risk.data_functions.emdat_processing import event_filter
from climate_risk.models.blocks import add_data, add_hierarchical_effect
from climate_risk.models.frequency import FEATURES, MAX_MU, MIN_MU, TARGET, _with_ln_population
from climate_risk.replication_data import HYDROLOGICAL_TYPES, PANEL_KEY
from climate_risk.stats.scaling import standardize

# The EM-DAT types used as a control group: events with no pathway from a changing climate, but with
# the same reporting history as the climate hazards. Deliberately not added to ``DISASTER_CLASSES``.
GEOPHYSICAL_TYPES = ("Earthquake", "Volcanic activity", "Mass movement (dry)")

GEOPHYSICAL_COLUMN = "geophysical_disasters"


def recount_events(
    panel: pl.DataFrame,
    events: pl.DataFrame,
    types: Sequence[str],
    *,
    column: str,
    filters: EventFilters | None = None,
) -> pl.DataFrame:
    """
    Return the panel with ``column`` holding the count of ``types`` events under ``filters``.

    The panel's rows are kept exactly as they are, so a fit on the new column sees the same
    country-years as a fit on the published one.

    Parameters
    ----------
    panel : DataFrame
        The annual replication panel, keyed on ``ISO`` and ``date``.
    events : DataFrame
        EM-DAT events from ``load_emdat_events``, carrying ``ISO``, ``date``, ``Disaster Type``,
        ``Deaths`` and ``Total_Affected``.
    types : sequence of str
        The ``Disaster Type`` values to count.
    column : str
        Name of the count column. An existing column of that name is replaced in place.
    filters : EventFilters, optional
        Which events count. Default ``EventFilters()``, the thresholds the published panel uses.

    Returns
    -------
    recounted : DataFrame
        The panel with ``column`` set. A country-year with no matching event is null.
    """
    counted = (
        events.with_columns(pl.col("date").dt.truncate("1y"))
        .filter(event_filter(filters or EventFilters()) & pl.col("Disaster Type").is_in(list(types)))
        .group_by(PANEL_KEY)
        .len()
        .select(*PANEL_KEY, pl.col("len").cast(pl.Float64).alias(column))
    )

    # An existing column keeps its position; a new one goes on the end.
    order = panel.columns if column in panel.columns else [*panel.columns, column]

    return panel.drop(column, strict=False).join(counted, on=PANEL_KEY, how="left").select(order)


def with_geophysical_disasters(panel: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """
    Add ``geophysical_disasters``: earthquakes, volcanic activity and dry mass movements per country-year.

    Same event filters and same null-means-no-record convention as the published counts, so it can
    stand in for the target in the placebo and joint models.

    Parameters
    ----------
    panel : DataFrame
        The annual replication panel.
    events : DataFrame
        EM-DAT events from ``load_emdat_events``.

    Returns
    -------
    extended : DataFrame
        The panel with one more column, ``geophysical_disasters``, as a float count.
    """
    return recount_events(panel, events, GEOPHYSICAL_TYPES, column=GEOPHYSICAL_COLUMN)


def with_high_impact_hydro(panel: pl.DataFrame, events: pl.DataFrame, *, min_deaths: int) -> pl.DataFrame:
    """
    Replace ``hydrological_disasters`` by the count of events that killed more than ``min_deaths`` people.

    Big events are the ones least likely to depend on reporting effort, so this is the Test 3 target.

    Parameters
    ----------
    panel : DataFrame
        The annual replication panel.
    events : DataFrame
        EM-DAT events from ``load_emdat_events``.
    min_deaths : int
        An event must kill more than this many people to count.

    Returns
    -------
    high_impact : DataFrame
        The panel with ``hydrological_disasters`` recounted under the death threshold.
    """
    return recount_events(
        panel,
        events,
        HYDROLOGICAL_TYPES,
        column="hydrological_disasters",
        filters=EventFilters(min_deaths=min_deaths),
    )


HAMILTON_COLUMN = "hamilton_temp_shock"
JOINT_CLASSES = ("hydro", "geophysical")
JOINT_TARGETS = {"hydro": TARGET, "geophysical": GEOPHYSICAL_COLUMN}


def hamilton_shock(series: pd.Series, *, horizon: int = 2, lags: int = 2) -> pd.Series:
    """
    Return the Hamilton-filter shock of a yearly series: what the series' own past cannot predict.

    Regress ``y[t]`` on a constant and ``y[t - horizon]`` ... ``y[t - horizon - lags + 1]``. With the
    defaults that is a constant, ``y[t-2]`` and ``y[t-3]``. The residual is the shock. Unlike a
    smoothing filter it uses only the past, so a shock at ``t`` does not depend on later years.

    Parameters
    ----------
    series : Series
        A yearly series indexed by year, in order, with no gaps.
    horizon : int, optional
        How far back the nearest regressor sits. Default 2.
    lags : int, optional
        How many consecutive lagged values enter. Default 2.

    Returns
    -------
    shock : Series
        The residuals, indexed by year. The first ``horizon + lags - 1`` years have no shock and are
        left out.
    """
    first = horizon + lags - 1
    regressors = np.column_stack(
        [np.ones(len(series) - first)] + [series.shift(horizon + k).iloc[first:] for k in range(lags)]
    )
    target = series.iloc[first:]
    coefficients, *_ = np.linalg.lstsq(regressors, target.to_numpy(), rcond=None)

    return pd.Series(target.to_numpy() - regressors @ coefficients, index=target.index, name=HAMILTON_COLUMN)


def with_hamilton_temperature_shock(panel: pl.DataFrame, temperature: pl.DataFrame) -> pl.DataFrame:
    """
    Add ``hamilton_temp_shock``: the Hamilton-filter shock of the annual global ocean temperature.

    Every country gets the same value in a given year, as with ``co2``.

    Parameters
    ----------
    panel : DataFrame
        The annual replication panel.
    temperature : DataFrame
        One row per year carrying ``date`` and ``Temp``, from ``build_time_series``.

    Returns
    -------
    extended : DataFrame
        The panel with ``hamilton_temp_shock`` on the end. A year with no shock is null.
    """
    observed = temperature.drop_nulls("Temp").sort("date").to_pandas()
    yearly = pd.Series(observed["Temp"].to_numpy(), index=observed["date"].dt.year.to_numpy())
    shock = hamilton_shock(yearly)
    lookup = pl.DataFrame({"year": shock.index.to_numpy(), HAMILTON_COLUMN: shock.to_numpy()})

    return (
        panel.with_columns(pl.col("date").dt.year().cast(pl.Int64).alias("year"))
        .join(lookup.with_columns(pl.col("year").cast(pl.Int64)), on="year", how="left")
        .drop("year")
    )


def _prepare(panel: pl.DataFrame, features: Sequence[str], targets: Sequence[str]) -> pd.DataFrame:
    """Drop rows missing a covariate, zero-fill null counts and standardise the covariates."""
    complete = (
        _with_ln_population(panel)
        .drop_nulls(subset=list(features))
        .with_columns([pl.col(target).fill_null(0) for target in targets])
        .to_pandas()
    )
    _, standardized = standardize(complete, list(features))

    return standardized


def _year_index(standardized: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The calendar years the rows span and each row's position in them."""
    row_years = pd.to_datetime(standardized["date"]).dt.year.to_numpy()
    years = np.arange(row_years.min(), row_years.max() + 1)

    return years, row_years - row_years.min()


def build_year_effect_model(
    panel: pl.DataFrame,
    features: Sequence[str] = tuple(FEATURES),
    *,
    target: str = TARGET,
    year_effect_sd_prior: float = 0.3,
) -> pm.Model:
    """
    EXPERIMENTAL: the year-effect frequency model with a chosen list of covariates.

    With the default ``features`` and ``target`` this is the same model as
    :func:`climate_risk.models.frequency.build_year_effect_frequency_model` (a test checks that the two
    give the same log-probability). It exists so Test 1 can drop ``co2`` or swap it for another common
    time driver without touching the published builder.

    Parameters
    ----------
    panel : DataFrame
        The replication panel, with every column in ``features`` and ``target``, plus ``ISO`` and ``date``.
    features : sequence of str, optional
        Covariates, each standardised over the rows kept. Default the published ``FEATURES``.
    target : str, optional
        The count column. Default ``hydrological_disasters``.
    year_effect_sd_prior : float, optional
        Scale of the ``HalfNormal`` prior on the random walk's step size. Default 0.3.

    Returns
    -------
    Model
        Variables named as in the published year-effect model.
    """
    features = list(features)
    standardized = _prepare(panel, features, [target])
    iso_idx, isos = pd.factorize(standardized["ISO"], sort=True)
    years, year_idx = _year_index(standardized)
    coords = {"feature": features, "obs_id": np.arange(len(standardized)), "ISO": isos, "year": years}

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

        mu = pt.clip(pt.softplus(country_effect[iso_idx_data] + year_effect[year_idx_data] + X @ beta), MIN_MU, MAX_MU)
        psi = pm.Beta("psi", alpha=1.0, beta=1.0)
        alpha = pm.Gamma("alpha", alpha=2.0, beta=1.0)

        pm.ZeroInflatedNegativeBinomial(target, psi=psi, mu=mu, alpha=alpha, observed=y, dims="obs_id")

    return model


def build_joint_year_effect_model(
    panel: pl.DataFrame,
    features: Sequence[str] = tuple(FEATURES),
    *,
    co2_on_geophysical: bool,
    shared_year_effect: bool = True,
    year_effect_sd_prior: float = 0.3,
) -> pm.Model:
    """
    EXPERIMENTAL: hydrometeorological and geophysical counts in one model (Tests 2b, 2c and 2d).

    Each country-year has two counts. The geophysical class has no climate pathway but shares EM-DAT's
    reporting history, so it measures what the year effect has to absorb. The structure follows
    ``notebooks/global_emdat/06_joint_country_intercept.ipynb``: a class-specific intercept, slopes
    and overdispersion, and country intercepts drawn jointly (non-centred, LKJ prior on their correlation).
    The likelihood is the Stage 2 zero-inflated negative binomial with its own ``psi`` per class, so the
    hydro column stays comparable with the single-class fits.

    The year effect is a random walk fixed at zero in the first year. By default one walk is added to
    both classes' log rate, which assumes reporting changes move both classes by the same amount on the
    log scale. ``shared_year_effect=False`` gives each class its own walk, to test that assumption.

    Parameters
    ----------
    panel : DataFrame
        The replication panel with ``geophysical_disasters`` added (see :func:`with_geophysical_disasters`).
    features : sequence of str, optional
        Covariates. Must include ``co2``. Default the published ``FEATURES``.
    co2_on_geophysical : bool
        If False the ``co2`` slope is fixed at 0 for geophysical counts (Test 2b). If True it is free (Test 2c).
    shared_year_effect : bool, optional
        One walk for both classes (default) or one per class.
    year_effect_sd_prior : float, optional
        Scale of the ``HalfNormal`` prior on the step size. Default 0.3.

    Returns
    -------
    Model
        Slopes ``beta`` (dims ``feature``, ``disaster``) of which the masked one is a prior draw that never
        enters the likelihood; use ``beta_used`` for the slopes the likelihood sees. The observed variable is ``y``.
    """
    features = list(features)
    targets = [JOINT_TARGETS[name] for name in JOINT_CLASSES]
    standardized = _prepare(panel, features, targets)
    iso_idx, isos = pd.factorize(standardized["ISO"], sort=True)
    years, year_idx = _year_index(standardized)
    classes = list(JOINT_CLASSES)

    mask = np.ones((len(features), len(classes)))
    if not co2_on_geophysical:
        mask[features.index("co2"), classes.index("geophysical")] = 0.0

    coords = {
        "feature": features,
        "obs_id": np.arange(len(standardized)),
        "ISO": isos,
        "year": years,
        "disaster": classes,
        "walk": ["shared"] if shared_year_effect else classes,
    }

    with pm.Model(coords=coords) as model:
        X = pm.Data(
            "X",
            standardized[[f"{feature}__standardized" for feature in features]].to_numpy(),
            dims=["obs_id", "feature"],
        )
        y = standardized[targets].to_numpy()
        iso_idx_data = pm.Data("iso_idx", iso_idx, dims="obs_id")
        year_idx_data = pm.Data("year_idx", year_idx, dims="obs_id")

        intercept = pm.Normal("intercept", 0.0, 1.0, dims="disaster")
        beta = pm.Normal("beta", 0.0, 1.0, dims=["feature", "disaster"])
        beta_used = pm.Deterministic("beta_used", beta * mask, dims=["feature", "disaster"])

        country_chol, country_corr, _ = pm.LKJCholeskyCov(
            "country_cov", n=2, eta=2, sd_dist=pm.HalfNormal.dist(1.0), compute_corr=True
        )
        pm.Deterministic("country_corr", country_corr[0, 1])
        country_offset = pm.Normal("country_offset", 0.0, 1.0, dims=["ISO", "disaster"])
        country_effect = country_offset @ country_chol.T

        n_walks = 1 if shared_year_effect else len(classes)
        year_effect_sd = pm.HalfNormal("year_effect_sd", sigma=year_effect_sd_prior, shape=n_walks)
        year_effect_step = pm.Normal("year_effect_step", 0.0, 1.0, shape=(len(years) - 1, n_walks))
        walk = pt.concatenate([pt.zeros((1, n_walks)), pt.cumsum(year_effect_sd * year_effect_step, axis=0)], axis=0)
        # One column broadcasts over both classes; two columns give each class its own walk.
        year_effect = pm.Deterministic("year_effect", walk, dims=["year", "walk"])

        log_rate = intercept + X @ beta_used + country_effect[iso_idx_data] + year_effect[year_idx_data]
        mu = pt.clip(pt.softplus(log_rate), MIN_MU, MAX_MU)

        psi = pm.Beta("psi", alpha=1.0, beta=1.0, dims="disaster")
        alpha = pm.Gamma("alpha", alpha=2.0, beta=1.0, dims="disaster")

        pm.ZeroInflatedNegativeBinomial("y", psi=psi, mu=mu, alpha=alpha, observed=y, dims=["obs_id", "disaster"])

    return model
