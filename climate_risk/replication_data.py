import logging

from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import polars as pl

from statsmodels.tsa.seasonal import STL

from climate_risk.config.registry import resolve_isos
from climate_risk.config.schema import Place
from climate_risk.data_functions.combine_data import (
    annual_precipitation,
    build_country_year_panel,
    build_time_series,
)
from climate_risk.data_functions.shapefiles_data_loader import load_shapefile
from climate_risk.geo.raster import ISO_COLUMN

_log = logging.getLogger(__name__)

PANEL_KEY = ["ISO", "year"]

HYDROLOGICAL_TYPES = ["Flood", "Storm"]
CLIMATOLOGICAL_TYPES = ["Extreme temperature", "Wildfire", "Drought"]

# The WMO reference period each country's precipitation is centered on, inclusive of both ends.
CLIMATOLOGY_BASELINE = (1961, 1990)

# The seasonal period the ocean-heat trend is fitted with.
OCEAN_TREND_PERIOD = 3

# The trend regressor counts years over a century, so it stays comparable with the other columns.
TREND_BASE_YEAR = 1980

MILLION = 1e6

# The covariates every model in the paper conditions on. A country-year missing any one of them
# cannot enter, because a regressor matrix has no room for a hole.
MODEL_FEATURES = (
    "ln_population_density",
    "ln_gdp_pc",
    "square_ln_gdp_pc",
    "precip_deviation",
    "co2",
    "population",
)

PUBLISHED_COLUMNS = [
    "ISO",
    "year",
    "climatological_disasters",
    "hydrological_disasters",
    "population",
    "ln_population_density",
    "ln_gdp_pc",
    "square_ln_gdp_pc",
    "dev_from_trend_ocean_temp",
    "co2",
    "precip_deviation",
    "Total_Damage_Adjusted_hydro",
    "Total_Damage_Adjusted_clim",
    "Total_Affected_hydro",
]


def _counted_or_missing(types: list[str]) -> pl.Expr:
    """Total the given disaster types, keeping a country-year with no record of any of them missing."""
    return (
        pl.when(pl.all_horizontal(pl.col(name).is_null() for name in types))
        .then(None)
        .otherwise(pl.sum_horizontal(pl.col(name) for name in types))
    )


def _precipitation_deviation(precipitation: pl.DataFrame, baseline: tuple[int, int]) -> pl.DataFrame:
    """
    Center each country's precipitation on its own mean over the baseline climatology period.

    Parameters
    ----------
    precipitation : DataFrame
        One row per country and year, carrying ``ISO``, ``year`` and ``precip``.
    baseline : tuple of int
        The first and last year of the reference period, both included.

    Returns
    -------
    deviation : DataFrame
        One row per country and year, carrying the deviation from that country's baseline mean.
    """
    first_year, last_year = baseline
    within_baseline = pl.col("year").dt.year().is_between(first_year, last_year)
    reference = precipitation.filter(within_baseline)

    span = last_year - first_year + 1
    covered = reference["year"].dt.year().n_unique()
    if covered < span:
        raise ValueError(
            f"The precipitation record covers {covered} of the {span} years in the "
            f"{first_year}-{last_year} baseline, so the climatology would be drawn from a shorter period "
            f"than the one it is named for."
        )

    climatology = reference.group_by("ISO").agg(pl.col("precip").mean().alias("baseline"))

    return precipitation.join(climatology, on="ISO", how="left").select(
        *PANEL_KEY, (pl.col("precip") - pl.col("baseline")).alias("precip_deviation")
    )


def _deviation_from_trend(climate: pl.DataFrame) -> pl.DataFrame:
    """Return the ocean temperature's residual around its STL trend. statsmodels fits pandas only."""
    observed = climate.drop_nulls("Temp").to_pandas().set_index("year")["Temp"]
    residual = observed - STL(observed, period=OCEAN_TREND_PERIOD).fit().trend

    converted: pl.DataFrame = pl.from_pandas(residual.rename("dev_from_trend_ocean_temp").reset_index())

    # The pandas round-trip widens the key to a datetime, which would not join back.
    return converted.with_columns(pl.col("year").cast(pl.Date))


def create_replication_data(cache_dir: Path, *, baseline: tuple[int, int] = CLIMATOLOGY_BASELINE) -> pl.DataFrame:
    """
    Assemble the country-year panel the paper's results are estimated from.

    Joins the disaster panel to the climate series and adds the detrended deviations, so the frame
    holds both the levels and the departures from trend the models use.

    Damage arrives in the unit EM-DAT publishes, thousands of US dollars, with each disaster class
    totalled on its own column and no combined total.

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under.
    baseline : tuple of int, optional
        First and last year of the climatology the deviations are measured against.

    Returns
    -------
    panel : DataFrame
        One row per country and year.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.replication_data import create_replication_data

        panel = create_replication_data(Path("data"))
    """
    panel = build_country_year_panel(cache_dir).rename({"Start_Year": "year"})

    # The first and last years are dropped. The reason is unrecorded, and the trend below is fitted
    # over this window, so widening it moves every published deviation.
    climate = build_time_series(cache_dir).select("year", "co2", "Temp", "precip").slice(1, -1)

    regressors = panel.select(
        *PANEL_KEY,
        _counted_or_missing(CLIMATOLOGICAL_TYPES).alias("climatological_disasters"),
        _counted_or_missing(HYDROLOGICAL_TYPES).alias("hydrological_disasters"),
        (pl.col("population") / MILLION).alias("population"),
        pl.col("population_density").log().alias("ln_population_density"),
        pl.col("gdp_per_cap_usd").log().alias("ln_gdp_pc"),
    ).with_columns(
        (pl.col("ln_gdp_pc") ** 2).alias("square_ln_gdp_pc"),
        (pl.col("ln_population_density") ** 2).alias("ln_population_density_squared"),
    )

    damages = panel.select(
        *PANEL_KEY, "Total_Damage_Adjusted_hydro", "Total_Damage_Adjusted_clim", "Total_Affected_hydro"
    )

    # Drawn from the whole precipitation record, which reaches back before the panel's first year
    # and so can cover the baseline climatology.
    deviation = _precipitation_deviation(annual_precipitation(cache_dir), baseline)

    frame = (
        regressors.join(damages, on=PANEL_KEY, how="left")
        .join(deviation, on=PANEL_KEY, how="left")
        .join(climate.select("year", "co2"), on="year", how="left")
        .join(_deviation_from_trend(climate), on="year", how="left")
    )

    return frame.select(
        *PUBLISHED_COLUMNS,
        "ln_population_density_squared",
        ((pl.col("year").dt.year() - TREND_BASE_YEAR) / 100).alias("time_period"),
    )


def model_frame(
    panel: pl.DataFrame,
    boundaries: gpd.GeoDataFrame,
    *,
    isos: Sequence[str] | None = None,
    features: Sequence[str] = MODEL_FEATURES,
) -> tuple[pl.DataFrame, gpd.GeoDataFrame]:
    """
    Reduce the panel and the boundaries to the rows a model reads and the countries both describe.

    A model draws its covariates from the panel and its spatial structure from the boundaries, so a
    country in one and not the other enters the fit as a hole. Both are returned, filtered together.

    Parameters
    ----------
    panel : DataFrame
        One row per country and year, keyed on ``ISO`` and ``year``, from
        :func:`create_replication_data`.
    boundaries : GeoDataFrame
        Country geometries carrying an ``ISO_A3`` column, from
        :func:`~climate_risk.data_functions.shapefiles_data_loader.load_shapefile`.
    isos : sequence of str, optional
        Restrict to these countries. Default None, meaning every country the two have in common.
    features : sequence of str, optional
        The columns a row must carry a value in to be kept. Default ``MODEL_FEATURES``.

    Returns
    -------
    rows : DataFrame
        The panel's complete rows for the retained countries, sorted by country and year.
    geometry : GeoDataFrame
        The boundaries of those same countries, in the same country order.
    """
    complete = panel.drop_nulls(list(features))
    paneled = set(complete["ISO"].unique())
    mapped = set(boundaries[ISO_COLUMN])

    # Logged before any place narrows the result, so this names what reconciliation cost rather than
    # every country the caller did not ask for.
    unreconciled = sorted(paneled ^ mapped)
    if unreconciled:
        _log.warning(f"Dropping {len(unreconciled)} countries only one side describes: {', '.join(unreconciled)}")

    described = paneled & mapped
    if isos is not None:
        absent = sorted(set(isos) - described)
        if absent:
            raise ValueError(f"{absent} carry no complete panel row, or no geometry, or neither of the two.")
        described = set(isos)

    if not described:
        raise ValueError("No country carries both a complete panel row and a geometry.")

    rows = complete.filter(pl.col("ISO").is_in(described)).sort(PANEL_KEY)
    geometry = boundaries[boundaries[ISO_COLUMN].isin(described)].sort_values(ISO_COLUMN)

    return rows, geometry


def load_model_frame(
    cache_dir: Path,
    *,
    place: Place | None = None,
    baseline: tuple[int, int] = CLIMATOLOGY_BASELINE,
    features: Sequence[str] = MODEL_FEATURES,
) -> tuple[pl.DataFrame, gpd.GeoDataFrame]:
    """
    Build the panel and the boundaries a model is estimated on, agreeing on their countries.

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under.
    place : CountryConfig or RegionConfig, optional
        Restrict to the countries this place covers. Default None, meaning every country available.
    baseline : tuple of int, optional
        First and last year of the climatology the deviations are measured against.
    features : sequence of str, optional
        The columns a row must carry a value in to be kept. Default ``MODEL_FEATURES``.

    Returns
    -------
    rows : DataFrame
        The panel's complete rows for the retained countries, sorted by country and year.
    geometry : GeoDataFrame
        The boundaries of those same countries, in the same country order.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.config.registry import load_place
        from climate_risk.replication_data import load_model_frame

        rows, geometry = load_model_frame(Path("data"), place=load_place("sea"))
    """
    return model_frame(
        create_replication_data(cache_dir, baseline=baseline),
        load_shapefile("world", cache_dir),
        isos=resolve_isos(place) if place is not None else None,
        features=features,
    )
