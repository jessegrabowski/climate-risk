"""
Country-year covariates for how likely a disaster is to be reported: internet use and mobile subscriptions.

Both come from the World Bank's World Development Indicators, through the same client and country table as
:mod:`climate_risk.data.world_bank`, kept apart so the published panel's cache is not touched.
"""

from pathlib import Path

import polars as pl

from kuznets import wb

from climate_risk.data.cache import builder_fingerprint, cached, polars_parquet
from climate_risk.data.world_bank import FIRST_YEAR, REQUESTED_COUNTRY_CODES, transform_world_bank

# IT.NET.USER.ZS: individuals using the internet (% of population).
# IT.CEL.SETS.P2: mobile cellular subscriptions per 100 people.
REPORTING_INDICATORS = {"IT.NET.USER.ZS": "internet_users_pct", "IT.CEL.SETS.P2": "mobile_per_100"}

EARLY_YEAR = 1995


def zero_before_first_observation(indicators: pl.DataFrame, *, early_year: int = EARLY_YEAR) -> pl.DataFrame:
    """
    Set an indicator to 0 in the years before ``early_year`` that precede a country's first observation.

    Neither technology existed to any measurable degree then, so a blank means none. A blank from ``early_year`` on,
    or one after the first observation, stays missing.

    Parameters
    ----------
    indicators : DataFrame
        ``country_code``, ``year`` and one column per indicator.
    early_year : int, optional
        Blanks before the first observation are filled only for years before this one. Default 1995.

    Returns
    -------
    filled : DataFrame
        The same rows with those blanks set to 0.
    """
    out = []
    for column in REPORTING_INDICATORS.values():
        first = pl.col("year").filter(pl.col(column).is_not_null()).min().over("country_code")
        blank_early = (
            pl.col(column).is_null() & (pl.col("year") < early_year) & (first.is_null() | (pl.col("year") < first))
        )
        out.append(pl.when(blank_early).then(0.0).otherwise(pl.col(column)).alias(column))

    return indicators.with_columns(out)


def load_reporting_indicators(cache_dir: Path, *, force_reload: bool = False) -> pl.DataFrame:
    """
    Load internet use and mobile subscriptions, one row per country and year, as published (blanks kept).

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under.
    force_reload : bool, optional
        Download again. Default False.

    Returns
    -------
    indicators : DataFrame
        ``country_code``, ``year``, ``internet_users_pct``, ``mobile_per_100``.
    """

    def build() -> pl.DataFrame:
        downloaded = wb.download(
            indicator=list(REPORTING_INDICATORS),
            country=REQUESTED_COUNTRY_CODES,
            start=FIRST_YEAR,
            end=None,
            output_type="polars",
        )
        if not isinstance(downloaded, pl.DataFrame):
            raise TypeError(f"kuznets returned a {type(downloaded).__name__} for output_type='polars'")

        return transform_world_bank(downloaded, REPORTING_INDICATORS)

    reading = builder_fingerprint(build, REPORTING_INDICATORS, REQUESTED_COUNTRY_CODES)

    return cached(
        cache_dir, "world_bank_reporting", build, polars_parquet(), params={"reading": reading}, force=force_reload
    )
