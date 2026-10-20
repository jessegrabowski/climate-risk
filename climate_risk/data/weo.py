import logging

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from kuznets.imf import IMFReader

from climate_risk.data.cache import builder_fingerprint, cached, polars_parquet
from climate_risk.data.source import ApiSource

_log = logging.getLogger(__name__)

WEO = ApiSource(
    url="https://api.imf.org/external/sdmx/2.1/data/IMF.RES,WEO",
    license="© International Monetary Fund. All rights reserved. https://www.imf.org/external/terms.htm",
    citation="International Monetary Fund, World Economic Outlook database, https://data.imf.org/en/datasets/IMF.RES:WEO.",
    retrieved="2026-10-17",
)

# General government gross debt as a percentage of GDP.
GROSS_DEBT_INDICATOR = "GGXWDG_NGDP"

PERCENT = 100.0

# Earlier than the database's own coverage, so the series starts wherever the WEO's does.
FIRST_YEAR = 1900


def transform_government_debt(observations: pl.DataFrame) -> pl.DataFrame:
    """
    Key general government gross debt by ISO code and year, as a fraction of GDP.

    Parameters
    ----------
    observations : DataFrame
        The WEO indicator as ``kuznets`` returns it, with ``COUNTRY``, ``period`` and ``value`` columns.

    Returns
    -------
    debt : DataFrame
        One row per country and year, sorted, carrying ``government_debt`` as a fraction of GDP.

    Examples
    --------
    .. code-block:: python

        from climate_risk.data.weo import transform_government_debt

        debt = transform_government_debt(observations)
    """
    missing = sorted({"COUNTRY", "period", "value"} - set(observations.columns))
    if missing:
        raise ValueError(f"The WEO dataflow carried no {missing} column; its dimensions may have been renamed.")

    return (
        observations.select(
            pl.col("COUNTRY").alias("country_code"),
            pl.col("period").dt.year().cast(pl.Int64).alias("year"),
            (pl.col("value") / PERCENT).alias("government_debt"),
        )
        .drop_nulls("government_debt")
        .sort("country_code", "year")
    )


def load_government_debt(
    cache_dir: Path,
    countries: Sequence[str],
    *,
    force_reload: bool = False,
) -> pl.DataFrame:
    """
    Return IMF World Economic Outlook general government gross debt, one row per country and year.

    The WEO reports the most recent years as projections and does not flag them in the series, and
    countries on fiscal years, India among them, are reported by the calendar year a fiscal year
    starts in.

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under.
    countries : sequence of str
        ISO 3166-1 alpha-3 codes to ask for.
    force_reload : bool, optional
        Download again and rebuild the cache rather than reading it. Default False.

    Returns
    -------
    debt : DataFrame
        ``country_code``, ``year`` and ``government_debt``, the last a fraction of GDP.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.weo import load_government_debt

        debt = load_government_debt(Path("data"), ["IND"])
    """
    codes = sorted(countries)

    def build() -> pl.DataFrame:
        _log.info(f"Downloading WEO general government gross debt for {', '.join(codes)}")
        downloaded = IMFReader(
            "WEO",
            {"COUNTRY": codes, "INDICATOR": GROSS_DEBT_INDICATOR},
            start=FIRST_YEAR,
            end=None,
            output_type="polars",
        ).read()
        if not isinstance(downloaded, pl.DataFrame):
            raise TypeError(f"kuznets returned a {type(downloaded).__name__} for output_type='polars'")

        shaped = transform_government_debt(downloaded)

        silent = sorted(set(codes) - set(shaped["country_code"]))
        if silent:
            _log.warning(f"The WEO answered for {len(codes) - len(silent)} countries and not for {', '.join(silent)}")

        return shaped

    reading = builder_fingerprint(build, transform_government_debt, GROSS_DEBT_INDICATOR, FIRST_YEAR)

    return cached(
        cache_dir,
        "weo_government_debt",
        build,
        polars_parquet(),
        params={"countries": "-".join(codes), "reading": reading},
        force=force_reload,
    )
