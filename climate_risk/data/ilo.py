import logging

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from kuznets.ilostat import ILOSTATReader

from climate_risk.data.cache import builder_fingerprint, cached, polars_parquet
from climate_risk.data.source import ApiSource
from climate_risk.data.world_bank import REQUESTED_COUNTRY_CODES

_log = logging.getLogger(__name__)

ILO = ApiSource(
    url="https://sdmx.ilo.org/rest",
    license="ILO terms of use, https://www.ilo.org/disclaimer-privacy-policy",
    citation="International Labour Organization, ILOSTAT database, https://ilostat.ilo.org.",
    retrieved="2026-09-12",
)

# Labour income as a share of GDP, which the ILO estimates from harmonized survey microdata.
LABOUR_INCOME_SHARE = "DF_LAP_2GDP_NOC_RT"

# ILO's area codelist carries no entry for these, and one unknown code fails the whole request. They
# are areas the World Bank reports on and the ILO does not.
UNCOVERED_AREAS = frozenset({"MAF", "SXM", "XKX"})

# The dataflow reports a share of GDP as a percentage.
PERCENT = 100.0

# No country's labour share approaches this, and a series already expressed as a fraction could not
# exceed it. Dividing such a series by PERCENT would leave a hundredth of itself, still inside the
# nought-to-one band the column is documented to hold.
LOWEST_CREDIBLE_PERCENT = 1.0

# Earlier than the estimate's own coverage, so the series starts wherever the ILO's does. Left
# unstated, the service answers with a handful of recent years rather than everything it holds.
FIRST_YEAR = 1900


def transform_labour_share(observations: pl.DataFrame) -> pl.DataFrame:
    """
    Key the ILO labour income share by ISO code and year, as a fraction of GDP.

    Parameters
    ----------
    observations : DataFrame
        The dataflow as ``kuznets`` returns it, with ``REF_AREA``, ``period`` and ``value`` columns.

    Returns
    -------
    labour_share : DataFrame
        One row per country and year, sorted, carrying ``labour_income_share`` between 0 and 1.

    Examples
    --------
    .. code-block:: python

        from climate_risk.data.ilo import transform_labour_share

        labour_share = transform_labour_share(observations)
    """
    missing = sorted({"REF_AREA", "period", "value"} - set(observations.columns))
    if missing:
        raise ValueError(f"The ILO dataflow carried no {missing} column; its dimensions may have been renamed.")

    reported = observations["value"].drop_nulls().to_list()
    if reported:
        already_a_fraction = max(reported) <= LOWEST_CREDIBLE_PERCENT
        outside_a_share = min(reported) < 0 or max(reported) > PERCENT
        if already_a_fraction or outside_a_share:
            raise ValueError(
                f"The ILO labour share runs {min(reported)} to {max(reported)}, which is not a percentage of GDP."
            )

    return (
        observations.select(
            pl.col("REF_AREA").alias("country_code"),
            pl.col("period").dt.year().cast(pl.Int64).alias("year"),
            (pl.col("value") / PERCENT).alias("labour_income_share"),
        )
        .drop_nulls("labour_income_share")
        .sort("country_code", "year")
    )


def load_labour_income_share(
    cache_dir: Path,
    countries: Sequence[str] | None = None,
    *,
    force_reload: bool = False,
) -> pl.DataFrame:
    """
    Return the ILO labour income share, one row per country and year.

    This is not interchangeable with :func:`~climate_risk.data.pwt.load_pwt_data`'s ``labour_share``,
    which estimates the same concept from national accounts and does not agree with it.

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under.
    countries : sequence of str, optional
        ISO 3166-1 alpha-3 codes to ask for. Default None, meaning the countries the World Bank panel
        requests.
    force_reload : bool, optional
        Download again and rebuild the cache rather than reading it. Default False.

    Returns
    -------
    labour_share : DataFrame
        ``country_code``, ``year`` and ``labour_income_share``, the last a fraction of GDP.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.ilo import load_labour_income_share

        labour_share = load_labour_income_share(Path("data"))
    """
    wanted = REQUESTED_COUNTRY_CODES if countries is None else countries
    asked = [code for code in wanted if code not in UNCOVERED_AREAS]

    def build() -> pl.DataFrame:
        _log.info(f"Downloading the ILO labour income share for {len(asked)} countries")
        downloaded = ILOSTATReader(
            LABOUR_INCOME_SHARE, {"REF_AREA": asked}, start=FIRST_YEAR, end=None, output_type="polars"
        ).read()
        if not isinstance(downloaded, pl.DataFrame):
            raise TypeError(f"kuznets returned a {type(downloaded).__name__} for output_type='polars'")

        shaped = transform_labour_share(downloaded)

        silent = sorted(set(asked) - set(shaped["country_code"]))
        if silent:
            _log.warning(f"The ILO answered for {len(asked) - len(silent)} countries and not for {', '.join(silent)}")

        return shaped

    reading = builder_fingerprint(build, transform_labour_share, LABOUR_INCOME_SHARE, FIRST_YEAR, asked)

    return cached(
        cache_dir, "ilo_labour_share", build, polars_parquet(), params={"reading": reading}, force=force_reload
    )
