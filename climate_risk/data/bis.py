import logging

from pathlib import Path

import pandas as pd

from kuznets.bis import BISReader

from climate_risk.data.cache import cached, pandas_parquet
from climate_risk.data.source import ApiSource

_log = logging.getLogger(__name__)

BIS = ApiSource(
    url="https://stats.bis.org/api/v1",
    license="BIS terms of use for statistics, https://www.bis.org/about/legal/permitted-use-statistics.",
    citation="Bank for International Settlements, effective exchange rate indices, https://data.bis.org/.",
    retrieved="2026-11-19",
)

EFFECTIVE_EXCHANGE_RATES = "WS_EER"

# Earlier than the broad basket's start in 1994. Left out, kuznets asks for the last few years only.
FIRST_DATE = "1900-01-01"


def transform_effective_exchange_rate(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Index a monthly BIS effective exchange rate by month.

    Parameters
    ----------
    raw : DataFrame
        The single series kuznets returns for one country's selection, indexed by the first day of each month.

    Returns
    -------
    rate : DataFrame
        One column, ``real_effective_exchange_rate``, indexed by monthly period, with months the BIS leaves blank
        dropped.
    """
    if raw.shape[1] != 1:
        raise ValueError(f"Expected one BIS series for the selection, got {raw.shape[1]}: {list(raw.columns)}.")

    series = raw.iloc[:, 0].astype(float).dropna()

    return (
        series.set_axis(pd.DatetimeIndex(series.index).to_period("M")).rename("real_effective_exchange_rate").to_frame()
    )


def load_real_effective_exchange_rate(country: str, cache_dir: Path, *, force_reload: bool = False) -> pd.DataFrame:
    """
    Return a country's monthly broad real effective exchange rate from the BIS.

    Parameters
    ----------
    country : str
        ISO 3166-1 alpha-2 code, the BIS's ``REF_AREA``.
    cache_dir : Path
        Directory to cache the series in.
    force_reload : bool, optional
        Download again even when the series is cached. Default False.

    Returns
    -------
    rate : DataFrame
        The index, 2020 = 100, indexed by monthly period. A rise is a real appreciation.

    Examples
    --------
    India's series, from the start of the BIS's broad basket:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.bis import load_real_effective_exchange_rate

        rate = load_real_effective_exchange_rate("IN", Path("data"))
    """

    def build() -> pd.DataFrame:
        _log.info(f"Downloading the BIS real effective exchange rate for {country}")
        selections = {"FREQ": "M", "EER_TYPE": "R", "EER_BASKET": "B", "REF_AREA": country}

        reader = BISReader(dataflow=EFFECTIVE_EXCHANGE_RATES, selections=selections, start=FIRST_DATE)

        raw = reader.read()
        if not isinstance(raw, pd.DataFrame):
            raise TypeError(f"kuznets returned a {type(raw).__name__} for output_type='pandas'")

        return transform_effective_exchange_rate(raw)

    return cached(
        cache_dir,
        "bis_real_effective_exchange_rate",
        build,
        pandas_parquet(),
        params={"country": country},
        force=force_reload,
    )
