"""
A market-dollar capital stock series, the denominator :func:`climate_risk.dsge.scenarios.damage_fraction` needs.

Damage draws are in nominal US dollars at market exchange rates, so the capital they destroy has to be
too. No single source gives that. The Penn World Table's capital stock (``rnna``) is at constant 2017
*national* prices and its GDP (``rgdpna``) in the same units, so their ratio is a unit-free
capital-output ratio. World Bank real GDP (``NY.GDP.MKTP.KD``) is in constant 2015 US dollars at market
rates. Their product is capital in 2015 market dollars:

``capital_usd[t] = capital_output[t] * gdp_usd[t] * price_factor``

Notes
-----
The repo's :func:`~climate_risk.data.pwt.load_pwt_data` ``capital`` column is ``rkna``, an index with
2017 equal to 1, so it cannot give a ratio of levels and is not used here.
"""

import logging

from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr

from climate_risk.data.pwt import PWT, SHEET

_log = logging.getLogger(__name__)

PWT_CAPITAL = "rnna"
PWT_GDP = "rgdpna"


def load_capital_output(cache_dir: Path, iso: str) -> pl.DataFrame:
    """
    Read one country's capital-output ratio from the Penn World Table workbook.

    Parameters
    ----------
    cache_dir : Path
        Directory holding the PWT workbook.
    iso : str
        ISO alpha-3 code.

    Returns
    -------
    DataFrame
        Columns ``year`` and ``capital_output``, the capital stock over real GDP, both at constant
        national prices, so unit-free.
    """
    path = PWT.require(cache_dir)
    raw = pl.read_excel(path, sheet_name=SHEET, columns=["countrycode", "year", PWT_CAPITAL, PWT_GDP])
    ratio = (
        raw.filter(pl.col("countrycode") == iso)
        .select(pl.col("year").cast(pl.Int64), (pl.col(PWT_CAPITAL) / pl.col(PWT_GDP)).alias("capital_output"))
        .drop_nulls()
        .sort("year")
    )
    if ratio.is_empty():
        raise ValueError(f"The Penn World Table has no capital-output ratio for {iso}.")
    return ratio


def capital_stock_usd(
    gdp_usd: xr.DataArray,
    capital_output: pl.DataFrame,
    *,
    price_factor: float = 1.0,
) -> xr.DataArray:
    """
    Capital stock in market US dollars from GDP and a capital-output ratio.

    Parameters
    ----------
    gdp_usd : DataArray
        Real GDP in market US dollars, indexed by ``year``. It may run past the Penn World Table's
        last year, and into the future, if it is a forecast.
    capital_output : DataFrame
        ``year`` and ``capital_output`` from :func:`load_capital_output`. Years after its last one
        use that last ratio, held flat; years before its first use the first.
    price_factor : float
        Multiplier restating ``gdp_usd``'s dollar-year into the dollar-year of the damage draws. The
        World Bank series is constant 2015 dollars; EM-DAT's adjusted damages are stated in a later
        year's dollars, so a US price-level ratio between the two belongs here. The default of 1
        leaves them unrestated, which understates capital and overstates the destroyed fraction.

    Returns
    -------
    DataArray
        Capital in market dollars, indexed by ``year``, for :func:`damage_fraction`.
    """
    if price_factor <= 0:
        raise ValueError(f"price_factor must be positive, got {price_factor}")
    years = capital_output["year"].to_numpy()
    ratio = capital_output["capital_output"].to_numpy()
    gdp_years = gdp_usd["year"].to_numpy()
    if gdp_years.max() > years.max():
        _log.warning(f"Holding the capital-output ratio at its {years.max()} value for later years.")
    held = np.interp(gdp_years, years, ratio)  # np.interp holds the end values flat outside the range
    capital = xr.DataArray(held, dims="year", coords={"year": gdp_years}) * gdp_usd * price_factor
    return capital.rename("capital_usd")
