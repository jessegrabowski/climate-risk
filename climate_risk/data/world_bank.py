import logging

from collections.abc import Mapping
from pathlib import Path

import polars as pl

from kuznets import wb

from climate_risk.data.cache import builder_fingerprint, cached, polars_parquet
from climate_risk.data.source import ApiSource

_log = logging.getLogger(__name__)

WORLD_BANK = ApiSource(
    url="https://api.worldbank.org/v2/country",
    license="CC BY 4.0",
    citation=(
        "World Bank, World Development Indicators, https://databank.worldbank.org/source/world-development-indicators."
    ),
    retrieved="2026-09-05",
)

# Every country name the World Bank can return, its ISO code, and whether we ask for it. The rows
# it does not request are the Bank's regional and income aggregates, which are not countries.
COUNTRIES_FILE = Path(__file__).parent / "world_bank_countries.csv"


def _read_countries() -> tuple[dict[str, str], list[str]]:
    """Return the name-to-code mapping and the codes to download, read from ``COUNTRIES_FILE``."""
    table = pl.read_csv(COUNTRIES_FILE)

    return (
        dict(zip(table["country"], table["country_code"], strict=True)),
        table.filter("requested")["country_code"].to_list(),
    )


COUNTRY_CODE_BY_NAME, REQUESTED_COUNTRY_CODES = _read_countries()

# One panel, because these are one country-year grid from one API. A column name carries the price
# base wherever the same quantity is published on more than one: KD is constant 2015 US dollars and
# KN is constant local currency, so the ratios the model forms within a country share one unit.
INDICATOR_NAMES = {
    "EN.POP.DNST": "population_density",
    "NY.GDP.PCAP.KD": "gdp_per_cap_usd",
    "SP.POP.TOTL": "population",
    "NY.GDP.MKTP.KD": "real_gdp_usd",
    "AG.SRF.TOTL.K2": "surface_area_km2",
    # The national accounts, prices, rates and fiscal aggregates the small open economy model is
    # estimated on.
    "NY.GDP.MKTP.KN": "real_gdp_lcu",
    "NE.CON.PRVT.KN": "real_consumption_lcu",
    "NE.GDI.FTOT.KN": "real_investment_lcu",
    "NE.CON.GOVT.KN": "real_government_lcu",
    "NE.EXP.GNFS.KN": "real_exports_lcu",
    "NE.IMP.GNFS.KN": "real_imports_lcu",
    # Paired with the constant-price series above, these give consumption and investment deflators,
    # which is what carries import prices when no import price index is available.
    "NE.CON.PRVT.CN": "nominal_consumption",
    "NE.GDI.FTOT.CN": "nominal_investment",
    "FP.CPI.TOTL": "cpi",
    "NY.GDP.DEFL.ZS": "gdp_deflator",
    "PA.NUS.FCRF": "exchange_rate",
    "FR.INR.LEND": "lending_rate",
    "FR.INR.DPST": "deposit_rate",
    "TM.TAX.MRCH.WM.AR.ZS": "import_tariff",
    "DT.DOD.DECT.CD": "external_debt",
    "BN.CAB.XOKA.GD.ZS": "current_account_gdp",
    "SL.EMP.TOTL.SP.ZS": "employment_rate",
    "GC.TAX.TOTL.GD.ZS": "tax_revenue_gdp",
    "GC.XPN.TOTL.GD.ZS": "government_expense_gdp",
}

WB_INDICATORS = list(INDICATOR_NAMES)

# Earlier than any indicator's coverage, so the series starts wherever the data does.
FIRST_YEAR = 1900


def transform_world_bank(raw: pl.DataFrame, indicator_names: Mapping[str, str]) -> pl.DataFrame:
    """
    Key the World Bank indicators by ISO code and year, under their readable names.

    Parameters
    ----------
    raw : DataFrame
        Indicators as ``kuznets`` returns them tidy, with ``country`` and ``year`` columns.
    indicator_names : mapping of str to str
        The indicator codes to keep, each mapped to the name it is stored under.

    Returns
    -------
    indicators : DataFrame
        One row per country and year, sorted, with an integer ``year``.
    """
    # kuznets warns rather than raising when the Bank rejects a code, and omits the column entirely.
    missing = sorted(set(indicator_names) - set(raw.columns))
    if missing:
        raise ValueError(f"The World Bank returned no column for {missing}; the codes may have been retired.")

    coded = raw.with_columns(pl.col("country").replace_strict(COUNTRY_CODE_BY_NAME, default=None).alias("country_code"))

    unmatched = sorted(set(coded.filter(pl.col("country_code").is_null())["country"].to_list()))
    if unmatched:
        _log.warning(f"Dropping {len(unmatched)} countries with no ISO code: {', '.join(unmatched)}")

    return (
        coded.drop_nulls("country_code")
        .select(
            "country_code",
            # Upstream dates the year. dt.year() narrows it to Int32, and a join key that changes
            # width is a join that stops matching.
            pl.col("year").dt.year().cast(pl.Int64),
            *(pl.col(code).alias(name) for code, name in indicator_names.items()),
        )
        .sort("country_code", "year")
    )


def load_wb_data(cache_dir: Path, *, force_reload: bool = False) -> pl.DataFrame:
    """
    Return the World Bank panel, one row per country and year.

    Every indicator lands in one frame because they are one country-year grid from one API. The
    entry is keyed on how it was built as well as on its name, so editing ``INDICATOR_NAMES`` turns it over instead of reading back what an earlier set produced.

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under.
    force_reload : bool, optional
        Download again and rebuild the cache rather than reading it. Default False.

    Returns
    -------
    indicators : DataFrame
        One row per country and year, one column per indicator under its readable name.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk import load_wb_data

        indicators = load_wb_data(Path("data"))
    """

    def build() -> pl.DataFrame:
        _log.info(f"Downloading {len(INDICATOR_NAMES)} World Bank indicators")
        downloaded = wb.download(
            indicator=WB_INDICATORS,
            country=REQUESTED_COUNTRY_CODES,
            start=FIRST_YEAR,
            end=None,
            output_type="polars",
        )
        if not isinstance(downloaded, pl.DataFrame):
            raise TypeError(f"kuznets returned a {type(downloaded).__name__} for output_type='polars'")

        return transform_world_bank(downloaded, INDICATOR_NAMES)

    reading = builder_fingerprint(build, INDICATOR_NAMES)

    return cached(cache_dir, "world_bank", build, polars_parquet(), params={"reading": reading}, force=force_reload)
