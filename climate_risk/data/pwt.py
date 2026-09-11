import logging

from pathlib import Path

import polars as pl

from climate_risk.data.cache import cached, polars_parquet
from climate_risk.data.source import ManualSource

_log = logging.getLogger(__name__)

PWT = ManualSource(
    filename="pwt100.xlsx",
    homepage="https://www.rug.nl/ggdc/productivity/pwt/",
    license="Creative Commons Attribution 4.0",
    citation=(
        "Feenstra, Robert C., Robert Inklaar and Marcel P. Timmer (2015), "
        "'The Next Generation of the Penn World Table', American Economic Review 105(10), 3150-3182"
    ),
    retrieved="2026-08-13",
)

SHEET = "Data"

# The `na` family is comparable over time within a country; `ctfp` is a level at current PPPs with
# the USA at 1, and is the only cross-country comparison the table supports.
PWT_COLUMNS = {
    "rgdpna": "pwt_real_gdp",
    "rkna": "capital",
    "emp": "employment",
    "pop": "pwt_population",
    "labsh": "labour_share",
    "delta": "depreciation",
    "ctfp": "tfp_relative_to_usa",
}


def transform_pwt(raw: pl.DataFrame) -> pl.DataFrame:
    """
    Key the Penn World Table national-accounts series by ISO code and year.

    ``rgdpna`` and ``rkna`` are indices on separate bases, so their levels are not comparable to each
    other or to any other source. Their ratio and their growth rates are.

    Parameters
    ----------
    raw : DataFrame
        The ``Data`` sheet of the workbook, carrying ``countrycode`` and ``year``.

    Returns
    -------
    pwt : DataFrame
        One row per country and year, sorted, with an integer ``year``.
    """
    missing = sorted(set(PWT_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"The `{SHEET}` sheet is missing {missing}; the release may have renamed them.")

    return (
        raw.select(
            pl.col("countrycode").alias("country_code"),
            pl.col("year").cast(pl.Int64),
            *(pl.col(code).cast(pl.Float64).alias(name) for code, name in PWT_COLUMNS.items()),
        )
        .drop_nulls("country_code")
        .sort("country_code", "year")
    )


def load_pwt_data(cache_dir: Path, *, force_reload: bool = False) -> pl.DataFrame:
    """
    Return the Penn World Table national-accounts panel, reading the workbook on a cache miss.

    Parameters
    ----------
    cache_dir : Path
        Directory holding the workbook and the cached parquet.
    force_reload : bool, optional
        Re-read the workbook even when the parquet is present. Default False.

    Returns
    -------
    pwt : DataFrame
        One row per country and year.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.pwt import load_pwt_data

        pwt = load_pwt_data(Path("data"))
    """

    def build() -> pl.DataFrame:
        path = PWT.require(cache_dir)
        _log.info(f"Reading Penn World Table from {path}")

        # Two columns elsewhere in the sheet have no inferable dtype, so name the ones we read.
        workbook = pl.read_excel(path, sheet_name=SHEET, columns=["countrycode", "year", *PWT_COLUMNS])

        return transform_pwt(workbook)

    return cached(cache_dir, "penn_world_table", build, polars_parquet(), force=force_reload)
