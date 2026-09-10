import logging

from pathlib import Path

import polars as pl

from climate_risk.data.cache import cached, polars_parquet
from climate_risk.data.co2 import load_co2_data
from climate_risk.data.source import VendoredSource

_log = logging.getLogger(__name__)

# SEDAC was decommissioned in June 2025 and no successor serves this workbook, so it ships with the
# package rather than being fetched.
IPCC = VendoredSource(
    filename="ipcc_ar6_syr_csb2_fig1a.xlsx",
    homepage="https://doi.org/10.7927/baxv-nj53",
    license="Attribution 4.0 International (CC BY 4.0)",
    citation=(
        "IPCC, 2024. IPCC AR6 Synthesis Report LR Cross-Section Box.2, Figure 1 (a). Palisades, "
        "New York: NASA Socioeconomic Data and Applications Center (SEDAC). "
        "https://doi.org/10.7927/baxv-nj53"
    ),
    retrieved="2026-09-09",
)

IPCC_SHEET = "CO2 Emissions"

SCENARIO_COLUMNS = {
    "Panel emissions - SSP1-19 - x (year)": "year",
    "Panel emissions - SSP1-19 - y": "SSP1-19",
    "Panel emissions - SSP1-26 - y": "SSP1-26",
    "Panel emissions - SSP2-45 - y": "SSP2-45",
    "Panel emissions - SSP3-70 - y": "SSP3-70",
    "Panel emissions - SSP5-85 - y": "SSP5-85",
}

# The projection starts here, taking the observed level, and accumulates changes from it.
ANCHOR_YEAR = 2020
LAST_PROJECTED_YEAR = 2100


def transform_ipcc(scenarios: pl.DataFrame, co2_observations: pl.DataFrame) -> pl.DataFrame:
    """
    Turn five-yearly emission changes into annual levels, anchored on observed CO2.

    Parameters
    ----------
    scenarios : DataFrame
        Published changes, with a ``year`` column and one column per SSP scenario.
    co2_observations : DataFrame
        Observed CO2, with a dated ``year`` column and a ``co2`` column.

    Returns
    -------
    projections : DataFrame
        One row per year from the anchor to ``LAST_PROJECTED_YEAR``, one column per scenario.
    """
    observed = co2_observations.select(pl.col("year").dt.year().alias("year"), "co2")
    scenario_names = [name for name in SCENARIO_COLUMNS.values() if name != "year"]

    anchor_level = pl.col("co2").filter(pl.col("year") == ANCHOR_YEAR).first()

    def level(name: str) -> pl.Expr:
        # Each published row is a change from the one before, so a level is the anchor plus every
        # change since it. Years at or before the anchor contribute nothing to that running total.
        accumulated = pl.when(pl.col("year") > ANCHOR_YEAR).then(pl.col(f"{name}_change")).otherwise(0.0).cum_sum()

        return (anchor_level + accumulated).alias(name)

    published = (
        scenarios.join(observed, on="year", how="left")
        .sort("year")
        .rename({name: f"{name}_change" for name in scenario_names})
    )
    levels = published.with_columns(level(name) for name in scenario_names)

    every_year = pl.DataFrame({"year": range(ANCHOR_YEAR, LAST_PROJECTED_YEAR + 1)}, schema={"year": pl.Int64})

    return (
        every_year.join(levels, on="year", how="left")
        .sort("year")
        .with_columns(pl.exclude("year").interpolate())
        .drop("co2")
    )


def process_ipcc_scenarios(cache_dir: Path, *, force_reload: bool = False) -> pl.DataFrame:
    """
    Build the IPCC AR6 emissions scenarios, anchored to the observed CO2 record.

    The scenario workbook ships with the package, so only the CO2 series it is anchored to is
    fetched.

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under.
    force_reload : bool, optional
        Download again and rebuild the cache rather than reading it. Default False.

    Returns
    -------
    scenarios : DataFrame
        One row per year from the anchor onward, one column per scenario.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk import process_ipcc_scenarios

        scenarios = process_ipcc_scenarios(Path("data"))
    """

    def build() -> pl.DataFrame:
        _log.info("Reading IPCC scenario emissions")
        published = pl.read_excel(IPCC.path(), sheet_name=IPCC_SHEET)
        scenarios = published.select(pl.col(code).alias(name) for code, name in SCENARIO_COLUMNS.items())

        return transform_ipcc(scenarios, load_co2_data(cache_dir).rename({"Date": "year"}).sort("year"))

    return cached(cache_dir, "ipcc_scenarios", build, polars_parquet(), force=force_reload)
