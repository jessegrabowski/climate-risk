import logging

from pathlib import Path

import polars as pl

from climate_risk.data.cache import builder_fingerprint, cached, polars_parquet
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

# The projection starts here, taking the observed level, and accumulates emissions from it.
ANCHOR_YEAR = 2020
LAST_PROJECTED_YEAR = 2100

# The workbook's Metadata sheet gives the unit as GtCO2/year, sampled every five years, so a
# published value is a rate and covers the step that follows it.
PUBLISHED_STEP_YEARS = 5

# One ppm of atmospheric CO2 is 2.13 GtC, and CO2 masses 44/12 times its carbon.
GTCO2_PER_PPM = 7.81

# The share of emitted CO2 that stays in the atmosphere rather than entering a sink.
AIRBORNE_FRACTION = 0.45


def transform_ipcc(scenarios: pl.DataFrame, co2_observations: pl.DataFrame) -> pl.DataFrame:
    """
    Turn five-yearly emission rates into annual concentrations, anchored on observed CO2.

    A published rate covers the step that follows it, so it becomes a quantity of CO2, of which
    ``AIRBORNE_FRACTION`` reaches the atmosphere at ``GTCO2_PER_PPM``. Holding the airborne fraction
    constant is an approximation: the real one rises with cumulative emissions, so the highest
    pathways come out low against the concentrations AR6 reports.

    Parameters
    ----------
    scenarios : DataFrame
        Published emissions in GtCO2 per year, with a ``year`` column and one column per SSP
        scenario.
    co2_observations : DataFrame
        Observed CO2 in ppm, with a dated ``year`` column and a ``co2`` column.

    Returns
    -------
    projections : DataFrame
        One row per year from the anchor to ``LAST_PROJECTED_YEAR``, carrying the published rate per
        scenario as ``<scenario>_emissions`` and the concentration it accumulates to as
        ``<scenario>``.
    """
    observed = co2_observations.select(pl.col("year").dt.year().alias("year"), "co2")
    scenario_names = [name for name in SCENARIO_COLUMNS.values() if name != "year"]

    anchor_level = pl.col("co2").filter(pl.col("year") == ANCHOR_YEAR).first()

    def level(name: str) -> pl.Expr:
        # Years at or before the anchor contribute nothing to the running total.
        rate = pl.when(pl.col("year") > ANCHOR_YEAR).then(pl.col(f"{name}_emissions")).otherwise(0.0)
        emitted = (rate * PUBLISHED_STEP_YEARS).cum_sum()

        return (anchor_level + emitted * AIRBORNE_FRACTION / GTCO2_PER_PPM).alias(name)

    published = (
        scenarios.join(observed, on="year", how="left")
        .sort("year")
        .rename({name: f"{name}_emissions" for name in scenario_names})
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
        One row per year from the anchor onward, with the published emission rate and the
        concentration it accumulates to per scenario.

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

    reading = builder_fingerprint(
        build,
        transform_ipcc,
        ANCHOR_YEAR,
        PUBLISHED_STEP_YEARS,
        GTCO2_PER_PPM,
        AIRBORNE_FRACTION,
    )

    return cached(cache_dir, "ipcc_scenarios", build, polars_parquet(), params={"reading": reading}, force=force_reload)
