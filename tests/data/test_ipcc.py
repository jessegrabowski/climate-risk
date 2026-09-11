import re

from datetime import date

import polars as pl
import pytest

from climate_risk.data.ipcc import IPCC, IPCC_SHEET, SCENARIO_COLUMNS, transform_ipcc

# The published series runs five-yearly and the projection ends in 2100. Stated literally, not
# imported, so a changed constant fails instead of moving the expectation with it.
STEP_YEARS = 5
FIRST_PROJECTED_YEAR = 2020
LAST_PROJECTED_YEAR = 2100

SCENARIOS = ["SSP1-19", "SSP1-26", "SSP2-45", "SSP3-70", "SSP5-85"]
PUBLISHED_YEARS = list(range(2015, 2106, STEP_YEARS))


def level(levels: pl.DataFrame, year: int, scenario: str) -> float | None:
    """The one projected level for a year and scenario."""
    match = levels.filter(pl.col("year") == year)

    assert len(match) == 1, f"expected one row for {year}, got {len(match)}"
    value: float | None = match[scenario].item()

    return value


@pytest.fixture
def scenarios():
    """One GtCO2 per year in every step, so each step emits five and the rise is one arithmetic."""
    return pl.DataFrame({"year": PUBLISHED_YEARS} | {name: [1.0] * len(PUBLISHED_YEARS) for name in SCENARIOS})


@pytest.fixture
def co2_observations():
    return pl.DataFrame({"year": [date(year, 1, 1) for year in PUBLISHED_YEARS], "co2": [400.0] * len(PUBLISHED_YEARS)})


def test_the_anchor_years_take_the_observed_level(scenarios, co2_observations):
    """The published series holds emissions, so the concentrations have to start somewhere real."""
    levels = transform_ipcc(scenarios, co2_observations)

    assert level(levels, FIRST_PROJECTED_YEAR, "SSP1-19") == 400.0


def test_a_published_rate_accumulates_as_a_concentration(scenarios, co2_observations):
    """The workbook publishes GtCO2 per year, so a step emits five of them. At a 0.45 airborne
    fraction that is 2.25 GtCO2 reaching the atmosphere, and 7.81 GtCO2 is one ppm."""
    levels = transform_ipcc(scenarios, co2_observations)

    assert level(levels, 2025, "SSP1-19") == pytest.approx(400.288092, abs=1e-6)
    assert level(levels, 2030, "SSP1-19") == pytest.approx(400.576184, abs=1e-6)


def test_the_five_yearly_series_becomes_annual(scenarios, co2_observations):
    levels = transform_ipcc(scenarios, co2_observations)

    assert levels["year"].to_list() == list(range(FIRST_PROJECTED_YEAR, LAST_PROJECTED_YEAR + 1))


def test_the_years_between_steps_are_interpolated(scenarios, co2_observations):
    """A published point every five years, so the intervening levels are straight lines."""
    levels = transform_ipcc(scenarios, co2_observations)

    assert level(levels, 2027, "SSP1-19") == pytest.approx(400.403329, abs=1e-6)


def test_the_observed_column_does_not_survive(scenarios, co2_observations):
    """`co2` is the anchor, not a scenario; leaving it in would read as a sixth pathway."""
    levels = transform_ipcc(scenarios, co2_observations)

    assert "co2" not in levels.columns


def test_the_year_is_a_column_the_cache_can_carry(scenarios, co2_observations):
    """Parquet stores columns; a year that only existed as an index would not survive the round trip."""
    levels = transform_ipcc(scenarios, co2_observations)

    assert levels.columns[0] == "year"


def test_every_scenario_is_carried_through(scenarios, co2_observations):
    levels = transform_ipcc(scenarios, co2_observations)

    assert set(SCENARIOS) <= set(levels.columns)


def test_the_published_rate_survives_beside_the_concentration(scenarios, co2_observations):
    """A caller wanting emissions rather than concentration should not have to undo the conversion."""
    levels = transform_ipcc(scenarios, co2_observations)

    assert {f"{name}_emissions" for name in SCENARIOS} <= set(levels.columns)
    assert levels["SSP1-19_emissions"].to_list() == [1.0] * len(levels)


def test_a_missing_observation_leaves_the_level_missing(scenarios, co2_observations):
    """Anchoring on a year the observations do not cover must not invent a level."""
    without_anchor = co2_observations.filter(pl.col("year") != date(FIRST_PROJECTED_YEAR, 1, 1))

    levels = transform_ipcc(scenarios, without_anchor)

    assert level(levels, FIRST_PROJECTED_YEAR, "SSP1-19") is None


def test_the_vendored_workbook_carries_what_the_loader_reads():
    """The workbook ships with the package, so replacing it must not quietly change what is read."""
    published = pl.read_excel(IPCC.path(), sheet_name=IPCC_SHEET)

    assert set(SCENARIO_COLUMNS) <= set(published.columns)
    assert published["Panel emissions - SSP1-19 - x (year)"].to_list() == list(range(2015, 2101, STEP_YEARS))


def test_the_vendored_workbook_keeps_its_attribution():
    """It is redistributed under CC BY, which is only satisfied while the credit ships beside it."""
    attribution = (IPCC.path().parent / "ATTRIBUTION.md").read_text()

    assert IPCC.filename in attribution
    assert "CC BY 4.0" in attribution


def test_the_declaration_agrees_with_the_attribution_notice():
    """Terms are now stated twice, and the notice beside the file is the one a redistributor reads."""
    attribution = (IPCC.path().parent / "ATTRIBUTION.md").read_text()
    stated = re.search(r"\*\*License\.\*\* (.+?),", attribution, re.DOTALL)

    assert stated is not None, "ATTRIBUTION.md states no license"
    assert " ".join(stated.group(1).split()) == IPCC.license
    assert IPCC.homepage in attribution
