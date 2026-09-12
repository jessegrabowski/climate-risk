import logging

from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pytest

from shapely.geometry import Point

from climate_risk.replication_data import create_replication_data, model_frame
from tests.conftest import (
    GPCC_CACHE_FILE,
    emdat_event,
    seed_ocean_heat_cache,
    seed_world_bank_cache,
    write_emdat_workbook,
)

# Long enough that `iloc[1:-1]` still leaves an STL-able series: STL(period=3) needs 2*3+1 points.
YEARS = tuple(range(1985, 2021))

# Precipitation reaches back before the panel does, as the real record does, and opens well before
# the baseline so that the reference period is neither the start of the record nor the whole of it.
PRECIPITATION_YEARS = tuple(range(1955, 2021))

# AAA and BBB appear everywhere. CCC is EM-DAT only and DDD World Bank only, so reconciliation has
# something to drop from each side. EEE has both but no precipitation; FFF has only precipitation.
EMDAT_COUNTRIES = ("AAA", "BBB", "CCC", "EEE")
WORLD_BANK_COUNTRIES = ("AAA", "BBB", "DDD", "EEE")
PRECIPITATION_COUNTRIES = ("AAA", "BBB", "FFF")

# Stated literally rather than imported, so a changed constant fails instead of moving with it.
TREND_BASE_YEAR = 1980

# What tests/conftest.py writes into every event's "Total Damage, Adjusted ('000 US$)" cell. AAA has
# one flood and one drought a year, so each class totals one event's figure rather than two.
DAMAGE_THOUSANDS = 120

# The second year of the span, so the expected values below are the fixture's formulas at i = 1.
SAMPLE_YEAR = YEARS[1]

# The panel grid reaches back before the first event, so these carry no disasters at all.
QUIET_YEAR = 1975


@pytest.fixture(scope="module")
def wide_cache(tmp_path_factory):
    """A cache spanning enough years for the trend fit and the climatology to do anything.

    AAA gets a drought on top of its flood, so climatological and hydrological damage are both
    present for one country and only hydrological for the others.
    """
    tmp_path = tmp_path_factory.mktemp("replication")
    events = [
        emdat_event({"ISO": iso, "DisNo.": f"{iso}-{year}", "Start Year": year, "End Year": year})
        for iso in EMDAT_COUNTRIES
        for year in YEARS
    ]
    events += [
        emdat_event(
            {
                "ISO": "AAA",
                "DisNo.": f"AAA-drought-{year}",
                "Start Year": year,
                "End Year": year,
                "Disaster Type": "Drought",
            }
        )
        for year in YEARS
    ]
    write_emdat_workbook(tmp_path, events)

    world_bank = [
        (iso, year, 1000 + 100 * n + 10 * i, 10 + 10 * n + i, 1_000_000 + 100_000 * n + 1000 * i)
        for n, iso in enumerate(WORLD_BANK_COUNTRIES)
        for i, year in enumerate(YEARS)
    ]
    seed_world_bank_cache(tmp_path, world_bank)
    pl.DataFrame(
        {"Date": [date(year, 1, 1) for year in YEARS], "co2": [float(350 + i) for i in range(len(YEARS))]}
    ).write_parquet(tmp_path / "co2.parquet")
    # A wave rather than a ramp, so STL has a trend to separate a deviation from.
    seed_ocean_heat_cache(
        tmp_path,
        pl.DataFrame(
            {"Date": [date(year, 1, 1) for year in YEARS], "Temp": [i + np.sin(i) for i in range(len(YEARS))]}
        ),
    )
    # A whole year of months, since only years the record covers in full survive the annual total.
    # They share the year's total evenly, so the annual series is the 100(n + 1) + 5i below.
    precipitation = [
        (iso, pd.Timestamp(f"{year}-{month:02d}-01"), (100.0 * (n + 1) + 5 * i) / 12)
        for n, iso in enumerate(PRECIPITATION_COUNTRIES)
        for i, year in enumerate(PRECIPITATION_YEARS)
        for month in range(1, 13)
    ]
    pd.DataFrame(precipitation, columns=["country_code", "time", "precip"]).set_index(
        ["country_code", "time"]
    ).to_parquet(tmp_path / GPCC_CACHE_FILE)
    return tmp_path


@pytest.fixture(scope="module")
def replication(wide_cache):
    """Built once: every test here only reads the panel."""
    return create_replication_data(wide_cache)


def row(frame: pl.DataFrame, iso: str, year: int) -> dict:
    """The one row for a country and year, as a plain mapping."""
    match = frame.filter((pl.col("ISO") == iso) & (pl.col("year") == date(year, 1, 1)))

    assert len(match) == 1, f"expected one row for {iso} {year}, got {len(match)}"
    return match.to_dicts()[0]


def test_the_frame_is_one_row_per_country_and_year(replication):
    assert not replication.select("ISO", "year").is_duplicated().any()


def test_only_countries_present_in_both_disaster_and_indicator_data_survive(replication):
    """CCC has no World Bank data and DDD no EM-DAT, so neither belongs in the panel."""
    assert sorted(replication["ISO"].unique()) == ["AAA", "BBB", "EEE"]


def test_the_published_columns_are_all_present(replication):
    """Downstream models select by name, so a dropped column is a silent regression."""
    expected = {
        "ISO",
        "year",
        "climatological_disasters",
        "hydrological_disasters",
        "population",
        "ln_population_density",
        "ln_gdp_pc",
        "square_ln_gdp_pc",
        "dev_from_trend_ocean_temp",
        "co2",
        "precip_deviation",
        "Total_Damage_Adjusted_hydro",
        "Total_Damage_Adjusted_clim",
        "Total_Affected_hydro",
        "ln_population_density_squared",
        "time_period",
    }

    assert set(replication.columns) == expected


def test_population_is_reported_in_millions(replication):
    """The regressor is in millions; the World Bank supplies raw headcounts."""
    assert row(replication, "AAA", SAMPLE_YEAR)["population"] == pytest.approx(1.001)


def test_the_gdp_and_density_logs_are_natural(replication):
    entry = row(replication, "AAA", SAMPLE_YEAR)

    assert entry["ln_gdp_pc"] == pytest.approx(np.log(1010))
    assert entry["ln_population_density"] == pytest.approx(np.log(11))


def test_both_squared_terms_square_their_own_log(replication):
    """One is written as a product and the other as a power; both must hold for every row."""
    known = replication.drop_nulls(["ln_gdp_pc", "ln_population_density"])

    assert len(known) > 0
    assert (known["square_ln_gdp_pc"] - known["ln_gdp_pc"] ** 2).abs().max() < 1e-12
    assert (known["ln_population_density_squared"] - known["ln_population_density"] ** 2).abs().max() < 1e-12


def test_the_time_trend_counts_years_over_a_century(replication):
    """A raw year would dwarf every other regressor; the trend is rescaled."""
    assert row(replication, "AAA", SAMPLE_YEAR)["time_period"] == pytest.approx((SAMPLE_YEAR - TREND_BASE_YEAR) / 100)


def test_a_country_with_no_climatological_damage_reports_none(replication):
    """A class with nothing recorded stays missing. Whether that should count as zero damage depends on
    the model reading it, so the panel does not decide.
    """
    entry = row(replication, "BBB", SAMPLE_YEAR)

    assert entry["Total_Damage_Adjusted_clim"] is None
    assert entry["Total_Damage_Adjusted_hydro"] is not None


def test_a_country_year_with_no_disasters_stays_missing(replication):
    """A country-year with no record stays missing; summing it as a zero would erase the distinction."""
    quiet = replication.filter(pl.col("year") == date(QUIET_YEAR, 1, 1))

    assert len(quiet) > 0
    assert quiet["hydrological_disasters"].is_null().all()


def test_damage_is_published_in_the_unit_em_dat_reports(replication):
    """EM-DAT reports thousands of US dollars. Rescaling it here would put every reader of the panel on
    a scale one model happened to want.
    """
    entry = row(replication, "AAA", SAMPLE_YEAR)

    assert entry["Total_Damage_Adjusted_hydro"] == pytest.approx(DAMAGE_THOUSANDS)
    assert entry["Total_Damage_Adjusted_clim"] == pytest.approx(DAMAGE_THOUSANDS)


def test_precipitation_deviation_is_measured_against_the_named_baseline_period(replication):
    """The baseline is a fixed calendar period, not whichever years the record happens to open with."""
    # AAA's precipitation runs 100 + 5i from 1955. The 1961-1990 window is i = 6..35 and averages
    # 202.5, so 1985 sits 47.5 above it; the record's first 30 years would average 172.5 and put it
    # at 77.5 instead.
    sample = replication.filter((pl.col("ISO") == "AAA") & (pl.col("year") == date(1985, 1, 1)))

    assert sample["precip_deviation"].to_list() == [pytest.approx(47.5)]


def test_a_baseline_the_record_does_not_cover_is_refused(wide_cache):
    """A baseline drawn from a handful of years would be published under the name of a full period."""
    with pytest.raises(ValueError, match="baseline"):
        create_replication_data(wide_cache, baseline=(1900, 1929))


def test_the_ocean_temperature_deviation_is_residual_around_its_trend(replication):
    """It is the STL residual, so it is centered near zero and not the level of Temp itself."""
    deviations = replication["dev_from_trend_ocean_temp"].drop_nulls()

    assert deviations.mean() == pytest.approx(0.0, abs=0.5)
    assert deviations.abs().max() < 5.0


def paneled(rows) -> pl.DataFrame:
    """A panel carrying only the key and the two features the tests below select on."""
    return pl.DataFrame(rows, schema=["ISO", "year", "ln_gdp_pc", "co2"], orient="row").with_columns(
        pl.col("year").cast(pl.Date)
    )


def bounded(isos) -> gpd.GeoDataFrame:
    """Country geometries, whose shape does not matter because only the codes are read."""
    return gpd.GeoDataFrame({"ISO_A3": list(isos)}, geometry=[Point(n, n) for n in range(len(isos))])


TWO_FEATURES = ["ln_gdp_pc", "co2"]


def test_a_row_missing_a_feature_cannot_enter_the_model():
    """A regressor matrix has no room for a hole, so the row leaves rather than the column."""
    panel = paneled([("AAA", date(2000, 1, 1), 1.0, 350.0), ("AAA", date(2001, 1, 1), None, 351.0)])

    rows, _ = model_frame(panel, bounded(["AAA"]), features=TWO_FEATURES)

    assert rows["year"].to_list() == [date(2000, 1, 1)]


def test_a_null_in_a_column_the_caller_did_not_name_keeps_its_row():
    """The panel carries damages that are legitimately missing, and dropping those rows would throw
    away the covariates a model still reads.
    """
    panel = paneled([("AAA", date(2000, 1, 1), 1.0, 350.0)]).with_columns(pl.lit(None).alias("damage"))

    rows, _ = model_frame(panel, bounded(["AAA"]), features=TWO_FEATURES)

    assert rows["damage"].to_list() == [None]


def test_a_country_with_no_geometry_leaves_the_panel():
    panel = paneled([("AAA", date(2000, 1, 1), 1.0, 350.0), ("BBB", date(2000, 1, 1), 2.0, 350.0)])

    rows, geometry = model_frame(panel, bounded(["AAA"]), features=TWO_FEATURES)

    assert rows["ISO"].to_list() == ["AAA"]
    assert geometry["ISO_A3"].to_list() == ["AAA"]


def test_a_country_with_no_complete_row_leaves_the_geometry():
    """Both sides are filtered, so a model indexing the two by position cannot misalign them."""
    panel = paneled([("AAA", date(2000, 1, 1), 1.0, 350.0), ("BBB", date(2000, 1, 1), None, 350.0)])

    rows, geometry = model_frame(panel, bounded(["AAA", "BBB"]), features=TWO_FEATURES)

    assert rows["ISO"].to_list() == ["AAA"]
    assert geometry["ISO_A3"].to_list() == ["AAA"]


def test_a_place_narrows_both_sides_to_its_members():
    panel = paneled(
        [
            ("AAA", date(2000, 1, 1), 1.0, 350.0),
            ("BBB", date(2000, 1, 1), 2.0, 350.0),
            ("CCC", date(2000, 1, 1), 3.0, 350.0),
        ]
    )

    rows, geometry = model_frame(panel, bounded(["CCC", "BBB", "AAA"]), isos=["AAA", "CCC"], features=TWO_FEATURES)

    assert rows["ISO"].to_list() == ["AAA", "CCC"]
    assert geometry["ISO_A3"].to_list() == ["AAA", "CCC"]


def test_a_requested_country_the_data_cannot_answer_for_is_refused():
    """Returning the members that did survive would fit a region quietly missing one of its countries."""
    panel = paneled([("AAA", date(2000, 1, 1), 1.0, 350.0)])

    with pytest.raises(ValueError, match=r"\['BBB'\]"):
        model_frame(panel, bounded(["AAA"]), isos=["AAA", "BBB"], features=TWO_FEATURES)


def test_two_sides_naming_no_country_in_common_are_refused():
    panel = paneled([("AAA", date(2000, 1, 1), 1.0, 350.0)])

    with pytest.raises(ValueError, match="No country"):
        model_frame(panel, bounded(["BBB"]), features=TWO_FEATURES)


def test_the_countries_reconciliation_drops_are_named(caplog):
    """Losing a country to a mismatch between two sources is the failure this project keeps hitting, so
    it is reported rather than left to be noticed in a row count.
    """
    panel = paneled([("AAA", date(2000, 1, 1), 1.0, 350.0), ("BBB", date(2000, 1, 1), 2.0, 350.0)])

    with caplog.at_level(logging.WARNING, logger="climate_risk.replication_data"):
        model_frame(panel, bounded(["AAA", "CCC"]), features=TWO_FEATURES)

    assert "BBB" in caplog.text
    assert "CCC" in caplog.text


def test_a_place_narrowing_the_result_is_not_reported_as_a_loss(caplog):
    """The warning above is worth reading only if it names countries nobody asked to drop. Reporting a
    single-country place would bury that under every other country in the world.
    """
    panel = paneled([("AAA", date(2000, 1, 1), 1.0, 350.0), ("BBB", date(2000, 1, 1), 2.0, 350.0)])

    with caplog.at_level(logging.WARNING, logger="climate_risk.replication_data"):
        model_frame(panel, bounded(["AAA", "BBB"]), isos=["AAA"], features=TWO_FEATURES)

    assert caplog.text == ""
