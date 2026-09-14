from datetime import date, datetime

import polars as pl
import pytest

from climate_risk.config.schema import EventFilters
from climate_risk.data_functions.combine_data import (
    _total_precipitation,
    build_country_panel,
    build_time_series,
    total_precipitation,
)
from tests.conftest import emdat_event, write_merge_cache


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    """Built once: every test here only reads the merge, and rebuilding it per test dominated them."""
    return write_merge_cache(tmp_path_factory.mktemp("merge"))


@pytest.fixture(scope="module")
def panel(cache_dir):
    return build_country_panel(cache_dir)


@pytest.fixture(scope="module")
def time_series(cache_dir):
    return build_time_series(cache_dir)


def test_a_country_missing_from_either_disaster_source_is_dropped(panel):
    """CCC has no World Bank data and DDD no EM-DAT data, so neither belongs in the panel."""
    assert sorted(panel["ISO"].unique()) == ["AAA", "BBB", "EEE"]


def test_a_country_without_precipitation_keeps_its_row(panel):
    """EEE has everything but rainfall; dropping it would lose its disasters along with its precip."""
    assert "EEE" in panel["ISO"].to_list()
    assert panel.filter(pl.col("ISO") == "EEE")["precip"].is_null().all()


def test_the_precipitation_record_outlives_the_panel(cache_dir, panel):
    """FFF has only rainfall, so the panel drops it — the record itself still has to carry it."""
    assert "FFF" not in panel["ISO"].to_list()
    assert "FFF" in total_precipitation(cache_dir)["ISO"].to_list()


def test_the_panel_spans_the_full_country_grid(panel):
    countries = panel["ISO"].n_unique()
    years = panel["date"].n_unique()

    assert len(panel) == countries * years
    assert not panel.select("ISO", "date").is_duplicated().any()


def test_a_country_year_with_no_indicators_still_reaches_the_panel(panel):
    """The panel spans the event filter's window; the indicators cover 1990-91 alone."""
    early = panel.filter(pl.col("date") == date(1985, 1, 1))

    assert len(early) == panel["ISO"].n_unique()
    assert early["gdp_per_cap_usd"].is_null().all()


def test_the_panel_opens_on_the_year_the_event_filter_does(panel):
    """A grid opening before the counts do fills the gap with nulls no reader can tell from real ones."""
    assert panel["date"].min() == date(EventFilters().start_year, 1, 1)


def test_repartitioning_the_panel_moves_no_events(cache_dir):
    """A finer grid slices the same events differently. It must not create, drop or misplace any."""
    annual = build_country_panel(cache_dir)
    monthly = build_country_panel(cache_dir, frequency="monthly")

    assert monthly["Flood"].sum() == annual["Flood"].sum()
    assert monthly["date"].dt.month().n_unique() == 12

    # AAA's landslide starts in September 1991, and only there.
    landslides = monthly.filter((pl.col("ISO") == "AAA") & (pl.col("Mass movement (wet)") == 1))
    assert landslides["date"].to_list() == [date(1991, 9, 1)]


def test_precipitation_is_totalled_over_the_year_not_averaged(cache_dir):
    """GPCC publishes monthly; the panel wants the year's total rainfall, not a monthly mean."""
    annual = total_precipitation(cache_dir).filter((pl.col("ISO") == "AAA") & (pl.col("date") == date(1990, 1, 1)))

    # AAA's 1990 months run 101..112, totalling 1278 against a monthly mean of 106.5.
    assert annual["precip"].to_list() == [pytest.approx(1278.0)]


def test_a_year_the_record_only_partly_covers_is_dropped():
    """The near-real-time product ends mid-year, and a part-year total reads as a drought, not a gap."""
    monthly = pl.DataFrame(
        {
            "country_code": ["AAA"] * 15,
            "time": [datetime(2020, month, 1) for month in range(1, 13)]
            + [datetime(2021, month, 1) for month in range(1, 4)],
            "precip": [10.0] * 15,
        }
    )

    by_country, worldwide = _total_precipitation(monthly)

    assert by_country["date"].to_list() == [date(2020, 1, 1)]
    assert worldwide["date"].to_list() == [date(2020, 1, 1)]


def test_a_monthly_panel_carries_each_month_of_rain_on_its_own_row(cache_dir):
    """The panel builder has to hand its frequency to the precipitation totals, or every month but January reads null."""
    monthly = build_country_panel(cache_dir, frequency="monthly")
    june = monthly.filter((pl.col("ISO") == "AAA") & (pl.col("date") == date(1990, 6, 1)))

    # AAA's 1990 months run 101..112, so June alone is 106.
    assert june["precip"].to_list() == [pytest.approx(106.0)]


def test_precipitation_totals_to_the_quarter_and_drops_a_partial_one():
    """Three months make a quarter, and two months of the next year do not."""
    monthly = pl.DataFrame(
        {
            "country_code": ["AAA"] * 14,
            "time": [datetime(2020, month, 1) for month in range(1, 13)] + [datetime(2021, 1, 1), datetime(2021, 2, 1)],
            "precip": [float(month) for month in range(1, 13)] + [50.0, 50.0],
        }
    )

    by_country, _ = _total_precipitation(monthly, frequency="quarterly")

    assert by_country["date"].to_list() == [date(2020, 1, 1), date(2020, 4, 1), date(2020, 7, 1), date(2020, 10, 1)]
    assert by_country["precip"].to_list() == [
        pytest.approx(6.0),
        pytest.approx(15.0),
        pytest.approx(24.0),
        pytest.approx(33.0),
    ]


def test_the_worldwide_series_totals_every_country(time_series):
    """It feeds a country-invariant regressor, so it sums across countries rather than averaging."""
    nineteen_ninety = time_series.filter(pl.col("date") == date(1990, 1, 1))

    assert nineteen_ninety["precip"].to_list() == [pytest.approx(1278.0 + 2478.0 + 3678.0)]


def test_the_time_series_carries_no_country(time_series):
    """The aggregate series feed country-invariant regressors, so an ISO level would broadcast wrong."""
    assert time_series.columns[0] == "date"
    assert "ISO" not in time_series.columns
    assert {"co2", "Temp", "precip"} <= set(time_series.columns)


def test_every_disaster_type_gets_a_column_even_when_unobserved(write_emdat_cache, write_full_cache):
    """Unstacking yields a column per observed type, so downstream code naming all of them breaks."""
    cache_dir = write_full_cache()
    write_emdat_cache(
        emdat_event({"ISO": iso, "DisNo.": f"{iso}-{year}", "Start Year": year, "Disaster Type": "Drought"})
        for iso in ("AAA", "BBB")
        for year in (1990, 1991)
    )

    events = build_country_panel(cache_dir)

    assert {"Drought", "Flood", "Storm", "Wildfire", "Extreme temperature"} <= set(events.columns)
    assert events["Wildfire"].is_null().all()


def test_damage_columns_survive_a_class_with_no_events(write_emdat_cache, write_full_cache):
    """An empty pivot emits no columns at all, so a country with only floods loses the clim split."""
    cache_dir = write_full_cache()
    write_emdat_cache(
        emdat_event({"ISO": iso, "DisNo.": f"{iso}-{year}", "Start Year": year, "Disaster Type": "Flood"})
        for iso in ("AAA", "BBB")
        for year in (1990, 1991)
    )

    damage = build_country_panel(cache_dir)

    assert "Total_Damage_Adjusted_clim" in damage.columns
    assert damage["Total_Damage_Adjusted_clim"].is_null().all()


def test_only_the_measures_with_a_reader_are_split_by_class(panel):
    """Both splits carry the same variable names, so they collide unless suffixed apart. Suffixing every
    measure costs one line and lands eleven columns nobody selects.

    Stated literally rather than derived from CLASS_MEASURES, so widening that constant has to be a
    decision someone writes down here too.
    """
    split = {column for column in panel.columns if column.endswith(("_hydro", "_clim"))}

    assert split == {"Total_Damage_Adjusted_hydro", "Total_Damage_Adjusted_clim"}


def test_world_bank_years_become_timestamps(panel):
    """The panel joins on date, which EM-DAT supplies as a timestamp."""
    assert panel.schema["date"] == pl.Date
