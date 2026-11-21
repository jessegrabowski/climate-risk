import datetime as dt

import polars as pl

from climate_risk.models.identification import (
    GEOPHYSICAL_COLUMN,
    with_geophysical_disasters,
    with_high_impact_hydro,
)


def toy_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ISO": ["AAA", "AAA", "BBB", "BBB"],
            "date": [dt.date(1990, 1, 1), dt.date(1991, 1, 1)] * 2,
            "hydrological_disasters": [2.0, None, 1.0, None],
            "co2": [350.0, 352.0, 350.0, 352.0],
        }
    )


def toy_events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ISO": ["AAA", "AAA", "AAA", "BBB", "AAA"],
            "date": [
                dt.date(1990, 3, 1),
                dt.date(1990, 9, 1),
                dt.date(1991, 5, 1),
                dt.date(1990, 2, 1),
                dt.date(1990, 6, 1),
            ],
            "Disaster Type": ["Flood", "Flood", "Earthquake", "Storm", "Volcanic activity"],
            "Deaths": [150.0, 5.0, 10.0, 200.0, None],
            "Total_Affected": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def test_geophysical_counts_land_on_the_right_country_year_and_leave_the_rest_null():
    out = with_geophysical_disasters(toy_panel(), toy_events())

    assert out.columns == [*toy_panel().columns, GEOPHYSICAL_COLUMN]
    assert out[GEOPHYSICAL_COLUMN].to_list() == [1.0, 1.0, None, None]


def test_adding_the_geophysical_column_does_not_change_any_existing_column():
    before = toy_panel()
    after = with_geophysical_disasters(before, toy_events()).select(before.columns)

    assert after.equals(before)


def test_high_impact_hydro_keeps_only_events_above_the_death_threshold_and_keeps_column_order():
    out = with_high_impact_hydro(toy_panel(), toy_events(), min_deaths=100)

    assert out.columns == toy_panel().columns
    assert out["hydrological_disasters"].to_list() == [1.0, None, 1.0, None]


def test_an_event_with_unrecorded_deaths_fails_a_death_threshold():
    events = toy_events().with_columns(
        pl.lit("Flood").alias("Disaster Type"), pl.lit(None, dtype=pl.Float64).alias("Deaths")
    )
    out = with_high_impact_hydro(toy_panel(), events, min_deaths=0)

    assert out["hydrological_disasters"].null_count() == out.height
