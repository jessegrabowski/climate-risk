import datetime as dt

import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
import pytest

from climate_risk.models.frequency import FEATURES, TARGET, build_year_effect_frequency_model
from climate_risk.models.identification import (
    GEOPHYSICAL_COLUMN,
    HAMILTON_COLUMN,
    build_joint_year_effect_model,
    build_year_effect_model,
    hamilton_shock,
    with_geophysical_disasters,
    with_hamilton_temperature_shock,
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


def model_panel(seed: int = 0, countries: int = 6, years: int = 8) -> pl.DataFrame:
    """A small panel carrying every column the models read, with a trend-free random covariate set."""
    rng = np.random.default_rng(seed)
    n = countries * years
    data: dict[str, object] = {feature: rng.normal(size=n) for feature in FEATURES if feature != "ln_population"}
    data["population"] = rng.lognormal(2.0, 1.0, size=n)
    data["co2"] = np.tile(np.linspace(350.0, 380.0, years), countries)
    data[TARGET] = rng.poisson(2, size=n).astype(float)
    data[GEOPHYSICAL_COLUMN] = rng.poisson(0.3, size=n).astype(float)
    data["ISO"] = np.repeat([f"C{i}" for i in range(countries)], years)
    data["date"] = [dt.date(1990 + year, 1, 1) for _ in range(countries) for year in range(years)]

    return pl.DataFrame(data).with_columns(pl.col("date").cast(pl.Date))


def test_hamilton_shock_is_zero_when_a_series_follows_its_own_lags_exactly():
    y = [1.0, 2.0, 1.5]
    for t in range(3, 30):
        y.append(1.0 + 0.5 * y[t - 2] + 0.2 * y[t - 3])
    shock = hamilton_shock(pd.Series(y, index=np.arange(1960, 1960 + len(y))))

    assert shock.index[0] == 1963
    assert np.abs(shock).max() < 1e-9


def test_hamilton_shock_picks_out_a_one_off_jump():
    y = np.sin(np.arange(40) / 3.0)
    y[25] += 3.0
    shock = hamilton_shock(pd.Series(y, index=np.arange(1960, 2000)))

    assert shock.abs().idxmax() == 1960 + 25


def test_hamilton_temperature_shock_adds_one_column_and_is_null_before_the_first_shock_year():
    panel = model_panel()
    temperature = pl.DataFrame(
        {"date": [dt.date(year, 1, 1) for year in range(1985, 2000)], "Temp": np.sin(np.arange(15) / 2.0).tolist()}
    )
    out = with_hamilton_temperature_shock(panel, temperature)

    assert out.columns == [*panel.columns, HAMILTON_COLUMN]
    assert out.select(panel.columns).equals(panel)
    early = out.filter(pl.col("date").dt.year() < 1988)
    assert early[HAMILTON_COLUMN].null_count() == early.height
    assert out.filter(pl.col("date").dt.year() >= 1988)[HAMILTON_COLUMN].null_count() == 0


def test_the_configurable_year_effect_model_matches_the_published_one_at_default_settings():
    panel = model_panel()
    published = build_year_effect_frequency_model(panel)
    configurable = build_year_effect_model(panel)

    assert sorted(published.named_vars) == sorted(configurable.named_vars)
    point = published.initial_point()
    assert configurable.compile_logp()(point) == pytest.approx(published.compile_logp()(point))


def test_dropping_co2_removes_its_coefficient_and_still_builds():
    model = build_year_effect_model(model_panel(), [f for f in FEATURES if f != "co2"])

    assert "co2" not in list(model.coords["feature"])


def test_the_joint_model_fixes_the_geophysical_co2_slope_at_zero_only_when_asked():
    panel = model_panel()
    fixed = build_joint_year_effect_model(panel, co2_on_geophysical=False)
    free = build_joint_year_effect_model(panel, co2_on_geophysical=True)
    co2 = FEATURES.index("co2")

    assert np.all(pm.draw(fixed["beta_used"], draws=5, random_seed=1)[:, co2, 1] == 0.0)
    assert np.all(pm.draw(fixed["beta_used"], draws=5, random_seed=1)[:, co2, 0] != 0.0)
    assert np.all(pm.draw(free["beta_used"], draws=5, random_seed=1)[:, co2, 1] != 0.0)
    assert np.isfinite(fixed.compile_logp()(fixed.initial_point()))


def test_one_shared_walk_or_one_per_class():
    panel = model_panel()
    shared = build_joint_year_effect_model(panel, co2_on_geophysical=False)
    separate = build_joint_year_effect_model(panel, co2_on_geophysical=False, shared_year_effect=False)

    assert pm.draw(shared["year_effect"], draws=2).shape[-1] == 1
    assert pm.draw(separate["year_effect"], draws=2).shape[-1] == 2
    assert np.isfinite(separate.compile_logp()(separate.initial_point()))
