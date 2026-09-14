import datetime as dt

from functools import partial, reduce
from pathlib import Path

import pandas as pd
import polars as pl

from climate_risk.config.schema import EventFilters
from climate_risk.data.co2 import load_co2_data
from climate_risk.data.frequency import AGGREGATION_INTERVALS, MONTHS_PER_PERIOD, AggregationFrequency
from climate_risk.data.gpcc import load_gpcc_data
from climate_risk.data.ocean_heat import load_ocean_heat_data
from climate_risk.data.world_bank import load_wb_data
from climate_risk.data_functions.emdat_processing import (
    CLIMATOLOGICAL,
    HYDROMETEOROLOGICAL,
    count_events_by_type,
    country_grid,
    event_filter,
    load_emdat_events,
    total_damage,
)

# The panel is keyed on the EM-DAT column names, which the other sources are renamed onto.
PANEL_KEY = ["ISO", "date"]
SERIES_KEY = "date"

# The measures the panel carries split by disaster class. The rest of DAMAGE_VARS reaches it totalled
# across classes only.
CLASS_MEASURES = ("Total_Damage_Adjusted",)

# Annual precipitation is a sum, so a year the record only partly covers totals low. Dropping those
# keeps a part-year at the end of the record from entering the panel as a drought.


def _as_polars(frame: pd.DataFrame) -> pl.DataFrame:
    """Convert a pandas frame from the geospatial layer into the polars this module merges in."""
    return pl.from_pandas(frame.reset_index())


def _outer_join(left: pl.DataFrame, right: pl.DataFrame, on: list[str] | str) -> pl.DataFrame:
    """Join keeping every key from both sides, with the key columns merged rather than suffixed."""
    return left.join(right, on=on, how="full", coalesce=True)


def _left_join(left: pl.DataFrame, right: pl.DataFrame, on: list[str] | str) -> pl.DataFrame:
    """Join keeping only the left side's keys, which fixes the rows the panel spans."""
    return left.join(right, on=on, how="left", coalesce=True)


def _suffixed_damage(damage: pl.DataFrame, suffix: str) -> pl.DataFrame:
    """Tag one disaster class's measures so both classes can sit in one frame."""
    return damage.select(*PANEL_KEY, *(pl.col(name).alias(f"{name}_{suffix}") for name in CLASS_MEASURES))


def _combine_emdat(events: pl.DataFrame, grid: pl.DataFrame, counts: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split EM-DAT into counts and damages.

    Returns
    -------
    events : DataFrame
        Counts per country, period and disaster type.
    damage : DataFrame
        Totals per country and period, with each disaster class joined on as suffixed columns.
    """
    damage = total_damage(events, grid)
    for disaster_class, suffix in ((HYDROMETEOROLOGICAL, "hydro"), (CLIMATOLOGICAL, "clim")):
        by_class = total_damage(events.filter(pl.col("disaster_class") == disaster_class), grid)
        damage = damage.join(_suffixed_damage(by_class, suffix), on=PANEL_KEY, how="left")

    return counts.drop("Subregion"), damage


def _shape_world_bank(indicators: pl.DataFrame) -> pl.DataFrame:
    """Rename the World Bank columns onto the panel's key and date its years."""
    return indicators.select(
        pl.col("country_code").alias("ISO"),
        pl.date(pl.col("year"), 1, 1).alias("date"),
        pl.exclude("country_code", "year"),
    )


def _total_precipitation(
    gpcc: pl.DataFrame, *, frequency: AggregationFrequency = "annual"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Total monthly precipitation to periods.

    Periods the record covers only partly are dropped from both.

    Returns
    -------
    by_country : DataFrame
        One row per country and period.
    worldwide : DataFrame
        One row per period, summed across countries.
    """
    dated = gpcc.select(
        pl.col("country_code").alias("ISO"),
        pl.col("time").dt.truncate(AGGREGATION_INTERVALS[frequency]).dt.date().alias(SERIES_KEY),
        pl.col("time").dt.month().alias("month"),
        "precip",
    )

    whole_periods = (
        dated.group_by(SERIES_KEY)
        .agg(pl.col("month").n_unique().alias("months"))
        .filter(pl.col("months") == MONTHS_PER_PERIOD[frequency])
        .select(SERIES_KEY)
    )
    covered = dated.join(whole_periods, on=SERIES_KEY, how="inner")

    # Added in calendar order. polars gathers each group's rows in whatever order its threads
    # finish, and floating-point addition is not associative, so an unordered sum of the same
    # months lands on a different total from one run to the next.
    by_country = (
        covered.group_by("ISO", SERIES_KEY).agg(pl.col("precip").sort_by("month").sum()).sort("ISO", SERIES_KEY)
    )
    worldwide = by_country.group_by(SERIES_KEY).agg(pl.col("precip").sort_by("ISO").sum()).sort(SERIES_KEY)

    return by_country, worldwide


def _keyed_by_year(series: pl.DataFrame) -> pl.DataFrame:
    """Re-key an annual series onto the panel's year column. Both loaders date theirs to a year start."""
    return series.rename({"Date": SERIES_KEY}).sort(SERIES_KEY)


def _countries_in_common(*frames: pl.DataFrame) -> set[str]:
    """Return the ISO codes present in every frame."""
    return set.intersection(*(set(frame["ISO"].unique().to_list()) for frame in frames))


def _only_countries(frame: pl.DataFrame, codes: set[str]) -> pl.DataFrame:
    return frame.filter(pl.col("ISO").is_in(codes))


def total_precipitation(cache_dir: Path, *, frequency: AggregationFrequency = "annual") -> pl.DataFrame:
    """
    Total each country's precipitation to periods.

    Covers the whole GPCC record, which reaches back further than the EM-DAT panel does.

    Parameters
    ----------
    cache_dir : Path
        Directory the GPCC cache lives under.
    frequency : {'annual', 'quarterly', 'monthly'}, optional
        How long one period runs. Default ``'annual'``.

    Returns
    -------
    precipitation : DataFrame
        ``ISO``, ``date`` and ``precip``, one row per country and period.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk import total_precipitation

        precipitation = total_precipitation(Path("data"))
    """
    by_country, _ = _total_precipitation(_as_polars(load_gpcc_data(cache_dir)), frequency=frequency)

    return by_country


def build_time_series(cache_dir: Path) -> pl.DataFrame:
    """
    Merge the worldwide annual series into one frame.

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under.

    Returns
    -------
    series : DataFrame
        ``date``, CO2, ocean temperature and worldwide precipitation, one row per year.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk import build_time_series

        series = build_time_series(Path("data"))
    """
    _, worldwide = _total_precipitation(_as_polars(load_gpcc_data(cache_dir)))

    return reduce(
        partial(_outer_join, on=SERIES_KEY),
        [_keyed_by_year(load_co2_data(cache_dir)), _keyed_by_year(load_ocean_heat_data(cache_dir)), worldwide],
    ).sort(SERIES_KEY)


def build_country_panel(cache_dir: Path, *, frequency: AggregationFrequency = "annual") -> pl.DataFrame:
    """
    Merge events, damages, development indicators and precipitation onto one country-year row.

    Covers the years the event filter selects and the countries that have both a disaster record and
    development indicators. Precipitation reaches further back, and :func:`total_precipitation` has
    the whole record.

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under.
    frequency : {'annual', 'quarterly', 'monthly'}, optional
        How long one period of the panel runs. Default ``'annual'``.

    Returns
    -------
    panel : DataFrame
        One row per country and period, keyed on ``ISO`` and ``date``.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk import build_country_panel

        panel = build_country_panel(Path("data"))
    """
    filters = EventFilters()

    # Events are dated to their period once, and the grid opens on the filter's first year, so the
    # grid and the totals can disagree on neither the period a date falls in nor the years covered.
    # A grid period with no counts to join reads as null, which is what a quiet period reads as too.
    events_by_period = load_emdat_events(cache_dir).with_columns(
        pl.col("date").dt.truncate(AGGREGATION_INTERVALS[frequency])
    )
    selected = events_by_period.filter(event_filter(filters))
    grid = country_grid(events_by_period, window_start=dt.date(filters.start_year, 1, 1), frequency=frequency)

    events, damage = _combine_emdat(selected, grid, count_events_by_type(selected, grid))
    world_bank = _shape_world_bank(load_wb_data(cache_dir))
    precipitation = total_precipitation(cache_dir, frequency=frequency)

    # A country needs both a disaster record and development indicators to earn a row.
    common = _countries_in_common(damage, world_bank)
    events = _only_countries(events, common).drop("Region")
    damage = _only_countries(damage, common)
    world_bank = _only_countries(world_bank, common)

    # Left-joined onto the event grid, so the panel spans the filter's window and no more.
    return reduce(
        partial(_left_join, on=PANEL_KEY),
        [events, damage, world_bank, precipitation],
    ).sort(PANEL_KEY)
