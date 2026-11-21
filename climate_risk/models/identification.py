"""
EXPERIMENTAL panel helpers for the CO2-identification tests (tasks/2026-11-21-climate-identification-tests.md).

Nothing here changes how the published panel or models behave. Each helper takes the finished panel
from :func:`climate_risk.replication_data.create_replication_data` and the EM-DAT event table, and
returns a copy of the panel with one count column recomputed or added.

The convention is the panel's own: a country-year with no recorded event of the counted types reads as
null, not zero. The frequency models zero-fill a null target themselves, so null here means "no record",
which is the reporting assumption the panel already makes (see ``_counted_or_missing``).
"""

from collections.abc import Sequence

import polars as pl

from climate_risk.config.schema import EventFilters
from climate_risk.data_functions.emdat_processing import event_filter
from climate_risk.replication_data import HYDROLOGICAL_TYPES, PANEL_KEY

# The EM-DAT types used as a control group: events with no pathway from a changing climate, but with
# the same reporting history as the climate hazards. Deliberately not added to ``DISASTER_CLASSES``.
GEOPHYSICAL_TYPES = ("Earthquake", "Volcanic activity", "Mass movement (dry)")

GEOPHYSICAL_COLUMN = "geophysical_disasters"


def recount_events(
    panel: pl.DataFrame,
    events: pl.DataFrame,
    types: Sequence[str],
    *,
    column: str,
    filters: EventFilters | None = None,
) -> pl.DataFrame:
    """
    Return the panel with ``column`` holding the count of ``types`` events under ``filters``.

    The panel's rows are kept exactly as they are, so a fit on the new column sees the same
    country-years as a fit on the published one.

    Parameters
    ----------
    panel : DataFrame
        The annual replication panel, keyed on ``ISO`` and ``date``.
    events : DataFrame
        EM-DAT events from ``load_emdat_events``, carrying ``ISO``, ``date``, ``Disaster Type``,
        ``Deaths`` and ``Total_Affected``.
    types : sequence of str
        The ``Disaster Type`` values to count.
    column : str
        Name of the count column. An existing column of that name is replaced in place.
    filters : EventFilters, optional
        Which events count. Default ``EventFilters()``, the thresholds the published panel uses.

    Returns
    -------
    recounted : DataFrame
        The panel with ``column`` set. A country-year with no matching event is null.
    """
    counted = (
        events.with_columns(pl.col("date").dt.truncate("1y"))
        .filter(event_filter(filters or EventFilters()) & pl.col("Disaster Type").is_in(list(types)))
        .group_by(PANEL_KEY)
        .len()
        .select(*PANEL_KEY, pl.col("len").cast(pl.Float64).alias(column))
    )

    # An existing column keeps its position; a new one goes on the end.
    order = panel.columns if column in panel.columns else [*panel.columns, column]

    return panel.drop(column, strict=False).join(counted, on=PANEL_KEY, how="left").select(order)


def with_geophysical_disasters(panel: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """
    Add ``geophysical_disasters``: earthquakes, volcanic activity and dry mass movements per country-year.

    Same event filters and same null-means-no-record convention as the published counts, so it can
    stand in for the target in the placebo and joint models.

    Parameters
    ----------
    panel : DataFrame
        The annual replication panel.
    events : DataFrame
        EM-DAT events from ``load_emdat_events``.

    Returns
    -------
    extended : DataFrame
        The panel with one more column, ``geophysical_disasters``, as a float count.
    """
    return recount_events(panel, events, GEOPHYSICAL_TYPES, column=GEOPHYSICAL_COLUMN)


def with_high_impact_hydro(panel: pl.DataFrame, events: pl.DataFrame, *, min_deaths: int) -> pl.DataFrame:
    """
    Replace ``hydrological_disasters`` by the count of events that killed more than ``min_deaths`` people.

    Big events are the ones least likely to depend on reporting effort, so this is the Test 3 target.

    Parameters
    ----------
    panel : DataFrame
        The annual replication panel.
    events : DataFrame
        EM-DAT events from ``load_emdat_events``.
    min_deaths : int
        An event must kill more than this many people to count.

    Returns
    -------
    high_impact : DataFrame
        The panel with ``hydrological_disasters`` recounted under the death threshold.
    """
    return recount_events(
        panel,
        events,
        HYDROLOGICAL_TYPES,
        column="hydrological_disasters",
        filters=EventFilters(min_deaths=min_deaths),
    )
